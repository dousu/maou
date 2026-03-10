"""ゲームツリー可視化のインターフェース層．

App層のGameTreeQueryとInfra層のGradio UIを接続するアダプタ．
Cytoscape.js用のデータ変換や盤面SVG生成を担当する．
"""

from __future__ import annotations

import copy
import csv
import io
import logging
from typing import Any, NamedTuple

import polars as pl

from maou.app.game_tree.query import GameTreeQuery
from maou.domain.board.shogi import (
    Board,
    move_drop_hand_piece,
    move_from,
    move_is_drop,
    move_to,
    move_to_usi,
)
from maou.domain.game_tree.openings import (
    OpeningDatabase,
    OpeningInfo,
)
from maou.domain.visualization.board_renderer import (
    BoardPosition,
    MoveArrow,
    SVGBoardRenderer,
)
from maou.domain.visualization.piece_mapping import (
    get_piece_name_ja,
)

logger = logging.getLogger(__name__)

#: _get_board_for_position のキャッシュ型
_BoardCache = tuple[int, Board | None]


class MoveRow(NamedTuple):
    """指し手一覧テーブルの1行を表す．"""

    japanese: str
    """日本語表記の指し手(例: "7六歩")"""
    probability: str
    """選択確率(例: "60.0%")"""
    win_rate: str
    """勝率(例: "52.0%")"""
    child_hash: str
    """子局面のZobrist hash文字列"""


class GameTreeVisualizationInterface:
    """ゲームツリー可視化のインターフェース層．

    GameTreeQueryのデータをUI向けに変換する．
    """

    _ROW_MAP: dict[str, str] = {
        "a": "一",
        "b": "二",
        "c": "三",
        "d": "四",
        "e": "五",
        "f": "六",
        "g": "七",
        "h": "八",
        "i": "九",
    }

    _DROP_PIECE_MAP: dict[str, str] = {
        "P": "歩",
        "L": "香",
        "N": "桂",
        "S": "銀",
        "G": "金",
        "B": "角",
        "R": "飛",
    }

    def __init__(
        self,
        nodes_df: pl.DataFrame,
        edges_df: pl.DataFrame,
        initial_sfen: str | None = None,
    ) -> None:
        """初期化．

        Args:
            nodes_df: ノードデータ(nodes.feather相当)
            edges_df: エッジデータ(edges.feather相当)
            initial_sfen: 開始局面のSFEN文字列．Noneの場合は平手初期局面．

        Raises:
            ValueError: depth=0のルートノードが見つからない場合
        """
        self._query = GameTreeQuery(nodes_df, edges_df)
        root_nodes = nodes_df.filter(nodes_df["depth"] == 0)
        if len(root_nodes) == 0:
            msg = "ルートノード(depth=0)が見つかりません"
            raise ValueError(msg)
        if len(root_nodes) > 1:
            logger.warning(
                "複数のルートノード(depth=0)が存在します: %d件．"
                "最初のノードを使用します",
                len(root_nodes),
            )
        self._root_hash = int(
            root_nodes["position_hash"].item(0)
        )
        self._initial_sfen = initial_sfen
        self._renderer = SVGBoardRenderer()
        self._board_cache: _BoardCache | None = None
        self._opening_db = OpeningDatabase()

    def get_cytoscape_elements(
        self,
        root_hash: int,
        display_depth: int,
        min_probability: float,
    ) -> dict[str, list[dict[str, Any]]]:
        """Cytoscape.js用のノード・エッジデータを生成する．

        Args:
            root_hash: 表示するサブツリーのルートhash
            display_depth: 表示深さ
            min_probability: エッジの最小確率閾値

        Returns:
            {"nodes": [...], "edges": [...]} 形式のCytoscape elements
        """
        sub_nodes, sub_edges = self._query.get_subtree(
            root_hash, display_depth, min_probability
        )

        # エッジから各ノードへの親エッジ情報を取得(ラベル用)
        child_edge_map = self._build_child_edge_map(sub_edges)

        # サブツリー内の盤面をdepth順に漸進的に構築する．
        # get_path_to_root を毎回呼ぶ代わりに，親の盤面から
        # 1手適用して子の盤面を得る．
        local_boards = self._build_boards_incrementally(
            root_hash, sub_nodes, child_edge_map
        )

        cy_nodes: list[dict[str, Any]] = []
        for row in sub_nodes.iter_rows(named=True):
            pos_hash = row["position_hash"]
            edge_info = child_edge_map.get(pos_hash)

            label = "ROOT"
            probability = 1.0
            if edge_info is not None:
                label = self._move16_to_japanese(
                    edge_info["move16"],
                    local_boards.get(edge_info["parent_hash"]),
                )
                probability = edge_info["probability"]

            cy_nodes.append(
                {
                    "data": {
                        "id": str(pos_hash),
                        "label": label,
                        "result_value": float(
                            row["result_value"]
                        ),
                        "depth": int(row["depth"]),
                        "probability": float(probability),
                        "num_branches": int(
                            row["num_branches"]
                        ),
                        "is_depth_cutoff": bool(
                            row["is_depth_cutoff"]
                        ),
                    }
                }
            )

        cy_edges: list[dict[str, Any]] = []
        for row in sub_edges.iter_rows(named=True):
            move_label = self._move16_to_japanese(
                row["move16"],
                local_boards.get(row["parent_hash"]),
            )
            prob_label = f"{row['probability'] * 100:.1f}%"
            cy_edges.append(
                {
                    "data": {
                        "source": str(row["parent_hash"]),
                        "target": str(row["child_hash"]),
                        "label": f"{move_label} {prob_label}",
                        "probability": float(
                            row["probability"]
                        ),
                    }
                }
            )

        return {"nodes": cy_nodes, "edges": cy_edges}

    def get_board_svg(
        self,
        position_hash: int,
    ) -> str:
        """指定局面の盤面SVGを生成する．

        ルートからのパスを辿って盤面を復元し，SVGを生成する．

        Args:
            position_hash: 対象ノードのZobrist hash

        Returns:
            SVG文字列
        """
        path = self._query.get_path_to_root(position_hash)
        if not path:
            return "<p>盤面を復元できません</p>"

        if len(path) >= 2:
            # 親の盤面を復元し，最後の指し手の矢印を計算してから適用
            parent_board = self._reconstruct_board_from_path(
                path[:-1]
            )
            if parent_board is None:
                return "<p>盤面を復元できません</p>"

            edge = self._query.get_edge_between(
                path[-2], path[-1]
            )
            if edge is None:
                return "<p>盤面を復元できません</p>"

            move16 = edge["move16"]
            move = parent_board.get_move_from_move16(move16)
            move_arrow = self._move_to_arrow(move)
            parent_board.push_move(move)
            board = parent_board
        else:
            board = self._reconstruct_board_from_path(path)
            if board is None:
                return "<p>盤面を復元できません</p>"
            move_arrow = None

        # 盤面からBoardPositionを生成
        board_id_positions = board.get_board_id_positions()
        black_hand, white_hand = board.get_pieces_in_hand()
        pieces_in_hand = list(black_hand) + list(white_hand)
        position = BoardPosition(
            board_id_positions=board_id_positions,
            pieces_in_hand=pieces_in_hand,
        )

        turn = board.get_turn()

        return self._renderer.render(
            position=position,
            turn=turn,
            move_arrow=move_arrow,
        )

    def get_node_stats(
        self, position_hash: int
    ) -> dict[str, str]:
        """指定局面の統計情報を取得する．

        Args:
            position_hash: 対象ノードのZobrist hash

        Returns:
            局面統計の辞書
        """
        detail = self._query.get_node_detail(position_hash)
        if not detail:
            return {}

        stats: dict[str, str] = {
            "Zobrist Hash": str(detail["position_hash"]),
            "勝率": f"{detail['result_value'] * 100:.1f}%",
            "最善手勝率": f"{detail['best_move_win_rate'] * 100:.1f}%",
            "深さ": str(detail["depth"]),
            "分岐数": str(detail["num_branches"]),
        }

        opening = self.get_opening_name(position_hash)
        if opening is not None:
            stats["定跡"] = (
                f"{opening.name}({opening.category})"
            )

        return stats

    def get_move_table(
        self, position_hash: int
    ) -> list[MoveRow]:
        """指定局面の指し手一覧テーブルを生成する．

        Args:
            position_hash: 対象ノードのZobrist hash

        Returns:
            MoveRow のリスト(確率降順)．
            child_hashはUI上で非表示にされるが，行選択時のノード遷移に使用する．
        """
        children = self._query.get_children(position_hash)
        if len(children) == 0:
            return []

        board = self._get_board_for_position(position_hash)

        result: list[MoveRow] = []
        for row in children.iter_rows(named=True):
            japanese = self._move16_to_japanese(
                row["move16"], board
            )
            prob = f"{row['probability'] * 100:.1f}%"
            wr = f"{row['win_rate'] * 100:.1f}%"
            result.append(
                MoveRow(
                    japanese, prob, wr, str(row["child_hash"])
                )
            )

        return result

    def get_counts(self) -> tuple[int, int]:
        """ノード数とエッジ数を返す．

        Returns:
            (ノード数, エッジ数)
        """
        return (
            len(self._query.nodes_df),
            len(self._query.edges_df),
        )

    def get_root_hash(self) -> int:
        """ツリーのルートハッシュを返す．

        Returns:
            ルートノードのposition_hash
        """
        return self._root_hash

    def get_initial_sfen(self) -> str:
        """開始局面のSFEN文字列を返す．

        Returns:
            SFEN文字列．平手初期局面の場合は "startpos"
        """
        if self._initial_sfen:
            return self._initial_sfen
        return "startpos"

    def get_analytics_data(
        self, position_hash: int
    ) -> dict[str, Any]:
        """分岐分析用のデータを取得する．

        Args:
            position_hash: 対象ノードのZobrist hash

        Returns:
            上位指し手の確率・勝率データ
        """
        children = self._query.get_children(position_hash)
        if len(children) == 0:
            return {
                "moves": [],
                "probabilities": [],
                "win_rates": [],
            }

        board = self._get_board_for_position(position_hash)

        top_children = children.head(10)
        moves: list[str] = []
        probs: list[float] = []
        win_rates: list[float] = []

        for row in top_children.iter_rows(named=True):
            moves.append(
                self._move16_to_japanese(row["move16"], board)
            )
            probs.append(float(row["probability"]))
            win_rates.append(float(row["win_rate"]))

        return {
            "moves": moves,
            "probabilities": probs,
            "win_rates": win_rates,
        }

    def get_breadcrumb_data(
        self, position_hash: int
    ) -> list[dict[str, str]]:
        """パンくずリスト用のデータを取得する．

        ルートから指定ノードまでのパスを辿り，
        各ステップの指し手を日本語表記で返す．

        Args:
            position_hash: 対象ノードのZobrist hash

        Returns:
            [{"hash": "...", "label": "..."}, ...] 形式のリスト
        """
        path = self._query.get_path_to_root(position_hash)
        if not path:
            return []

        result: list[dict[str, str]] = [
            {"hash": str(path[0]), "label": "初期局面"}
        ]

        board = Board()
        if self._initial_sfen is not None:
            board.set_sfen(self._initial_sfen)

        for i in range(len(path) - 1):
            edge = self._query.get_edge_between(
                path[i], path[i + 1]
            )
            if edge is None:
                logger.warning(
                    "パンくずリスト: ノード %d → %d 間の"
                    "エッジが見つかりません",
                    path[i],
                    path[i + 1],
                )
                break

            move16 = edge["move16"]
            label = self._move16_to_japanese(move16, board)

            move = board.get_move_from_move16(move16)
            board.push_move(move)

            result.append(
                {"hash": str(path[i + 1]), "label": label}
            )

        return result

    def _get_moves_to_position(
        self, position_hash: int
    ) -> tuple[list[int], list[str]]:
        """ルートから指定局面までのパスとUSI指し手列を取得する．

        Args:
            position_hash: 対象ノードのZobrist hash

        Returns:
            (パスのハッシュ列, USI指し手列) のタプル．
            パスが存在しない場合は空リストのタプル．
        """
        path = self._query.get_path_to_root(position_hash)
        if not path:
            return [], []

        moves: list[str] = []
        for i in range(len(path) - 1):
            edge = self._query.get_edge_between(
                path[i], path[i + 1]
            )
            if edge is None:
                logger.warning(
                    "パス中のエッジが欠損: %d → %d",
                    path[i],
                    path[i + 1],
                )
                return [], []
            moves.append(move_to_usi(edge["move16"]))

        return path, moves

    def get_opening_name(
        self, position_hash: int
    ) -> OpeningInfo | None:
        """指定局面の定跡名を検索する．

        ルートからの指し手列を定跡データベースと照合し，
        一致するパターンがあれば定跡情報を返す．

        Args:
            position_hash: 対象ノードのZobrist hash

        Returns:
            一致した定跡の情報．見つからない場合None．
        """
        path, moves = self._get_moves_to_position(position_hash)
        if len(path) < 2:
            return None

        return self._opening_db.find_opening(moves)

    def export_sfen_path(self, position_hash: int) -> str:
        """指定局面までのUSI position文字列を生成する．

        将棋エンジンで使用可能な ``position`` コマンド形式で出力する．

        Args:
            position_hash: 対象ノードのZobrist hash

        Returns:
            USI position文字列
            (例: "position startpos moves 7g7f 3c3d")
        """
        _, moves = self._get_moves_to_position(position_hash)

        if self._initial_sfen is not None:
            base = f"position sfen {self._initial_sfen}"
        else:
            base = "position startpos"

        if moves:
            return f"{base} moves {' '.join(moves)}"
        return base

    def export_subtree_csv(
        self,
        root_hash: int,
        max_depth: int = 3,
        min_probability: float = 0.01,
    ) -> str:
        """サブツリーの指し手統計をCSV形式で出力する．

        Args:
            root_hash: サブツリーのルートhash
            max_depth: 最大深さ
            min_probability: 最小確率閾値

        Returns:
            CSV形式の文字列
        """
        sub_nodes, sub_edges = self._query.get_subtree(
            root_hash, max_depth, min_probability
        )

        # ノードとエッジを結合して各エッジの情報を出力
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "parent_hash",
                "child_hash",
                "move",
                "probability",
                "win_rate",
                "result_value",
                "depth",
            ]
        )

        # 親ノードごとにソートして出力
        edges_sorted = sub_edges.sort(
            ["parent_hash", "probability"],
            descending=[False, True],
        )

        # child_hash → result_value / depth のマップ
        node_map: dict[int, float] = {}
        node_depth_map: dict[int, int] = {}
        for row in sub_nodes.iter_rows(named=True):
            node_map[row["position_hash"]] = row["result_value"]
            node_depth_map[row["position_hash"]] = row["depth"]

        # 盤面を構築してUSI→日本語変換
        child_edge_map = self._build_child_edge_map(sub_edges)

        local_boards = self._build_boards_incrementally(
            root_hash, sub_nodes, child_edge_map
        )

        for row in edges_sorted.iter_rows(named=True):
            move_ja = self._move16_to_japanese(
                row["move16"],
                local_boards.get(row["parent_hash"]),
            )

            child_hash = row["child_hash"]
            result_value = node_map.get(
                child_hash, float("nan")
            )
            depth = node_depth_map.get(child_hash, -1)

            writer.writerow(
                [
                    str(row["parent_hash"]),
                    str(child_hash),
                    move_ja,
                    f"{row['probability']:.4f}",
                    f"{row['win_rate']:.4f}",
                    f"{result_value:.4f}",
                    depth,
                ]
            )

        return output.getvalue()

    def _reconstruct_board_from_path(
        self, path: list[int]
    ) -> Board | None:
        """パスに沿って盤面を復元する．

        Args:
            path: ルートから対象ノードまでのposition_hashリスト

        Returns:
            復元されたBoardオブジェクト．復元不能の場合None．
        """
        board = Board()
        if self._initial_sfen is not None:
            board.set_sfen(self._initial_sfen)

        # パスに沿って指し手を適用
        for i in range(len(path) - 1):
            parent = path[i]
            child = path[i + 1]
            edge = self._query.get_edge_between(parent, child)
            if edge is None:
                return None

            move16 = edge["move16"]
            move = board.get_move_from_move16(move16)
            board.push_move(move)

        return board

    @staticmethod
    def _build_child_edge_map(
        sub_edges: pl.DataFrame,
    ) -> dict[int, dict[str, Any]]:
        """エッジDFから child_hash → 最大確率エッジ行のマップを構築する．

        複数の親を持つノードは最大確率のエッジを採用する．

        Args:
            sub_edges: サブツリーのエッジDataFrame

        Returns:
            child_hash をキー，エッジ行(dict)を値とするマッピング
        """
        child_edge_map: dict[int, dict[str, Any]] = {}
        for row in sub_edges.iter_rows(named=True):
            child_hash = row["child_hash"]
            if (
                child_hash not in child_edge_map
                or row["probability"]
                > child_edge_map[child_hash]["probability"]
            ):
                child_edge_map[child_hash] = row
        return child_edge_map

    def _build_boards_incrementally(
        self,
        root_hash: int,
        sub_nodes: pl.DataFrame,
        child_edge_map: dict[int, dict[str, Any]],
    ) -> dict[int, Board | None]:
        """サブツリー内の全ノードの盤面をdepth順に構築する．

        ルートの盤面のみ get_path_to_root で復元し，
        残りは親の盤面をコピーして1手適用する．
        これにより get_path_to_root の呼び出しを1回に削減する．

        Args:
            root_hash: サブツリーのルートhash
            sub_nodes: サブツリー内のノードDF
            child_edge_map: child_hash → エッジ情報

        Returns:
            position_hash → Board のマッピング
        """
        boards: dict[int, Board | None] = {}

        # サブツリーのルート盤面を復元
        path = self._query.get_path_to_root(root_hash)
        boards[root_hash] = (
            self._reconstruct_board_from_path(path)
            if path
            else None
        )

        # depth順にソートして漸進的に構築
        sorted_nodes = sub_nodes.sort("depth")
        for row in sorted_nodes.iter_rows(named=True):
            pos_hash = row["position_hash"]
            if pos_hash in boards:
                continue

            edge_info = child_edge_map.get(pos_hash)
            if edge_info is None:
                boards[pos_hash] = None
                continue

            parent_hash = edge_info["parent_hash"]
            parent_board = boards.get(parent_hash)
            if parent_board is None:
                boards[pos_hash] = None
                continue

            try:
                child_board = copy.copy(parent_board)
                move16 = edge_info["move16"]
                move = child_board.get_move_from_move16(move16)
                child_board.push_move(move)
                boards[pos_hash] = child_board
            except (ValueError, RuntimeError, IndexError):
                logger.warning(
                    "盤面復元に失敗しました: "
                    "position_hash=0x%016X",
                    pos_hash,
                    exc_info=True,
                )
                boards[pos_hash] = None

        return boards

    def _get_board_for_position(
        self, position_hash: int
    ) -> Board | None:
        """指定局面のBoardを取得する(1エントリキャッシュ付き)．

        同一局面に対する連続呼び出し(get_move_table → get_analytics_data)で
        重複する盤面復元を回避する．

        Note:
            スレッドセーフではない．Gradioのマルチスレッド環境では
            キャッシュミスによる重複計算が発生しうるが，
            データ破壊は起きない(最悪ケースは性能劣化のみ)．

        Args:
            position_hash: 対象ノードのZobrist hash

        Returns:
            復元されたBoardオブジェクト．復元不能の場合None．
        """
        if (
            self._board_cache is not None
            and self._board_cache[0] == position_hash
        ):
            return self._board_cache[1]

        path = self._query.get_path_to_root(position_hash)
        board = (
            self._reconstruct_board_from_path(path)
            if path
            else None
        )
        self._board_cache = (position_hash, board)
        return board

    @staticmethod
    def _move_to_arrow(move: int) -> MoveArrow:
        """cshogiの指し手をMoveArrowに変換する．

        Args:
            move: cshogiの指し手(get_move_from_move16の返り値)

        Returns:
            MoveArrowオブジェクト
        """
        if move_is_drop(move):
            return MoveArrow(
                from_square=None,
                to_square=move_to(move),
                is_drop=True,
                drop_piece_type=move_drop_hand_piece(move),
            )

        return MoveArrow(
            from_square=move_from(move),
            to_square=move_to(move),
        )

    def _move16_to_japanese(
        self,
        move16: int,
        parent_board: Board | None,
    ) -> str:
        """move16を日本語の指し手表記に変換する．

        Args:
            move16: 16bit指し手
            parent_board: 指し手適用前の盤面．Noneの場合は駒名なしで変換．

        Returns:
            日本語表記(例: "7六歩"，"5五歩打")
        """
        usi = move_to_usi(move16)
        piece_name = (
            self.get_piece_name(parent_board, move16)
            if parent_board is not None
            and len(usi) >= 2
            and usi[1] != "*"
            else ""
        )
        return self.usi_to_japanese(usi, piece_name=piece_name)

    @staticmethod
    def get_piece_name(board: Board, move16: int) -> str:
        """盤面とmove16から移動元の駒名を取得する．

        呼び出し側で駒打ち(usi[1]=="*")を除外済みのため，
        通常の移動のみを処理する．

        Args:
            board: 指し手適用前の盤面
            move16: 16bit指し手(駒打ち以外)

        Returns:
            日本語の駒名(例: "歩"，"角")
        """
        move = board.get_move_from_move16(move16)
        from_sq = move_from(move)
        cshogi_piece = board.get_piece_at(from_sq)
        piece_id = Board.cshogi_piece_to_piece_id(cshogi_piece)
        return get_piece_name_ja(piece_id)

    @classmethod
    def usi_to_japanese(
        cls,
        usi: str,
        piece_name: str = "",
    ) -> str:
        """USI表記を日本語表記に変換する．

        Args:
            usi: USI形式の指し手(例: "7g7f", "P*5e")
            piece_name: 駒名(例: "歩"，"角")．通常の指し手で表記に含める．
                駒打ち(usi[1]=="*")の場合はUSI文字列から駒名を取得するため
                この引数は使用されない．

        Returns:
            日本語表記(例: "7六歩", "5五歩打")
        """
        if not usi or len(usi) < 4:
            return usi

        # 駒打ちの場合
        if usi[1] == "*":
            piece = cls._DROP_PIECE_MAP.get(usi[0], usi[0])
            col = usi[2]
            row = cls._ROW_MAP.get(usi[3], usi[3])
            return f"{col}{row}{piece}打"

        # 通常の指し手
        to_col = usi[2]
        to_row = cls._ROW_MAP.get(usi[3], usi[3])

        # 成りの場合
        promotion = ""
        if len(usi) > 4 and usi[4] == "+":
            promotion = "成"

        return f"{to_col}{to_row}{piece_name}{promotion}"
