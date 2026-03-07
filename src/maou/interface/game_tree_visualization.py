"""ゲームツリー可視化のインターフェース層．

App層のGameTreeQueryとInfra層のGradio UIを接続するアダプタ．
Cytoscape.js用のデータ変換や盤面SVG生成を担当する．
"""

from __future__ import annotations

from typing import Any

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
from maou.domain.visualization.board_renderer import (
    BoardPosition,
    MoveArrow,
    SVGBoardRenderer,
)
from maou.domain.visualization.piece_mapping import (
    get_piece_name_ja,
)


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
        self._root_hash = int(root_nodes["position_hash"][0])
        self._initial_sfen = initial_sfen
        self._renderer = SVGBoardRenderer()
        self._board_cache: tuple[int, Board | None] | None = (
            None
        )

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
        # child_hash → (move16, probability) のマッピング
        child_edge_map: dict[int, dict[str, Any]] = {}
        for row in sub_edges.iter_rows(named=True):
            child_hash = row["child_hash"]
            # 最大確率のエッジを採用(複数親の場合)
            if (
                child_hash not in child_edge_map
                or row["probability"]
                > child_edge_map[child_hash]["probability"]
            ):
                child_edge_map[child_hash] = row

        # 親ノードの盤面をキャッシュして駒名を取得する
        board_cache: dict[int, Board | None] = {}

        def _get_board(
            pos_hash: int,
        ) -> Board | None:
            if pos_hash not in board_cache:
                path = self._query.get_path_to_root(pos_hash)
                board_cache[pos_hash] = (
                    self._reconstruct_board_from_path(path)
                    if path
                    else None
                )
            return board_cache[pos_hash]

        cy_nodes: list[dict[str, Any]] = []
        for row in sub_nodes.iter_rows(named=True):
            pos_hash = row["position_hash"]
            edge_info = child_edge_map.get(pos_hash)

            label = "ROOT"
            probability = 1.0
            if edge_info is not None:
                move16 = edge_info["move16"]
                usi = move_to_usi(move16)
                parent_board = _get_board(
                    edge_info["parent_hash"]
                )
                # 駒打ち(usi[1]=="*")はUSIから駒名を取得するため
                # _get_piece_nameの呼び出しは不要
                piece_name = (
                    self._get_piece_name(parent_board, move16)
                    if parent_board is not None
                    and len(usi) >= 2
                    and usi[1] != "*"
                    else ""
                )
                label = self._usi_to_japanese(
                    usi, piece_name=piece_name
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
            cy_edges.append(
                {
                    "data": {
                        "source": str(row["parent_hash"]),
                        "target": str(row["child_hash"]),
                        "label": f"{row['probability'] * 100:.1f}%",
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

        return {
            "局面ハッシュ": f"0x{detail['position_hash']:016X}",
            "勝率": f"{detail['result_value'] * 100:.1f}%",
            "最善手勝率": f"{detail['best_move_win_rate'] * 100:.1f}%",
            "深さ": str(detail["depth"]),
            "分岐数": str(detail["num_branches"]),
        }

    def get_move_table(
        self, position_hash: int
    ) -> list[list[str]]:
        """指定局面の指し手一覧テーブルを生成する．

        Args:
            position_hash: 対象ノードのZobrist hash

        Returns:
            [[指し手, 確率, 勝率], ...] 形式のリスト(確率降順)
        """
        children = self._query.get_children(position_hash)
        if len(children) == 0:
            return []

        board = self._get_board_for_position(position_hash)

        result: list[list[str]] = []
        for row in children.iter_rows(named=True):
            move16 = row["move16"]
            usi = move_to_usi(move16)
            piece_name = (
                self._get_piece_name(board, move16)
                if board is not None
                and len(usi) >= 2
                and usi[1] != "*"
                else ""
            )
            japanese = self._usi_to_japanese(
                usi, piece_name=piece_name
            )
            prob = f"{row['probability'] * 100:.1f}%"
            wr = f"{row['win_rate'] * 100:.1f}%"
            result.append([japanese, prob, wr])

        return result

    def get_root_hash(self) -> int:
        """ツリーのルートハッシュを返す．

        Returns:
            ルートノードのposition_hash
        """
        return self._root_hash

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
            move16 = row["move16"]
            usi = move_to_usi(move16)
            piece_name = (
                self._get_piece_name(board, move16)
                if board is not None
                and len(usi) >= 2
                and usi[1] != "*"
                else ""
            )
            moves.append(
                self._usi_to_japanese(
                    usi, piece_name=piece_name
                )
            )
            probs.append(float(row["probability"]))
            win_rates.append(float(row["win_rate"]))

        return {
            "moves": moves,
            "probabilities": probs,
            "win_rates": win_rates,
        }

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

    def _get_board_for_position(
        self, position_hash: int
    ) -> Board | None:
        """指定局面のBoardを取得する(1エントリキャッシュ付き)．

        同一局面に対する連続呼び出し(get_move_table → get_analytics_data)で
        重複する盤面復元を回避する．

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

    @staticmethod
    def _get_piece_name(board: Board, move16: int) -> str:
        """盤面とmove16から移動元の駒名を取得する．

        Args:
            board: 指し手適用前の盤面
            move16: 16bit指し手

        Returns:
            日本語の駒名(例: "歩"，"角")
        """
        move = board.get_move_from_move16(move16)
        if move_is_drop(move):
            hand_piece = move_drop_hand_piece(move)
            piece_id = Board.cshogi_piece_to_piece_id(
                hand_piece
            )
            return get_piece_name_ja(piece_id)
        from_sq = move_from(move)
        cshogi_piece = board.board.piece(from_sq)
        piece_id = Board.cshogi_piece_to_piece_id(cshogi_piece)
        return get_piece_name_ja(piece_id)

    @classmethod
    def _usi_to_japanese(
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
