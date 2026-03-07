"""ゲームツリー可視化のインターフェース層．

App層のGameTreeQueryとInfra層のGradio UIを接続するアダプタ．
Cytoscape.js用のデータ変換や盤面SVG生成を担当する．
"""

from __future__ import annotations

from typing import Any

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


class GameTreeVisualizationInterface:
    """ゲームツリー可視化のインターフェース層．

    GameTreeQueryのデータをUI向けに変換する．
    """

    def __init__(
        self,
        query: GameTreeQuery,
        root_hash: int,
    ) -> None:
        """初期化．

        Args:
            query: ゲームツリークエリオブジェクト
            root_hash: ツリーのルートノードのhash
        """
        self._query = query
        self._root_hash = root_hash
        self._renderer = SVGBoardRenderer()

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

        cy_nodes: list[dict[str, Any]] = []
        for row in sub_nodes.iter_rows(named=True):
            pos_hash = row["position_hash"]
            edge_info = child_edge_map.get(pos_hash)

            label = "ROOT"
            probability = 1.0
            if edge_info is not None:
                usi = move_to_usi(edge_info["move16"])
                label = self._usi_to_japanese(usi)
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
        board = self._reconstruct_board(position_hash)
        if board is None:
            return "<p>盤面を復元できません</p>"

        # 盤面からBoardPositionを生成
        board_id_positions = self._pieces_to_board_id_positions(
            board
        )
        black_hand, white_hand = board.get_pieces_in_hand()
        pieces_in_hand = list(black_hand) + list(white_hand)
        position = BoardPosition(
            board_id_positions=board_id_positions,
            pieces_in_hand=pieces_in_hand,
        )

        # 親からの指し手を矢印で表示
        move_arrow = self._get_move_arrow(position_hash)
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

        result: list[list[str]] = []
        for row in children.iter_rows(named=True):
            usi = move_to_usi(row["move16"])
            japanese = self._usi_to_japanese(usi)
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

        top_children = children.head(10)
        moves: list[str] = []
        probs: list[float] = []
        win_rates: list[float] = []

        for row in top_children.iter_rows(named=True):
            usi = move_to_usi(row["move16"])
            moves.append(self._usi_to_japanese(usi))
            probs.append(float(row["probability"]))
            win_rates.append(float(row["win_rate"]))

        return {
            "moves": moves,
            "probabilities": probs,
            "win_rates": win_rates,
        }

    def _reconstruct_board(
        self, position_hash: int
    ) -> Board | None:
        """ルートからのパスを辿って盤面を復元する．

        Args:
            position_hash: 対象ノードのZobrist hash

        Returns:
            復元されたBoardオブジェクト．復元不能の場合None．
        """
        path = self._query.get_path_to_root(position_hash)
        if not path:
            return None

        board = Board()

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

    def _get_move_arrow(
        self, position_hash: int
    ) -> MoveArrow | None:
        """親からの指し手をMoveArrowに変換する．

        Args:
            position_hash: 対象ノードのZobrist hash

        Returns:
            MoveArrowオブジェクト．ルートの場合None．
        """
        path = self._query.get_path_to_root(position_hash)
        if len(path) < 2:
            return None

        parent = path[-2]
        edge = self._query.get_edge_between(
            parent, position_hash
        )
        if edge is None:
            return None

        move16 = edge["move16"]
        # move16からMoveArrowを構築
        board = self._reconstruct_board(parent)
        if board is None:
            return None

        move = board.get_move_from_move16(move16)
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
    def _usi_to_japanese(usi: str) -> str:
        """USI表記を日本語表記に変換する．

        Args:
            usi: USI形式の指し手(例: "7g7f", "P*5e")

        Returns:
            日本語表記(例: "7六歩", "5五歩打")
        """
        if not usi or len(usi) < 4:
            return usi

        # 駒打ちの場合
        if usi[1] == "*":
            piece_map = {
                "P": "歩",
                "L": "香",
                "N": "桂",
                "S": "銀",
                "G": "金",
                "B": "角",
                "R": "飛",
            }
            piece = piece_map.get(usi[0], usi[0])
            col = usi[2]
            row_map = {
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
            row = row_map.get(usi[3], usi[3])
            return f"{col}{row}{piece}打"

        # 通常の指し手
        row_map = {
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
        to_col = usi[2]
        to_row = row_map.get(usi[3], usi[3])

        # 成りの場合
        promotion = ""
        if len(usi) > 4 and usi[4] == "+":
            promotion = "成"

        return f"{to_col}{to_row}{promotion}"

    @staticmethod
    def _pieces_to_board_id_positions(
        board: Board,
    ) -> list[list[int]]:
        """Boardオブジェクトから9×9のboard_id_positionsを生成する．

        SVGBoardRendererは[row][col]形式を期待する．
        board.get_pieces()はcolumn-major(square = col * 9 + row)なので
        reshapeにorder="F"を使い，転置して[row][col]形式にする．

        Args:
            board: Boardオブジェクト

        Returns:
            9×9のPieceId二次元リスト([row][col]形式)
        """
        import numpy as np

        v_map = np.vectorize(
            Board._cshogi_piece_to_piece_id,
            otypes=[np.uint8],
        )
        # cshogi: square = col * 9 + row
        # Fortran reshape: 列ごとに埋める → 結果は[row][col]形式
        # これはSVGBoardRendererが期待する形式と一致
        positions = v_map(
            np.array(board.get_pieces(), dtype=np.uint8)
        ).reshape((9, 9), order="F")
        return positions.tolist()
