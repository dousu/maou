"""レコード描画のStrategy Pattern実装（アプリケーション層）．

このモジュールは，array_type固有の描画戦略を定義するRecordRendererと
その具象実装（HCPE, Stage1, Stage2, Preprocessing）を提供する．
"""

import logging
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Type,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go

from maou.domain.board.shogi import (
    CSHOGI_BLACK_MAX,
    CSHOGI_BLACK_MIN,
    DOMAIN_WHITE_MIN,
    DOMAIN_WHITE_OFFSET,
    Board,
    Turn,
    move_drop_hand_piece,
    move_from,
    move_is_drop,
    move_to,
)
from maou.domain.visualization.board_renderer import (
    BoardPosition,
    MoveArrow,
    SVGBoardRenderer,
)
from maou.domain.visualization.move_label_converter import (
    MoveLabelConverter,
)

logger: logging.Logger = logging.getLogger(__name__)


class RecordRenderer(ABC):
    """array_type固有の描画戦略を定義する抽象基底クラス．

    各array_type（HCPE, Stage1, Stage2, Preprocessing）は，
    このクラスを継承して型固有の描画ロジックを実装する．
    """

    def __init__(
        self,
        board_renderer: SVGBoardRenderer,
        move_converter: MoveLabelConverter,
    ):
        """RecordRendererを初期化する．

        Args:
            board_renderer: SVG描画エンジン
            move_converter: 駒移動ラベル変換サービス
        """
        self.board_renderer = board_renderer
        self.move_converter = move_converter

    @abstractmethod
    def render_board(self, record: Dict[str, Any]) -> str:
        """盤面SVGを型固有の拡張込みで描画する．

        Args:
            record: レコードデータ（boardIdPositions, piecesInHandを含む）

        Returns:
            SVG文字列
        """
        pass

    @abstractmethod
    def extract_display_fields(
        self, record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """UIに表示するフィールドを抽出する．

        Args:
            record: レコードデータ

        Returns:
            表示フィールドの辞書
        """
        pass

    @abstractmethod
    def get_table_columns(self) -> List[str]:
        """検索結果テーブルのカラム名を取得する．

        Returns:
            カラム名のリスト
        """
        pass

    @abstractmethod
    def format_table_row(
        self, index: int, record: Dict[str, Any]
    ) -> List[Any]:
        """テーブル行データをフォーマットする．

        Args:
            index: 行インデックス（表示用）
            record: レコードデータ

        Returns:
            セル値のリスト
        """
        pass

    @abstractmethod
    def generate_analytics(
        self, records: List[Dict[str, Any]]
    ) -> Optional["go.Figure"]:
        """レコード群からデータ分析用のPlotly Figureを生成する．

        Args:
            records: 分析対象のレコードリスト

        Returns:
            Plotly Figureオブジェクト，またはデータがない場合はNone
        """
        pass

    def _create_board_position(
        self, record: Dict[str, Any]
    ) -> BoardPosition:
        """レコードからBoardPositionを作成する共通ヘルパー．

        Args:
            record: レコードデータ

        Returns:
            BoardPosition インスタンス
        """
        return BoardPosition(
            board_id_positions=record.get(
                "boardIdPositions", []
            ),
            pieces_in_hand=record.get("piecesInHand", []),
        )

    def _create_board_from_record(
        self, record: Dict[str, Any]
    ) -> Board:
        """レコードからBoardインスタンスを再構築する．

        boardIdPositions（9x9配列）とpiecesInHand（14要素配列）から
        cshogi.Boardインスタンスを生成する．

        Args:
            record: レコードデータ

        Returns:
            Board インスタンス

        Note:
            この実装は簡易版であり，完全なHCP変換は今後の改善課題．
            現在はSFEN形式経由での変換を試みる．
        """
        board = Board()

        # boardIdPositions と piecesInHand から SFEN 形式を構築
        try:
            sfen = self._convert_to_sfen(
                board_id_positions=record.get(
                    "boardIdPositions", []
                ),
                pieces_in_hand=record.get("piecesInHand", []),
            )
            board.set_sfen(sfen)
        except Exception as e:
            logger.warning(
                f"Failed to reconstruct board from record: {e}"
            )
            # フォールバック: 初期局面
            board.set_sfen(
                "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
            )

        return board

    def _convert_to_sfen(
        self,
        board_id_positions: List[List[int]],
        pieces_in_hand: List[int],
    ) -> str:
        """boardIdPositionsとpiecesInHandからSFEN形式に変換する．

        Args:
            board_id_positions: 9x9の駒配置（PieceId値）
            pieces_in_hand: 持ち駒配列（14要素）

        Returns:
            SFEN形式の文字列

        Note:
            これは簡易実装であり，正確なSFEN生成には
            駒の所属（先手/後手）判定が必要．
            現状は基本的な変換のみ実装．
        """
        # 先手駒ID → SFEN文字マッピング
        # domain PieceId: 先手=1-14, 後手=15-28（先手+14）
        black_piece_to_sfen = {
            1: "P",  # 歩
            2: "L",  # 香
            3: "N",  # 桂
            4: "S",  # 銀
            5: "G",  # 金
            6: "B",  # 角
            7: "R",  # 飛
            8: "K",  # 玉
            9: "+P",  # と
            10: "+L",  # 成香
            11: "+N",  # 成桂
            12: "+S",  # 成銀
            13: "+B",  # 馬
            14: "+R",  # 龍
        }

        def get_sfen_char(piece_id: int) -> str:
            """駒IDからSFEN文字を取得．後手駒は小文字で返す．

            駒ID定数は shogi.py の DOMAIN_* を使用．
            boardIdPositionsはdomain PieceId形式(白駒=黒駒+14)．
            """
            if piece_id == 0:
                return ""
            if CSHOGI_BLACK_MIN <= piece_id <= CSHOGI_BLACK_MAX:
                # 先手駒（大文字）
                return black_piece_to_sfen.get(piece_id, "")
            elif piece_id >= DOMAIN_WHITE_MIN:
                # 後手駒（小文字）: piece_id - 14 の先手マッピングを小文字化
                black_char = black_piece_to_sfen.get(
                    piece_id - DOMAIN_WHITE_OFFSET, ""
                )
                return black_char.lower()
            return ""

        # 盤面をSFEN形式に変換
        ranks = []
        for row in board_id_positions:
            rank_str = ""
            empty_count = 0
            for piece_id in row:
                if piece_id == 0:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        rank_str += str(empty_count)
                        empty_count = 0
                    piece_char = get_sfen_char(piece_id)
                    rank_str += piece_char

            if empty_count > 0:
                rank_str += str(empty_count)

            ranks.append(rank_str if rank_str else "9")

        board_sfen = "/".join(ranks)

        # 持ち駒（先手: 大文字，後手: 小文字）
        hand_sfen = "-"
        hand_parts = []
        piece_chars = ["P", "L", "N", "S", "G", "B", "R"]

        # 先手の持ち駒（pieces_in_hand[0:7]）
        if len(pieces_in_hand) >= 7:
            for i, count in enumerate(pieces_in_hand[:7]):
                if count > 0:
                    if count > 1:
                        hand_parts.append(
                            f"{count}{piece_chars[i]}"
                        )
                    else:
                        hand_parts.append(piece_chars[i])

        # 後手の持ち駒（pieces_in_hand[7:14]）
        if len(pieces_in_hand) >= 14:
            for i, count in enumerate(pieces_in_hand[7:14]):
                if count > 0:
                    char = piece_chars[i].lower()
                    if count > 1:
                        hand_parts.append(f"{count}{char}")
                    else:
                        hand_parts.append(char)

        if hand_parts:
            hand_sfen = "".join(hand_parts)

        # 手番（デフォルト: 先手）
        turn = "b"

        # 手数（デフォルト: 1）
        move_count = "1"

        return f"{board_sfen} {turn} {hand_sfen} {move_count}"


class HCPERecordRenderer(RecordRenderer):
    """HCPE（棋譜データ）の描画戦略．

    評価値（eval）と手数（moves）を表示する．
    """

    def render_board(self, record: Dict[str, Any]) -> str:
        """盤面SVGを描画する（手番とレコードID表示込み）．

        Args:
            record: HCPEレコードデータ

        Returns:
            SVG文字列
        """
        position = self._create_board_position(record)

        # 手番を取得（DataRetrieverで抽出済み）
        turn_value = record.get("turn")
        turn = (
            Turn(turn_value) if turn_value is not None else None
        )

        # レコードIDを取得
        record_id = str(record.get("id", ""))

        # 矢印を生成
        move_arrow = self._create_move_arrow(record)

        return self.board_renderer.render(
            position,
            turn=turn,
            record_id=record_id,
            move_arrow=move_arrow,
        )

    def _create_move_arrow(
        self, record: Dict[str, Any]
    ) -> Optional[MoveArrow]:
        """レコードからMoveArrowを生成する．

        Args:
            record: HCPEレコードデータ

        Returns:
            MoveArrowオブジェクト，生成できない場合はNone
        """
        best_move = record.get("bestMove16")
        if best_move is None:
            return None

        try:
            to_sq = move_to(best_move)

            if move_is_drop(best_move):
                # 駒打ち
                piece_type = move_drop_hand_piece(best_move)
                return MoveArrow(
                    from_square=None,
                    to_square=to_sq,
                    is_drop=True,
                    drop_piece_type=piece_type,
                )
            else:
                # 通常移動
                from_sq = move_from(best_move)
                return MoveArrow(
                    from_square=from_sq,
                    to_square=to_sq,
                    is_drop=False,
                )
        except Exception:
            return None

    def extract_display_fields(
        self, record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """表示フィールドを抽出する．

        Args:
            record: HCPEレコードデータ

        Returns:
            id, eval, movesを含む辞書
        """
        return {
            "id": record.get("id"),
            "eval": record.get("eval"),
            "moves": record.get("moves"),
        }

    def get_table_columns(self) -> List[str]:
        """テーブルカラム名を取得する．

        Returns:
            ["Index", "ID", "Eval", "Moves"]
        """
        return ["Index", "ID", "Eval", "Moves"]

    def format_table_row(
        self, index: int, record: Dict[str, Any]
    ) -> List[Any]:
        """テーブル行をフォーマットする．

        Args:
            index: 行インデックス
            record: HCPEレコードデータ

        Returns:
            [index, id, eval, moves]のリスト
        """
        return [
            index,
            record.get("id"),
            record.get("eval"),
            record.get("moves"),
        ]

    def generate_analytics(
        self, records: List[Dict[str, Any]]
    ) -> Optional["go.Figure"]:
        """HCPEデータから評価値の分布チャートを生成する．

        Args:
            records: HCPEレコードのリスト

        Returns:
            Plotly Figureオブジェクト，またはデータがない場合はNone
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        if not records:
            return None

        # データ抽出（評価値のみ）
        evals = [
            r.get("eval", 0)
            for r in records
            if r.get("eval") is not None
        ]

        if not evals:
            return None

        # 評価値ヒストグラム（単一チャート）
        fig = go.Figure(
            data=[
                go.Histogram(
                    x=evals,
                    marker_color="rgba(0,112,243,0.6)",
                    nbinsx=30,
                    name="評価値",
                )
            ]
        )

        # レイアウト設定
        fig.update_layout(
            title="評価値分布",
            xaxis_title="評価値",
            yaxis_title="頻度",
            template="plotly_white",
            font=dict(family="system-ui", size=12),
            height=400,
            showlegend=False,
            margin=dict(l=40, r=40, t=60, b=40),
        )

        return fig


class Stage1RecordRenderer(RecordRenderer):
    """Stage1（到達可能マス）の描画戦略．

    reachableSquares（9x9 binary）をハイライト表示する．
    """

    def render_board(self, record: Dict[str, Any]) -> str:
        """盤面SVGを描画する（到達可能マスをハイライト，手番とレコードID表示込み）．

        Args:
            record: Stage1レコードデータ

        Returns:
            SVG文字列
        """
        # reachableSquares を抽出してハイライト
        reachable = record.get("reachableSquares", [])
        highlight_squares = self._extract_reachable_squares(
            reachable
        )

        position = self._create_board_position(record)

        # Stage1データは先手視点に正規化済み
        turn = Turn.BLACK
        record_id = str(record.get("id", ""))

        return self.board_renderer.render(
            position,
            highlight_squares,
            turn=turn,
            record_id=record_id,
        )

    def _extract_reachable_squares(
        self, reachable: List[List[int]]
    ) -> List[int]:
        """9x9 binary matrix → square indices (0-80)に変換する．

        Args:
            reachable: 9x9のバイナリ配列

        Returns:
            到達可能なマスのインデックスリスト（0-80）
        """
        squares = []
        for row in range(min(9, len(reachable))):
            for col in range(min(9, len(reachable[row]))):
                if reachable[row][col] == 1:
                    squares.append(row * 9 + col)
        return squares

    def extract_display_fields(
        self, record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """表示フィールドを抽出する．

        Args:
            record: Stage1レコードデータ

        Returns:
            id, reachable_count, array_typeを含む辞書
        """
        reachable = record.get("reachableSquares", [])
        num_reachable = sum(sum(row) for row in reachable)

        return {
            "id": record.get("id"),
            "reachable_count": num_reachable,
            "array_type": "stage1",
        }

    def get_table_columns(self) -> List[str]:
        """テーブルカラム名を取得する．

        Returns:
            ["Index", "ID", "Reachable Squares"]
        """
        return ["Index", "ID", "Reachable Squares"]

    def format_table_row(
        self, index: int, record: Dict[str, Any]
    ) -> List[Any]:
        """テーブル行をフォーマットする．

        Args:
            index: 行インデックス
            record: Stage1レコードデータ

        Returns:
            [index, id, reachable_count]のリスト
        """
        reachable = record.get("reachableSquares", [])
        num_reachable = sum(sum(row) for row in reachable)
        return [index, record.get("id"), num_reachable]

    def generate_analytics(
        self, records: List[Dict[str, Any]]
    ) -> Optional["go.Figure"]:
        """Stage1データから到達可能マス数の分布チャートを生成する．

        Args:
            records: Stage1レコードのリスト

        Returns:
            Plotly Figureオブジェクト，またはデータがない場合はNone
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        if not records:
            return None

        # 到達可能マス数を集計
        reachable_counts = []
        for r in records:
            reachable = r.get("reachableSquares", [])
            count = sum(sum(row) for row in reachable)
            reachable_counts.append(count)

        # ヒストグラム作成
        fig = go.Figure(
            data=[
                go.Histogram(
                    x=reachable_counts,
                    marker_color="rgba(76,175,80,0.6)",
                    nbinsx=20,
                    name="到達可能マス数",
                )
            ]
        )

        fig.update_layout(
            title="到達可能マス数の分布",
            xaxis_title="到達可能マス数",
            yaxis_title="頻度",
            template="plotly_white",
            font=dict(family="system-ui", size=12),
            height=400,
            showlegend=False,
            margin=dict(l=40, r=40, t=60, b=40),
        )

        return fig


class Stage2RecordRenderer(RecordRenderer):
    """Stage2（合法手）の描画戦略．

    legalMovesLabel（2187 binary）をUSI表記で表示する．
    """

    def render_board(self, record: Dict[str, Any]) -> str:
        """盤面SVGを描画する（手番とレコードID表示込み）．

        Args:
            record: Stage2レコードデータ

        Returns:
            SVG文字列
        """
        position = self._create_board_position(record)

        # Stage2データは先手視点に正規化済み
        turn = Turn.BLACK
        record_id = str(record.get("id", ""))

        return self.board_renderer.render(
            position, turn=turn, record_id=record_id
        )

    def extract_display_fields(
        self, record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """表示フィールドを抽出する．

        legalMovesLabelからUSI表記の合法手リストを生成する．

        Args:
            record: Stage2レコードデータ

        Returns:
            id, legal_moves_count, legal_moves, array_typeを含む辞書
        """
        # legalMovesLabel (2187 binary) → USI notation
        legal_labels = record.get("legalMovesLabel", [])
        legal_indices = [
            i for i, val in enumerate(legal_labels) if val == 1
        ]

        # Board 再構築
        board = self._create_board_from_record(record)
        usi_moves = (
            self.move_converter.convert_labels_to_usi_list(
                board, legal_indices, limit=20
            )
        )

        return {
            "id": record.get("id"),
            "legal_moves_count": len(legal_indices),
            "legal_moves": ", ".join(
                usi_moves[:10]
            ),  # 最初の10手
            "array_type": "stage2",
        }

    def get_table_columns(self) -> List[str]:
        """テーブルカラム名を取得する．

        Returns:
            ["Index", "ID", "Legal Moves Count"]
        """
        return ["Index", "ID", "Legal Moves Count"]

    def format_table_row(
        self, index: int, record: Dict[str, Any]
    ) -> List[Any]:
        """テーブル行をフォーマットする．

        Args:
            index: 行インデックス
            record: Stage2レコードデータ

        Returns:
            [index, id, legal_moves_count]のリスト
        """
        legal_labels = record.get("legalMovesLabel", [])
        num_legal = sum(legal_labels)
        return [index, record.get("id"), num_legal]

    def generate_analytics(
        self, records: List[Dict[str, Any]]
    ) -> Optional["go.Figure"]:
        """Stage2データから合法手数の分布チャートを生成する．

        Args:
            records: Stage2レコードのリスト

        Returns:
            Plotly Figureオブジェクト，またはデータがない場合はNone
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        if not records:
            return None

        # 合法手数を集計
        legal_counts = []
        for r in records:
            legal_labels = r.get("legalMovesLabel", [])
            count = sum(legal_labels)
            legal_counts.append(count)

        # ヒストグラム作成
        fig = go.Figure(
            data=[
                go.Histogram(
                    x=legal_counts,
                    marker_color="rgba(255,152,0,0.6)",
                    nbinsx=30,
                    name="合法手数",
                )
            ]
        )

        fig.update_layout(
            title="合法手数の分布",
            xaxis_title="合法手数",
            yaxis_title="頻度",
            template="plotly_white",
            font=dict(family="system-ui", size=12),
            height=400,
            showlegend=False,
            margin=dict(l=40, r=40, t=60, b=40),
        )

        return fig


class PreprocessingRecordRenderer(RecordRenderer):
    """Preprocessing（訓練データ）の描画戦略．

    moveLabel（2187 probabilities）を上位USI手として表示する．
    """

    def render_board(self, record: Dict[str, Any]) -> str:
        """盤面SVGを描画する（手番とレコードID表示込み）．

        Args:
            record: Preprocessingレコードデータ

        Returns:
            SVG文字列
        """
        position = self._create_board_position(record)

        # Preprocessingデータは先手視点に正規化済み
        turn = Turn.BLACK
        record_id = str(record.get("id", ""))

        return self.board_renderer.render(
            position, turn=turn, record_id=record_id
        )

    def extract_display_fields(
        self, record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """表示フィールドを抽出する．

        moveLabelから確率上位のUSI手リストを生成する．

        Args:
            record: Preprocessingレコードデータ

        Returns:
            id, result_value, top_moves, array_typeを含む辞書
        """
        # moveLabel (2187 probabilities) → Top-K USI moves
        move_labels = record.get("moveLabel", [])

        # Board 再構築
        board = self._create_board_from_record(record)
        top_moves = self.move_converter.convert_probability_labels_to_usi(
            board, move_labels, threshold=0.01, top_k=5
        )

        return {
            "id": record.get("id"),
            "result_value": record.get("resultValue"),
            "top_moves": ", ".join(
                f"{usi}({prob:.1%})" for usi, prob in top_moves
            ),
            "array_type": "preprocessing",
        }

    def get_table_columns(self) -> List[str]:
        """テーブルカラム名を取得する．

        Returns:
            ["Index", "ID", "Result Value"]
        """
        return ["Index", "ID", "Result Value"]

    def format_table_row(
        self, index: int, record: Dict[str, Any]
    ) -> List[Any]:
        """テーブル行をフォーマットする．

        Args:
            index: 行インデックス
            record: Preprocessingレコードデータ

        Returns:
            [index, id, result_value]のリスト
        """
        return [
            index,
            record.get("id"),
            f"{record.get('resultValue', 0):.2f}",
        ]

    def generate_analytics(
        self, records: List[Dict[str, Any]]
    ) -> Optional["go.Figure"]:
        """Preprocessingデータから勝率分布チャートを生成する．

        Args:
            records: Preprocessingレコードのリスト

        Returns:
            Plotly Figureオブジェクト，またはデータがない場合はNone
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        if not records:
            return None

        # resultValueを集計
        result_values = [
            r.get("resultValue", 0.0)
            for r in records
            if r.get("resultValue") is not None
        ]

        # ヒストグラム作成
        fig = go.Figure(
            data=[
                go.Histogram(
                    x=result_values,
                    marker_color="rgba(156,39,176,0.6)",
                    nbinsx=20,
                    name="勝率",
                )
            ]
        )

        fig.update_layout(
            title="勝率（Result Value）の分布",
            xaxis_title="勝率",
            yaxis_title="頻度",
            template="plotly_white",
            font=dict(family="system-ui", size=12),
            height=400,
            showlegend=False,
            margin=dict(l=40, r=40, t=60, b=40),
        )

        return fig


class RecordRendererFactory:
    """array_typeから適切なRendererを生成するファクトリ．

    Strategy Patternの具象クラスを選択するファクトリクラス．
    """

    @staticmethod
    def create(
        array_type: str,
        board_renderer: SVGBoardRenderer,
        move_converter: MoveLabelConverter,
    ) -> RecordRenderer:
        """array_typeに応じたRendererインスタンスを生成する．

        Args:
            array_type: データ型（"hcpe", "stage1", "stage2", "preprocessing"）
            board_renderer: SVG描画エンジン
            move_converter: 駒移動ラベル変換サービス

        Returns:
            array_type対応のRecordRendererインスタンス

        Raises:
            ValueError: 未知のarray_typeの場合

        Example:
            >>> factory = RecordRendererFactory()
            >>> renderer = factory.create(
            ...     "stage1",
            ...     SVGBoardRenderer(),
            ...     MoveLabelConverter(),
            ... )
            >>> isinstance(renderer, Stage1RecordRenderer)
            True
        """
        renderers: Dict[str, Type[RecordRenderer]] = {
            "hcpe": HCPERecordRenderer,
            "stage1": Stage1RecordRenderer,
            "stage2": Stage2RecordRenderer,
            "preprocessing": PreprocessingRecordRenderer,
        }

        renderer_class = renderers.get(array_type)
        if not renderer_class:
            raise ValueError(
                f"Unknown array_type: {array_type}"
            )

        return renderer_class(board_renderer, move_converter)
