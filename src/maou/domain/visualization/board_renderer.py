"""将棋盤のSVG描画ロジック（ドメイン層）．

このモジュールは，将棋盤の駒配置をSVG形式で視覚的に表現する．
外部ライブラリに依存しない純粋な描画ロジックを提供する．
"""

from dataclasses import dataclass
from typing import List, Optional

from maou.domain.board.shogi import PieceId


@dataclass(frozen=True)
class BoardPosition:
    """不変な将棋盤の状態表現．

    Attributes:
        board_id_positions: 9×9の駒配置（PieceIdの2次元リスト）
        pieces_in_hand: 持ち駒配列（14要素: 先手7種 + 後手7種）
    """

    board_id_positions: List[List[int]]
    pieces_in_hand: List[int]

    def __post_init__(self) -> None:
        """バリデーション: 盤面と持ち駒のサイズ確認．"""
        if len(self.board_id_positions) != 9:
            raise ValueError(
                f"board_id_positions must have 9 rows, got {len(self.board_id_positions)}"
            )
        for i, row in enumerate(self.board_id_positions):
            if len(row) != 9:
                raise ValueError(
                    f"Row {i} must have 9 columns, got {len(row)}"
                )

        if len(self.pieces_in_hand) != 14:
            raise ValueError(
                f"pieces_in_hand must have 14 elements, got {len(self.pieces_in_hand)}"
            )


class SVGBoardRenderer:
    """将棋盤のSVG描画クラス．

    9×9の将棋盤と持ち駒をSVG形式で描画する．
    日本語の駒記号を使用し，視覚的にわかりやすい表現を提供する．
    """

    # SVG描画パラメータ
    CELL_SIZE = 50  # セルサイズ（ピクセル）
    BOARD_WIDTH = 9 * CELL_SIZE  # 盤面幅
    BOARD_HEIGHT = 9 * CELL_SIZE  # 盤面高さ
    HAND_AREA_WIDTH = 150  # 持ち駒エリア幅
    MARGIN = 20  # マージン

    # 駒の日本語表記（PieceId → 文字）
    PIECE_SYMBOLS = {
        PieceId.EMPTY: "",
        PieceId.FU: "歩",
        PieceId.KY: "香",
        PieceId.KE: "桂",
        PieceId.GI: "銀",
        PieceId.KI: "金",
        PieceId.KA: "角",
        PieceId.HI: "飛",
        PieceId.OU: "王",
        PieceId.TO: "と",
        PieceId.NKY: "杏",  # 成香
        PieceId.NKE: "圭",  # 成桂
        PieceId.NGI: "全",  # 成銀
        PieceId.UMA: "馬",
        PieceId.RYU: "龍",
    }

    # 持ち駒の種類（インデックス順）
    HAND_PIECE_NAMES = [
        "歩",
        "香",
        "桂",
        "銀",
        "金",
        "角",
        "飛",
    ]

    # 色設定
    COLOR_BOARD_BG = "#f4d58d"  # 盤面背景（薄い黄土色）
    COLOR_GRID = "#8b4513"  # グリッド線（茶色）
    COLOR_BLACK_PIECE = "#000000"  # 先手駒（黒）
    COLOR_WHITE_PIECE = "#c41e3a"  # 後手駒（赤）
    COLOR_HIGHLIGHT = "#90ee90"  # ハイライト（薄緑）

    def render(
        self,
        position: BoardPosition,
        highlight_squares: Optional[List[int]] = None,
    ) -> str:
        """将棋盤をSVGとして描画する．

        Args:
            position: 描画する盤面状態
            highlight_squares: ハイライトするマス（0-80のインデックス）

        Returns:
            完全なSVG文字列（HTML埋め込み可能）
        """
        highlight_set = set(highlight_squares or [])

        svg_parts = [
            self._svg_header(),
            self._draw_grid(),
            self._draw_pieces(
                position.board_id_positions, highlight_set
            ),
            self._draw_pieces_in_hand(position.pieces_in_hand),
            self._draw_coordinates(),
            self._svg_footer(),
        ]

        return "\n".join(svg_parts)

    def _svg_header(self) -> str:
        """SVGヘッダー（開始タグと設定）を生成．"""
        total_width = (
            self.MARGIN * 2
            + self.BOARD_WIDTH
            + self.HAND_AREA_WIDTH * 2
        )
        total_height = self.MARGIN * 2 + self.BOARD_HEIGHT

        return f"""<svg xmlns="http://www.w3.org/2000/svg"
                    width="{total_width}"
                    height="{total_height}"
                    viewBox="0 0 {total_width} {total_height}">
    <style>
        .piece {{ font-family: serif; font-weight: bold; text-anchor: middle; }}
        .black-piece {{ fill: {self.COLOR_BLACK_PIECE}; }}
        .white-piece {{ fill: {self.COLOR_WHITE_PIECE}; }}
        .coord {{ font-family: monospace; font-size: 12px; fill: #333; }}
    </style>"""

    def _svg_footer(self) -> str:
        """SVGフッター（終了タグ）を生成．"""
        return "</svg>"

    def _draw_grid(self) -> str:
        """将棋盤のグリッド線を描画．"""
        grid_parts = []

        # 盤面背景
        grid_parts.append(
            f'<rect x="{self.MARGIN + self.HAND_AREA_WIDTH}" '
            f'y="{self.MARGIN}" '
            f'width="{self.BOARD_WIDTH}" '
            f'height="{self.BOARD_HEIGHT}" '
            f'fill="{self.COLOR_BOARD_BG}" '
            f'stroke="{self.COLOR_GRID}" stroke-width="2"/>'
        )

        # 縦線（10本: 0-9列の境界）
        for i in range(10):
            x = (
                self.MARGIN
                + self.HAND_AREA_WIDTH
                + i * self.CELL_SIZE
            )
            y1 = self.MARGIN
            y2 = self.MARGIN + self.BOARD_HEIGHT
            grid_parts.append(
                f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" '
                f'stroke="{self.COLOR_GRID}" stroke-width="1"/>'
            )

        # 横線（10本: 0-9行の境界）
        for i in range(10):
            y = self.MARGIN + i * self.CELL_SIZE
            x1 = self.MARGIN + self.HAND_AREA_WIDTH
            x2 = (
                self.MARGIN
                + self.HAND_AREA_WIDTH
                + self.BOARD_WIDTH
            )
            grid_parts.append(
                f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" '
                f'stroke="{self.COLOR_GRID}" stroke-width="1"/>'
            )

        return "\n".join(grid_parts)

    def _draw_pieces(
        self,
        board_id_positions: List[List[int]],
        highlight_set: set,
    ) -> str:
        """盤上の駒を描画．

        Args:
            board_id_positions: 9×9の駒配置
            highlight_set: ハイライトするマスのセット
        """
        piece_parts = []

        for row in range(9):
            for col in range(9):
                piece_id = board_id_positions[row][col]
                square_idx = row * 9 + col

                # マスのハイライト（駒の有無に関係なく描画）
                if square_idx in highlight_set:
                    x_rect = (
                        self.MARGIN
                        + self.HAND_AREA_WIDTH
                        + col * self.CELL_SIZE
                    )
                    y_rect = self.MARGIN + row * self.CELL_SIZE
                    piece_parts.append(
                        f'<rect x="{x_rect}" y="{y_rect}" '
                        f'width="{self.CELL_SIZE}" height="{self.CELL_SIZE}" '
                        f'fill="{self.COLOR_HIGHLIGHT}" opacity="0.5"/>'
                    )

                # 駒がない場合はスキップ
                if piece_id == PieceId.EMPTY:
                    continue

                # 駒の描画
                is_white = (
                    piece_id >= 16
                )  # 後手駒はPieceId + 16
                actual_piece_id = (
                    piece_id - 16 if is_white else piece_id
                )
                symbol = self.PIECE_SYMBOLS.get(
                    actual_piece_id, "?"
                )

                x = (
                    self.MARGIN
                    + self.HAND_AREA_WIDTH
                    + col * self.CELL_SIZE
                    + self.CELL_SIZE / 2
                )
                y = (
                    self.MARGIN
                    + row * self.CELL_SIZE
                    + self.CELL_SIZE / 2
                    + 8
                )  # 中央揃え調整

                color_class = (
                    "white-piece" if is_white else "black-piece"
                )

                # 後手駒は180度回転
                transform = (
                    f'transform="rotate(180 {x} {y - 8})"'
                    if is_white
                    else ""
                )

                piece_parts.append(
                    f'<text x="{x}" y="{y}" '
                    f'class="piece {color_class}" '
                    f'font-size="30" {transform}>{symbol}</text>'
                )

        return "\n".join(piece_parts)

    def _draw_pieces_in_hand(
        self, pieces_in_hand: List[int]
    ) -> str:
        """持ち駒を描画．

        Args:
            pieces_in_hand: 14要素の持ち駒配列
                インデックス0-6: 先手の持ち駒（歩香桂銀金角飛）
                インデックス7-13: 後手の持ち駒（歩香桂銀金角飛）
        """
        hand_parts = []

        # 先手の持ち駒（左側）
        hand_parts.append(
            self._draw_single_hand(
                pieces=pieces_in_hand[:7],
                x_base=self.MARGIN,
                y_base=self.MARGIN,
                title="先手持ち駒",
                is_black=True,
            )
        )

        # 後手の持ち駒（右側）
        hand_parts.append(
            self._draw_single_hand(
                pieces=pieces_in_hand[7:14],
                x_base=self.MARGIN
                + self.HAND_AREA_WIDTH
                + self.BOARD_WIDTH,
                y_base=self.MARGIN,
                title="後手持ち駒",
                is_black=False,
            )
        )

        return "\n".join(hand_parts)

    def _draw_single_hand(
        self,
        pieces: List[int],
        x_base: float,
        y_base: float,
        title: str,
        is_black: bool,
    ) -> str:
        """片側の持ち駒エリアを描画．

        Args:
            pieces: 7要素の持ち駒配列
            x_base: エリアの左端X座標
            y_base: エリアの上端Y座標
            title: タイトル文字列
            is_black: 先手（黒）の持ち駒かどうか
        """
        parts = []

        # 持ち駒エリアの背景（視認性向上のため）
        parts.append(
            f'<rect x="{x_base}" y="{y_base}" '
            f'width="{self.HAND_AREA_WIDTH}" height="{self.BOARD_HEIGHT}" '
            f'fill="#ffffff" stroke="{self.COLOR_GRID}" stroke-width="1" opacity="0.9"/>'
        )

        # タイトル
        parts.append(
            f'<text x="{x_base + self.HAND_AREA_WIDTH / 2}" '
            f'y="{y_base + 15}" '
            f'class="coord" text-anchor="middle" font-weight="bold">{title}</text>'
        )

        # 各駒種の表示
        y_offset = y_base + 40
        color_class = (
            "black-piece" if is_black else "white-piece"
        )

        for i, (piece_name, count) in enumerate(
            zip(self.HAND_PIECE_NAMES, pieces)
        ):
            if count == 0:
                continue

            # 駒名 + 枚数
            text = (
                f"{piece_name}×{count}"
                if count > 1
                else piece_name
            )

            parts.append(
                f'<text x="{x_base + 10}" '
                f'y="{y_offset + i * 25}" '
                f'class="piece {color_class}" font-size="20">{text}</text>'
            )

        return "\n".join(parts)

    def _draw_coordinates(self) -> str:
        """盤面の座標（1-9, a-i）を描画．"""
        coord_parts = []

        # 列番号（1-9，上から順）
        for col in range(9):
            x = (
                self.MARGIN
                + self.HAND_AREA_WIDTH
                + col * self.CELL_SIZE
                + self.CELL_SIZE / 2
            )
            y = self.MARGIN - 5

            coord_parts.append(
                f'<text x="{x}" y="{y}" '
                f'class="coord" text-anchor="middle">{9 - col}</text>'
            )

        # 行記号（a-i，左側）
        for row in range(9):
            x = self.MARGIN + self.HAND_AREA_WIDTH - 10
            y = int(
                self.MARGIN
                + row * self.CELL_SIZE
                + self.CELL_SIZE / 2
                + 5
            )

            coord_parts.append(
                f'<text x="{x}" y="{y}" '
                f'class="coord" text-anchor="end">{chr(ord("a") + row)}</text>'
            )

        return "\n".join(coord_parts)
