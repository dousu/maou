"""将棋盤のSVG描画を担当するモジュール．

重要: 将棋の座標系は直感に反する部分があります．
実装前に必ず docs/visualization/shogi-conventions.md を参照してください．
特に以下の点に注意:
- マスインデックスは square = col * 9 + row（row-major ではない）
- 座標変換には piece_mapping.py の関数を使用すること
"""

from dataclasses import dataclass
from typing import ClassVar

from maou.domain.board.shogi import (
    PieceId,
    Turn,
)
from maou.domain.visualization.piece_mapping import (
    get_piece_name_ja,
    is_white_piece,
    square_index_to_coords,
)


@dataclass(frozen=True)
class MoveArrow:
    """指し手を表す矢印データ．

    Attributes:
        from_square: 移動元マス（0-80）．駒打ちの場合はNone．
        to_square: 移動先マス（0-80）．
        is_drop: 駒打ちかどうか．
        drop_piece_type: 駒打ちの場合の駒種（0=歩, 1=香, ...）．
    """

    from_square: int | None
    to_square: int
    is_drop: bool = False
    drop_piece_type: int | None = None


# 矢印のデフォルトスタイル (ArrowSpec と SVGBoardRenderer で共有)
DEFAULT_ARROW_COLOR = "rgba(0, 100, 200, 0.6)"  # 半透明の青
DEFAULT_ARROW_WIDTH = 4  # 矢印の線幅


@dataclass(frozen=True)
class ArrowSpec:
    """スタイル付きの指し手矢印．

    候補手の複数表示など，色・太さ・不透明度・ラベルを矢印ごとに
    変えたい場合に使う．

    Attributes:
        move: 矢印にする指し手．
        color: 矢印の色 (CSS color 文字列)．
        width: 矢印の線幅 (ピクセル)．
        opacity: 不透明度 (0.0-1.0)．
        label: 矢印の始点寄りに描く短いラベル (候補手の順位等)．
            None で非表示．
    """

    move: MoveArrow
    color: str = DEFAULT_ARROW_COLOR
    width: float = DEFAULT_ARROW_WIDTH
    opacity: float = 1.0
    label: str | None = None


@dataclass(frozen=True)
class BoardTheme:
    """盤面 SVG の見た目 (寸法・配色・書体) をまとめた設定．

    既定値は従来の描画 (暖色系・角丸・影あり) と一致するため，
    引数なしの :class:`SVGBoardRenderer` の出力は変わらない．
    別テーマは :data:`MODERNIST_BOARD_THEME` を参照．
    """

    cell_size: int = 50
    hand_area_width: int = 150
    gap: int = 30
    margin: int = 20
    header_height: int = 50
    board_bg: str = "#f9f6f0"
    grid_color: str = "#d4c5a9"
    board_border_color: str = "#d4c5a9"
    board_border_width: float = 2
    black_piece_color: str = "#2c2c2c"
    white_piece_color: str = "#c41e3a"
    highlight_color: str = "rgba(0,112,243,0.12)"
    selected_color: str = "rgba(255,152,0,0.35)"
    destination_color: str = "rgba(76,175,80,0.28)"
    hover_color: str = "rgba(0,112,243,0.08)"
    hand_bg: str = "#fafafa"
    hand_border_color: str = "#d4c5a9"
    hand_border_width: float = 2
    hand_title_bg: str = "#f5f5f5"
    hand_font_size: int = 18
    coord_color: str = "#666666"
    badge_bg: str = "#ffffff"
    badge_border: str = "#d4d4d4"
    corner_radius: int = 6
    outer_radius: int = 8
    piece_font: str = '"Hiragino Mincho ProN", "Yu Mincho", "MS Mincho", serif'
    ui_font: str = '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    piece_font_size_ratio: float = 0.6
    piece_shadow: bool = True
    board_shadow: str = "0 4px 6px rgba(0,0,0,0.07)"
    show_header: bool = True
    show_coordinates: bool = True
    coordinate_badges: bool = True
    empty_hand_text: str = ""


MODERNIST_BOARD_THEME = BoardTheme(
    cell_size=52,
    hand_area_width=88,
    gap=28,
    margin=18,
    header_height=0,
    board_bg="#f8f4f4",
    grid_color="#bab6b6",
    board_border_color="#201e1d",
    black_piece_color="#201e1d",
    white_piece_color="#ec3013",
    highlight_color="rgba(236,48,19,0.12)",
    selected_color="rgba(32,30,29,0.16)",
    destination_color="rgba(236,48,19,0.26)",
    hover_color="rgba(32,30,29,0.06)",
    hand_bg="#f8f4f4",
    hand_border_color="#bab6b6",
    hand_border_width=1,
    hand_title_bg="transparent",
    hand_font_size=17,
    coord_color="#7d7979",
    corner_radius=0,
    outer_radius=0,
    piece_font='"Shippori Mincho", "Hiragino Mincho ProN", "Yu Mincho", serif',
    ui_font='"Archivo", "Noto Sans JP", system-ui, sans-serif',
    piece_font_size_ratio=0.62,
    piece_shadow=False,
    board_shadow="",
    show_header=False,
    coordinate_badges=False,
    empty_hand_text="なし",
)
"""棋譜解析 GUI (analyze-gui) のワークベンチ用テーマ (Modernist)．"""


@dataclass(frozen=True)
class BoardPosition:
    """不変な将棋盤の状態表現．

    Attributes:
        board_id_positions: 9×9の駒配置（PieceIdの2次元リスト）
        pieces_in_hand: 持ち駒配列（14要素: 先手7種 + 後手7種）
    """

    board_id_positions: list[list[int]]
    pieces_in_hand: list[int]

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
    GAP_BETWEEN_HAND_AND_BOARD = 30  # 持ち駒と盤面の間の隙間
    MARGIN = 20  # マージン

    # 持ち駒の種類（インデックス0-6: 歩香桂銀金角飛．
    # piece_mapping.get_piece_name_ja から導出し駒名表を一本化）
    HAND_PIECE_NAMES: ClassVar[list[str]] = [
        get_piece_name_ja(piece_id) for piece_id in range(1, 8)
    ]

    # 手番の日本語表記
    TURN_TEXT: ClassVar[dict[Turn, str]] = {
        Turn.BLACK: "先手番",
        Turn.WHITE: "後手番",
    }

    # ヘッダー表示設定
    HEADER_HEIGHT = 50  # ヘッダーエリアの高さ

    # 色設定（モダン・ミニマリストパレット）
    COLOR_BOARD_BG = "#f9f6f0"  # 盤面背景（微妙な暖色）
    COLOR_GRID = "#d4c5a9"  # グリッド線（ソフトな茶）
    COLOR_BLACK_PIECE = "#2c2c2c"  # 先手駒（コントラスト強化）
    COLOR_WHITE_PIECE = "#c41e3a"  # 後手駒（伝統的赤）
    COLOR_HIGHLIGHT = (
        "rgba(0,112,243,0.12)"  # ハイライト（モダンブルー）
    )
    COLOR_SELECTED = "rgba(255,152,0,0.35)"  # クリック選択中のマス（アンバー）
    COLOR_DESTINATION = (
        "rgba(76,175,80,0.28)"  # 選択駒の行き先候補（グリーン）
    )

    # 矢印の色設定 (module-level のデフォルトを共有)
    COLOR_ARROW = DEFAULT_ARROW_COLOR
    ARROW_WIDTH = DEFAULT_ARROW_WIDTH

    def __init__(self, theme: BoardTheme | None = None) -> None:
        """テーマを適用したレンダラーを作る．

        寸法・配色はインスタンス属性としてクラス定数を覆う．
        引数なしの場合は :class:`BoardTheme` の既定値 (= クラス定数と
        同じ値) になるため，従来どおりの出力になる．

        Args:
            theme: 適用するテーマ．None で既定テーマ．
        """
        self.theme = theme or BoardTheme()
        self.CELL_SIZE = self.theme.cell_size
        self.BOARD_WIDTH = 9 * self.CELL_SIZE
        self.BOARD_HEIGHT = 9 * self.CELL_SIZE
        self.HAND_AREA_WIDTH = self.theme.hand_area_width
        self.GAP_BETWEEN_HAND_AND_BOARD = self.theme.gap
        self.MARGIN = self.theme.margin
        self.HEADER_HEIGHT = self.theme.header_height
        self.COLOR_BOARD_BG = self.theme.board_bg
        self.COLOR_GRID = self.theme.grid_color
        self.COLOR_BLACK_PIECE = self.theme.black_piece_color
        self.COLOR_WHITE_PIECE = self.theme.white_piece_color
        self.COLOR_HIGHLIGHT = self.theme.highlight_color
        self.COLOR_SELECTED = self.theme.selected_color
        self.COLOR_DESTINATION = self.theme.destination_color

    def render(
        self,
        position: BoardPosition,
        highlight_squares: list[int] | None = None,
        turn: Turn | None = None,
        record_id: str | None = None,
        move_arrow: MoveArrow | None = None,
        move_arrows: list[ArrowSpec] | None = None,
        selected_squares: list[int] | None = None,
        destination_squares: list[int] | None = None,
        interactive: bool = False,
    ) -> str:
        """将棋盤をSVGとして描画する．

        Args:
            position: 描画する盤面状態
            highlight_squares: ハイライトするマス（0-80のインデックス．
                highlight/selected/destination は row * 9 + col の
                行優先索引 — 矢印の column-major とは異なる既存仕様）
            turn: 手番（Turn.BLACK または Turn.WHITE）
            record_id: レコードID
            move_arrow: 描画する指し手矢印（Noneの場合は矢印なし）．
                デフォルトスタイルの ArrowSpec 1 本と等価な後方互換引数
            move_arrows: スタイル付き矢印のリスト（候補手の複数表示等）．
                move_arrow と併用した場合は move_arrow が先に描画される
            selected_squares: クリック選択中として塗るマス
            destination_squares: 選択駒の行き先候補として塗るマス
            interactive: True でクリック標的の透明 rect
                (``data-click`` 属性付き) を盤上マスと持ち駒に重ねる

        Returns:
            完全なSVG文字列（HTML埋め込み可能）
        """
        highlight_set = set(highlight_squares or [])
        selected_set = set(selected_squares or [])
        destination_set = set(destination_squares or [])
        arrows: list[ArrowSpec] = []
        if move_arrow is not None:
            arrows.append(ArrowSpec(move=move_arrow))
        if move_arrows:
            arrows.extend(move_arrows)

        svg_parts = [
            self._svg_header(arrows),
            self._draw_header(turn, record_id),
            self._draw_grid(),
            self._draw_pieces(
                position.board_id_positions,
                highlight_set,
                selected_set,
                destination_set,
            ),
            self._draw_pieces_in_hand(position.pieces_in_hand),
            self._draw_arrows(
                arrows, position.pieces_in_hand, turn
            ),
            self._draw_coordinates()
            if self.theme.show_coordinates
            else "",
        ]
        if interactive:
            svg_parts.append(
                self._draw_click_targets(
                    position.pieces_in_hand
                )
            )
        svg_parts.append(self._svg_footer())

        return "\n".join(svg_parts)

    @staticmethod
    def _marker_id(style_index: int) -> str:
        """矢じりマーカーのSVG id（矢印スタイルごとに一意）．"""
        return f"arrowhead-{style_index}"

    @staticmethod
    def _arrow_styles(
        arrows: list[ArrowSpec],
    ) -> list[tuple[str, float]]:
        """矢印リストから一意な（色，不透明度）スタイル列を出現順で返す．

        矢じりマーカーはスタイルごとに 1 定義を共有する．
        """
        styles: list[tuple[str, float]] = []
        for spec in arrows:
            style = (spec.color, spec.opacity)
            if style not in styles:
                styles.append(style)
        return styles

    def _svg_header(self, arrows: list[ArrowSpec]) -> str:
        """SVGヘッダー（開始タグと設定）を生成．

        Args:
            arrows: 描画する矢印のリスト．スタイル（色，不透明度）ごとに
                矢じりマーカー定義を生成する
        """
        total_width = (
            self.MARGIN * 2
            + self.BOARD_WIDTH
            + self.HAND_AREA_WIDTH * 2
            + self.GAP_BETWEEN_HAND_AND_BOARD * 2
        )
        total_height = (
            self.MARGIN * 2
            + self.BOARD_HEIGHT
            + self.HEADER_HEIGHT
        )

        # 矢印マーカー定義（スタイルごとに 1 つ）
        arrow_marker = ""
        for i, (color, opacity) in enumerate(
            self._arrow_styles(arrows)
        ):
            arrow_marker += f"""
        <marker id="{self._marker_id(i)}" markerWidth="6" markerHeight="4"
                refX="5" refY="2" orient="auto" markerUnits="strokeWidth">
            <polygon points="0 0, 6 2, 0 4" fill="{color}" fill-opacity="{opacity}"/>
        </marker>"""

        # ヘッダー (レコードID/手番バッジ) は盤面上端より上の負 y 領域
        # (y = MARGIN - 46 付近) に描画されるため，viewBox の y 原点を
        # -HEADER_HEIGHT にしてヘッダー帯を可視域に含める
        theme = self.theme
        svg_style = "max-width: 100%; height: auto;"
        if theme.outer_radius:
            svg_style += (
                f" border-radius: {theme.outer_radius}px;"
            )
        if theme.board_shadow:
            svg_style += f" box-shadow: {theme.board_shadow};"
        shadow_filter = (
            """
        <filter id="piece-shadow" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="1" stdDeviation="1" flood-opacity="0.3"/>
        </filter>"""
            if theme.piece_shadow
            else ""
        )
        piece_filter = (
            "\n            filter: url(#piece-shadow);"
            if theme.piece_shadow
            else ""
        )

        return f"""<svg xmlns="http://www.w3.org/2000/svg"
                    width="{total_width}"
                    height="{total_height}"
                    viewBox="0 -{self.HEADER_HEIGHT} {total_width} {total_height}"
                    style="{svg_style}">
    <defs>{shadow_filter}{arrow_marker}
    </defs>
    <style>
        .piece {{
            font-family: {theme.piece_font};
            font-weight: 700;
            text-anchor: middle;{piece_filter}
        }}
        .black-piece {{
            fill: {self.COLOR_BLACK_PIECE};
        }}
        .white-piece {{
            fill: {self.COLOR_WHITE_PIECE};
        }}
        .coord {{
            font-family: {theme.ui_font};
            font-size: 12px;
            fill: {theme.coord_color};
        }}
        .board-square {{
            transition: fill 0.15s ease;
        }}
        .board-square:hover {{
            fill: {theme.hover_color};
        }}
        .header-text {{
            font-family: {theme.piece_font};
            font-size: 13px;
            font-weight: 600;
            fill: #1a1a1a;
        }}
        .turn-black {{
            fill: {self.COLOR_BLACK_PIECE};
        }}
        .turn-white {{
            fill: {self.COLOR_WHITE_PIECE};
        }}
    </style>"""

    def _svg_footer(self) -> str:
        """SVGフッター（終了タグ）を生成．"""
        return "</svg>"

    def _draw_header(
        self,
        turn: Turn | None,
        record_id: str | None,
    ) -> str:
        """盤面上部のヘッダー（レコードIDと手番）を描画．

        Args:
            turn: 手番（Turn.BLACK または Turn.WHITE）
            record_id: レコードID

        Returns:
            ヘッダーのSVG文字列
        """
        if not self.theme.show_header:
            return ""
        if turn is None and record_id is None:
            return ""  # 表示情報がない場合は空文字列

        header_parts = []

        # 盤面のX座標開始位置
        board_x_start = (
            self.MARGIN
            + self.HAND_AREA_WIDTH
            + self.GAP_BETWEEN_HAND_AND_BOARD
        )

        # ヘッダーのY座標（座標ラベルの上）
        header_y = self.MARGIN - 30

        # レコードID表示（左側）
        if record_id:
            record_id_text = f"ID: {record_id}"
            # 長いIDは切り詰め
            if len(record_id_text) > 30:
                record_id_text = record_id_text[:27] + "..."

            # 背景バッジ
            text_width = len(record_id_text) * 7 + 20  # 概算幅
            header_parts.append(
                f'<rect x="{board_x_start}" y="{header_y - 16}" '
                f'width="{text_width}" height="22" '
                f'fill="#ffffff" stroke="#d4d4d4" stroke-width="1" '
                f'rx="4" opacity="0.95"/>'
            )

            # テキスト
            header_parts.append(
                f'<text x="{board_x_start + 10}" y="{header_y}" '
                f'class="header-text" font-size="13">{record_id_text}</text>'
            )

        # 手番表示（右側）
        if turn is not None:
            turn_text = self.TURN_TEXT.get(turn, "")
            if turn_text:
                # 背景バッジ（右寄せ）
                badge_width = 80
                badge_x = (
                    board_x_start
                    + self.BOARD_WIDTH
                    - badge_width
                )

                header_parts.append(
                    f'<rect x="{badge_x}" y="{header_y - 16}" '
                    f'width="{badge_width}" height="22" '
                    f'fill="#ffffff" stroke="#d4d4d4" stroke-width="1" '
                    f'rx="4" opacity="0.95"/>'
                )

                # テキスト（中央揃え）
                text_x = badge_x + badge_width / 2
                color_class = (
                    "turn-black"
                    if turn == Turn.BLACK
                    else "turn-white"
                )

                header_parts.append(
                    f'<text x="{text_x}" y="{header_y}" '
                    f'class="header-text {color_class}" '
                    f'text-anchor="middle" font-size="14" font-weight="700">'
                    f"{turn_text}</text>"
                )

        return "\n".join(header_parts)

    def _draw_grid(self) -> str:
        """将棋盤のグリッド線を描画．"""
        grid_parts = []

        # 盤面のX座標開始位置（左側の持ち駒 + ギャップを考慮）
        board_x_start = (
            self.MARGIN
            + self.HAND_AREA_WIDTH
            + self.GAP_BETWEEN_HAND_AND_BOARD
        )

        # 盤面背景
        grid_parts.append(
            f'<rect x="{board_x_start}" '
            f'y="{self.MARGIN}" '
            f'width="{self.BOARD_WIDTH}" '
            f'height="{self.BOARD_HEIGHT}" '
            f'fill="{self.COLOR_BOARD_BG}" '
            f'stroke="{self.theme.board_border_color}" '
            f'stroke-width="{self.theme.board_border_width}"/>'
        )

        # ホバーターゲット（各マスに透明な矩形を配置）
        for row in range(9):
            for col in range(9):
                x = board_x_start + col * self.CELL_SIZE
                y = self.MARGIN + row * self.CELL_SIZE
                grid_parts.append(
                    f'<rect class="board-square" '
                    f'x="{x}" y="{y}" '
                    f'width="{self.CELL_SIZE}" height="{self.CELL_SIZE}" '
                    f'fill="transparent" />'
                )

        # 縦線（10本: 0-9列の境界）
        for i in range(10):
            x = board_x_start + i * self.CELL_SIZE
            y1 = self.MARGIN
            y2 = self.MARGIN + self.BOARD_HEIGHT
            grid_parts.append(
                f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" '
                f'stroke="{self.COLOR_GRID}" stroke-width="1"/>'
            )

        # 横線（10本: 0-9行の境界）
        for i in range(10):
            y = self.MARGIN + i * self.CELL_SIZE
            x1 = board_x_start
            x2 = board_x_start + self.BOARD_WIDTH
            grid_parts.append(
                f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" '
                f'stroke="{self.COLOR_GRID}" stroke-width="1"/>'
            )

        return "\n".join(grid_parts)

    def _draw_pieces(
        self,
        board_id_positions: list[list[int]],
        highlight_set: set,
        selected_set: set | None = None,
        destination_set: set | None = None,
    ) -> str:
        """盤上の駒を描画．

        Args:
            board_id_positions: 9×9の駒配置
                配列インデックス [row][col] は以下のように対応:
                - col: 0=右端(筋9), 8=左端(筋1) ← 将棋の筋は右から左
                - row: 0=上端(段a), 8=下端(段i) ← 段は上から下
            highlight_set: ハイライトするマスのセット
            selected_set: クリック選択中として塗るマスのセット
            destination_set: 行き先候補として塗るマスのセット
        """
        piece_parts = []
        selected_set = selected_set or set()
        destination_set = destination_set or set()
        piece_font_size = round(
            self.CELL_SIZE * self.theme.piece_font_size_ratio
        )
        # 文字の視覚的中心をマス中心に合わせるベースライン補正
        baseline_offset = round(self.CELL_SIZE * 0.16)

        # 盤面のX座標開始位置
        board_x_start = (
            self.MARGIN
            + self.HAND_AREA_WIDTH
            + self.GAP_BETWEEN_HAND_AND_BOARD
        )

        for row in range(9):
            for col in range(9):
                piece_id = board_id_positions[row][col]
                square_idx = row * 9 + col

                # 将棋の筋は右から左なので，描画時に列を反転
                visual_col = 8 - col  # col 0 → visual 8 (右端)

                # マスの塗り（駒の有無に関係なく描画）
                fills = []
                if square_idx in highlight_set:
                    fills.append((self.COLOR_HIGHLIGHT, 0.5))
                if square_idx in selected_set:
                    fills.append((self.COLOR_SELECTED, 1.0))
                if square_idx in destination_set:
                    fills.append((self.COLOR_DESTINATION, 1.0))
                for fill_color, fill_opacity in fills:
                    x_rect = (
                        board_x_start
                        + visual_col * self.CELL_SIZE
                    )
                    y_rect = self.MARGIN + row * self.CELL_SIZE
                    piece_parts.append(
                        f'<rect x="{x_rect}" y="{y_rect}" '
                        f'width="{self.CELL_SIZE}" height="{self.CELL_SIZE}" '
                        f'fill="{fill_color}" opacity="{fill_opacity}"/>'
                    )

                # 駒がない場合はスキップ
                if piece_id == PieceId.EMPTY:
                    continue

                # 駒の描画
                # boardIdPositionsはdomain PieceId形式
                # (先手=1-14, 後手=15-28)．判定・駒名変換は
                # piece_mapping に一本化
                is_white = is_white_piece(piece_id)
                symbol = get_piece_name_ja(piece_id)

                x = (
                    board_x_start
                    + visual_col * self.CELL_SIZE
                    + self.CELL_SIZE / 2
                )
                y = (
                    self.MARGIN
                    + row * self.CELL_SIZE
                    + self.CELL_SIZE / 2
                    + baseline_offset
                )  # 中央揃え調整

                color_class = (
                    "white-piece" if is_white else "black-piece"
                )

                # 後手駒は180度回転
                transform = (
                    "transform="
                    f'"rotate(180 {x} {y - baseline_offset})"'
                    if is_white
                    else ""
                )

                piece_parts.append(
                    f'<text x="{x}" y="{y}" '
                    f'class="piece {color_class}" '
                    f'font-size="{piece_font_size}" '
                    f"{transform}>{symbol}</text>"
                )

        return "\n".join(piece_parts)

    def _draw_pieces_in_hand(
        self, pieces_in_hand: list[int]
    ) -> str:
        """持ち駒を描画．

        Args:
            pieces_in_hand: 14要素の持ち駒配列
                インデックス0-6: 先手の持ち駒（歩香桂銀金角飛）
                インデックス7-13: 後手の持ち駒（歩香桂銀金角飛）
        """
        hand_parts = []

        # 後手の持ち駒（左側）
        hand_parts.append(
            self._draw_single_hand(
                pieces=pieces_in_hand[7:14],
                x_base=self.MARGIN,
                y_base=self.MARGIN,
                title="後手持ち駒",
                is_black=False,
            )
        )

        # 先手の持ち駒（右側）
        hand_parts.append(
            self._draw_single_hand(
                pieces=pieces_in_hand[:7],
                x_base=self.MARGIN
                + self.HAND_AREA_WIDTH
                + self.GAP_BETWEEN_HAND_AND_BOARD
                + self.BOARD_WIDTH
                + self.GAP_BETWEEN_HAND_AND_BOARD,
                y_base=self.MARGIN,
                title="先手持ち駒",
                is_black=True,
            )
        )

        return "\n".join(hand_parts)

    def _draw_single_hand(
        self,
        pieces: list[int],
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
        theme = self.theme

        # 持ち駒エリアの背景（視認性向上のため）
        parts.append(
            f'<rect x="{x_base}" y="{y_base}" '
            f'width="{self.HAND_AREA_WIDTH}" height="{self.BOARD_HEIGHT}" '
            f'fill="{theme.hand_bg}" '
            f'stroke="{theme.hand_border_color}" '
            f'stroke-width="{theme.hand_border_width}" '
            f'rx="{theme.corner_radius}" opacity="0.98"/>'
        )

        # タイトル背景バー
        parts.append(
            f'<rect x="{x_base}" y="{y_base}" '
            f'width="{self.HAND_AREA_WIDTH}" height="30" '
            f'fill="{theme.hand_title_bg}" '
            f'stroke="{theme.hand_border_color}" stroke-width="1" '
            f'rx="{theme.corner_radius}"/>'
        )

        # タイトル
        parts.append(
            f'<text x="{x_base + self.HAND_AREA_WIDTH / 2}" '
            f'y="{y_base + 20}" '
            f'class="coord" text-anchor="middle" font-weight="700" '
            f'font-size="14">{title}</text>'
        )

        # 各駒種の表示
        y_offset = y_base + 50  # タイトル(30px) + 余白(20px)
        color_class = (
            "black-piece" if is_black else "white-piece"
        )

        display_index = 0  # 実際に表示する駒のカウンター
        for piece_name, count in zip(
            self.HAND_PIECE_NAMES, pieces
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
                f'<text x="{x_base + self.HAND_AREA_WIDTH / 2}" '
                f'y="{y_offset + display_index * 30}" '
                f'class="piece {color_class}" '
                f'font-size="{theme.hand_font_size}">{text}</text>'
            )
            display_index += 1

        if display_index == 0 and theme.empty_hand_text:
            parts.append(
                f'<text x="{x_base + self.HAND_AREA_WIDTH / 2}" '
                f'y="{y_offset}" '
                f'class="piece {color_class}" '
                f'font-size="{theme.hand_font_size}">'
                f"{theme.empty_hand_text}</text>"
            )

        return "\n".join(parts)

    def _draw_coordinates(self) -> str:
        """盤面の座標（1-9列，1-9行）を描画．

        将棋の標準的な符号表記に従い，列番号は右から左へ（9→1）と表示する．
        配列では col=0 が右端（筋9），col=8 が左端（筋1）となっている．
        描画時に列を反転させているため，ラベルは右から左へ（9→1）と表示する．
        行番号は盤面の右側に1-9の数字で表示する．
        """
        coord_parts = []

        # 盤面のX座標開始位置
        board_x_start = (
            self.MARGIN
            + self.HAND_AREA_WIDTH
            + self.GAP_BETWEEN_HAND_AND_BOARD
        )

        # 列番号（9-1，右から左へ）
        # 描画時に列を反転させているため，ラベルも将棋の標準に合わせて右から左へ表示
        for visual_col in range(9):
            x = (
                board_x_start
                + visual_col * self.CELL_SIZE
                + self.CELL_SIZE / 2
            )
            y = self.MARGIN - 5

            # visual_col=0（左端）→ 筋9，visual_col=8（右端）→ 筋1
            col_number = 9 - visual_col

            # 洗練された背景バッジ
            if self.theme.coordinate_badges:
                coord_parts.append(
                    f'<rect x="{x - 10}" y="{y - 14}" '
                    f'width="20" height="18" '
                    f'fill="{self.theme.badge_bg}" '
                    f'stroke="{self.theme.badge_border}" '
                    f'stroke-width="1" rx="3" opacity="0.95"/>'
                )
            coord_parts.append(
                f'<text x="{x}" y="{y}" '
                f'class="coord" text-anchor="middle" '
                f'font-weight="600">{col_number}</text>'
            )

        # 行番号（1-9，盤面の右側）
        for row in range(9):
            # 盤面の右端から少し右に配置
            x = board_x_start + self.BOARD_WIDTH + 10
            y = int(
                self.MARGIN
                + row * self.CELL_SIZE
                + self.CELL_SIZE / 2
                + 5
            )

            row_number = row + 1  # 1-9 (not 0-8)

            # 洗練された背景バッジ
            if self.theme.coordinate_badges:
                coord_parts.append(
                    f'<rect x="{x - 10}" y="{y - 12}" '
                    f'width="20" height="18" '
                    f'fill="{self.theme.badge_bg}" '
                    f'stroke="{self.theme.badge_border}" '
                    f'stroke-width="1" rx="3" opacity="0.95"/>'
                )
            coord_parts.append(
                f'<text x="{x}" y="{y}" '
                f'class="coord" text-anchor="middle" '
                f'font-weight="600">{row_number}</text>'
            )

        return "\n".join(coord_parts)

    def _draw_arrows(
        self,
        arrows: list[ArrowSpec],
        pieces_in_hand: list[int],
        turn: Turn | None = None,
    ) -> str:
        """指し手を表す矢印群を描画する．

        Args:
            arrows: 描画する矢印のリスト（空の場合は空文字列を返す）
            pieces_in_hand: 持ち駒配列（駒打ちの矢印の始点計算に使用）
            turn: 手番．駒打ち矢印の始点をどちらの持ち駒エリアから
                引くかの判定に使用（None は先手側）

        Returns:
            矢印群のSVG文字列（不正なマス番号の矢印はスキップ）
        """
        if not arrows:
            return ""

        styles = self._arrow_styles(arrows)
        parts: list[str] = []
        for spec in arrows:
            endpoints = self._arrow_endpoints(
                spec.move, pieces_in_hand, turn
            )
            if endpoints is None:
                continue
            (from_x, from_y), (to_x, to_y) = endpoints
            marker = self._marker_id(
                styles.index((spec.color, spec.opacity))
            )
            parts.append(
                f'<line x1="{from_x}" y1="{from_y}" '
                f'x2="{to_x}" y2="{to_y}" '
                f'stroke="{spec.color}" '
                f'stroke-opacity="{spec.opacity}" '
                f'stroke-width="{spec.width}" '
                f'marker-end="url(#{marker})"/>'
            )
            if spec.label is not None:
                # ラベルは始点寄り（25%地点）に置き，矢じりと重ねない
                label_x = from_x + (to_x - from_x) * 0.25
                label_y = from_y + (to_y - from_y) * 0.25
                parts.append(
                    f'<circle cx="{label_x}" cy="{label_y}" r="9" '
                    f'fill="#ffffff" stroke="{spec.color}" '
                    f'stroke-width="1.5" opacity="0.9"/>'
                )
                parts.append(
                    f'<text x="{label_x}" y="{label_y + 4}" '
                    f'text-anchor="middle" font-size="12" '
                    f'font-weight="700" fill="#1a1a1a">'
                    f"{spec.label}</text>"
                )
        return "\n".join(parts)

    def _arrow_endpoints(
        self,
        move_arrow: MoveArrow,
        pieces_in_hand: list[int],
        turn: Turn | None = None,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """矢印の始点・終点のSVG座標を計算する．

        Args:
            move_arrow: 対象の指し手
            pieces_in_hand: 持ち駒配列（駒打ちの始点計算に使用）
            turn: 手番．後手番の駒打ちは後手持ち駒エリア（左側）を
                始点にする（None は先手側）

        Returns:
            ``((from_x, from_y), (to_x, to_y))``．
            マス番号が不正（0-80の範囲外）の場合は None
        """
        # Validate square indices (valid range is 0-80)
        if (
            move_arrow.to_square < 0
            or move_arrow.to_square > 80
        ):
            return None
        if move_arrow.from_square is not None and (
            move_arrow.from_square < 0
            or move_arrow.from_square > 80
        ):
            return None

        # 盤面のX座標開始位置
        board_x_start = (
            self.MARGIN
            + self.HAND_AREA_WIDTH
            + self.GAP_BETWEEN_HAND_AND_BOARD
        )

        # 移動先の座標を計算
        to_row, to_col = square_index_to_coords(
            move_arrow.to_square
        )
        to_visual_col = 8 - to_col  # 将棋は右から左

        to_x = (
            board_x_start
            + to_visual_col * self.CELL_SIZE
            + self.CELL_SIZE / 2
        )
        to_y = (
            self.MARGIN
            + to_row * self.CELL_SIZE
            + self.CELL_SIZE / 2
        )

        # 移動元の座標を計算
        from_x: float
        from_y: float
        if (
            move_arrow.is_drop
            and move_arrow.from_square is None
        ):
            # 駒打ちの場合: 手番側の持ち駒エリアから矢印を引く
            is_white_drop = turn == Turn.WHITE
            hand_pieces = (
                pieces_in_hand[7:14]
                if is_white_drop
                else pieces_in_hand[:7]
            )
            display_index = self._get_hand_piece_display_index(
                hand_pieces,
                move_arrow.drop_piece_type or 0,
            )
            if is_white_drop:
                # 後手持ち駒エリア（左側）の座標
                from_x = self.MARGIN + self.HAND_AREA_WIDTH / 2
            else:
                # 先手持ち駒エリア（右側）の座標
                from_x = (
                    self.MARGIN
                    + self.HAND_AREA_WIDTH
                    + self.GAP_BETWEEN_HAND_AND_BOARD
                    + self.BOARD_WIDTH
                    + self.GAP_BETWEEN_HAND_AND_BOARD
                    + self.HAND_AREA_WIDTH / 2
                )
            # タイトル(30px) + 余白(20px) + 各駒の位置
            from_y = self.MARGIN + 50 + display_index * 30
        else:
            # 通常の移動
            from_square = move_arrow.from_square or 0
            from_row, from_col = square_index_to_coords(
                from_square
            )
            from_visual_col = 8 - from_col

            from_x = (
                board_x_start
                + from_visual_col * self.CELL_SIZE
                + self.CELL_SIZE / 2
            )
            from_y = (
                self.MARGIN
                + from_row * self.CELL_SIZE
                + self.CELL_SIZE / 2
            )

        return ((from_x, from_y), (to_x, to_y))

    def _get_hand_piece_display_index(
        self,
        hand_pieces: list[int],
        piece_type: int,
    ) -> int:
        """持ち駒エリアにおける駒種の表示位置インデックスを取得する．

        持ち駒は枚数が1以上のものだけ表示されるため，
        表示位置は駒種のインデックスとは異なる．

        Args:
            hand_pieces: 7要素の持ち駒配列（歩香桂銀金角飛）
            piece_type: 駒種インデックス（0=歩, 1=香, ...）

        Returns:
            表示位置インデックス（0始まり）
        """
        display_index = 0
        for i in range(piece_type):
            if hand_pieces[i] > 0:
                display_index += 1
        return display_index

    def _draw_click_targets(
        self, pieces_in_hand: list[int]
    ) -> str:
        """クリック標的の透明 rect レイヤーを描画する．

        盤上の各マスに ``data-click="sq:{square}"`` (square は
        column-major = col * 9 + row)，持ち駒の各表示行に
        ``data-click="hand:{b|w}:{piece_type}"`` (piece_type は
        0=歩...6=飛) を持つ透明 rect を重ねる．矢印より後に描画する
        ことでクリックが常に標的に当たる．値の解釈は interface 層
        (analysis_gui) が行う．

        Args:
            pieces_in_hand: 14要素の持ち駒配列（表示行の計算に使用）

        Returns:
            クリック標的群のSVG文字列
        """
        parts: list[str] = []
        board_x_start = (
            self.MARGIN
            + self.HAND_AREA_WIDTH
            + self.GAP_BETWEEN_HAND_AND_BOARD
        )

        # 盤上のマス（描画は visual_col = 8 - col で反転）
        for row in range(9):
            for col in range(9):
                square = col * 9 + row  # column-major
                x = board_x_start + (8 - col) * self.CELL_SIZE
                y = self.MARGIN + row * self.CELL_SIZE
                parts.append(
                    f'<rect x="{x}" y="{y}" '
                    f'width="{self.CELL_SIZE}" '
                    f'height="{self.CELL_SIZE}" '
                    f'fill="transparent" '
                    f'data-click="sq:{square}" '
                    f'style="cursor:pointer"/>'
                )

        # 持ち駒（枚数 > 0 の表示行のみ．_draw_single_hand と同じ配置）
        hand_areas = [
            (
                "b",
                pieces_in_hand[:7],
                board_x_start
                + self.BOARD_WIDTH
                + self.GAP_BETWEEN_HAND_AND_BOARD,
            ),
            ("w", pieces_in_hand[7:14], self.MARGIN),
        ]
        for side, pieces, x_base in hand_areas:
            display_index = 0
            for piece_type, count in enumerate(pieces):
                if count == 0:
                    continue
                # テキスト行 (y_base + 50 + display_index * 30,
                # text-anchor=middle) を覆う矩形
                y_text = self.MARGIN + 50 + display_index * 30
                parts.append(
                    f'<rect x="{x_base + 10}" '
                    f'y="{y_text - 20}" '
                    f'width="{self.HAND_AREA_WIDTH - 20}" '
                    f'height="28" '
                    f'fill="transparent" '
                    f'data-click="hand:{side}:{piece_type}" '
                    f'style="cursor:pointer"/>'
                )
                display_index += 1

        return "\n".join(parts)
