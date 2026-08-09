"""定跡データベース．

指し手列(USI表記)から定跡名を検索する機能を提供する．
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpeningEntry:
    """定跡パターンの定義．

    Attributes:
        moves: USI形式の指し手列(前方一致で照合)
        name: 定跡名(例: "矢倉")
        category: カテゴリ(例: "相居飛車")
    """

    moves: tuple[str, ...]
    name: str
    category: str


@dataclass(frozen=True)
class OpeningInfo:
    """定跡の検索結果．

    Attributes:
        name: 定跡名
        category: カテゴリ
    """

    name: str
    category: str


_DEFAULT_OPENINGS: list[OpeningEntry] = [
    # --- 相居飛車系(長いパターン優先) ---
    OpeningEntry(
        moves=("7g7f", "3c3d", "2g2f", "8c8d", "2f2e", "8d8e"),
        name="横歩取り模様",
        category="相居飛車",
    ),
    OpeningEntry(
        moves=("2g2f", "8c8d", "2f2e", "8d8e"),
        name="相掛かり",
        category="相居飛車",
    ),
    OpeningEntry(
        moves=("7g7f", "8c8d", "7i6h"),
        name="矢倉",
        category="相居飛車",
    ),
    OpeningEntry(
        moves=("7g7f", "8c8d", "6i7h"),
        name="矢倉",
        category="相居飛車",
    ),
    OpeningEntry(
        moves=("7g7f", "3c3d", "8h2b+"),
        name="角換わり",
        category="相居飛車",
    ),
    # --- 振り飛車系 ---
    OpeningEntry(
        moves=(
            "7g7f",
            "3c3d",
            "6g6f",
            "3d3e",
        ),
        name="相振り飛車模様",
        category="振り飛車",
    ),
    OpeningEntry(
        moves=("7g7f", "3c3d", "6g6f"),
        name="振り飛車模様",
        category="振り飛車",
    ),
    OpeningEntry(
        moves=("7g7f", "3c3d", "2g2f", "5c5d", "2f2e", "5d5e"),
        name="ゴキゲン中飛車",
        category="振り飛車",
    ),
    OpeningEntry(
        moves=("5g5f",),
        name="先手中飛車",
        category="振り飛車",
    ),
]


# 最長一致で照合するため，長いパターンを先頭に並べておく．
# 並び順は表の性質でありインスタンスごとに変わらないので，
# 構築のたびにソートせずモジュールロード時に一度だけ確定させる．
_ENTRIES_LONGEST_FIRST: list[OpeningEntry] = sorted(
    _DEFAULT_OPENINGS,
    key=lambda e: len(e.moves),
    reverse=True,
)


class OpeningDatabase:
    """定跡データベース．

    デフォルトの定跡パターンを内蔵する．
    """

    def __init__(self) -> None:
        """初期化．デフォルトパターンを使用する．"""
        self._entries = _ENTRIES_LONGEST_FIRST

    def find_opening(
        self, moves: list[str]
    ) -> OpeningInfo | None:
        """指し手列から定跡を検索する．

        最長一致(longest prefix match)で照合し，
        最も具体的なパターンを返す．

        Important:
            **前提条件**: ``moves`` は平手初期局面から始まる手順で
            なければならない．``_DEFAULT_OPENINGS`` の全エントリが
            平手初期局面基準で定義されているためである．
            本メソッドはこの前提を検査しない (手順だけからは
            開始局面が分からない)．中盤局面などを起点とする手順を
            渡すと，偶然の前方一致で無関係な定跡名が返る．

            呼び出し側が起点が平手であることを保証すること．
            interface 層では
            ``game_graph_visualization.py`` の ``_root_is_startpos()``
            がグラフの根の SFEN を平手と比較してこれを担保している．

        Args:
            moves: USI形式の指し手列(平手初期局面からの手順)

        Returns:
            一致した定跡の情報．見つからない場合None．
        """
        for entry in self._entries:
            n = len(entry.moves)
            if (
                n <= len(moves)
                and tuple(moves[:n]) == entry.moves
            ):
                return OpeningInfo(
                    name=entry.name,
                    category=entry.category,
                )
        return None
