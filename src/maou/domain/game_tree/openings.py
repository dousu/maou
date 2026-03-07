"""定跡データベース．

指し手列(USI表記)から定跡名を検索する機能を提供する．
デフォルトの定跡パターンを内蔵し，外部JSONファイルからの追加読み込みも対応する．
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


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


class OpeningDatabase:
    """定跡データベース．

    デフォルトの定跡パターンを内蔵し，
    外部JSONファイルからの追加パターン読み込みにも対応する．
    """

    def __init__(
        self,
        entries: list[OpeningEntry] | None = None,
    ) -> None:
        """初期化．

        Args:
            entries: 定跡パターンのリスト．
                Noneの場合はデフォルトパターンを使用．
        """
        self._entries = sorted(
            entries if entries is not None else _DEFAULT_OPENINGS,
            key=lambda e: len(e.moves),
            reverse=True,
        )

    def find_opening(
        self, moves: list[str]
    ) -> OpeningInfo | None:
        """指し手列から定跡を検索する．

        最長一致(longest prefix match)で照合し，
        最も具体的なパターンを返す．

        Args:
            moves: USI形式の指し手列(ルートからの手順)

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

    @classmethod
    def load_from_json(cls, path: Path) -> OpeningDatabase:
        """JSONファイルから定跡データベースを読み込む．

        JSON形式:
        ```json
        [
            {
                "moves": ["7g7f", "3c3d", "6g6f"],
                "name": "振り飛車模様",
                "category": "振り飛車"
            }
        ]
        ```

        デフォルトパターンとマージされる(JSONのパターンが優先)．

        Args:
            path: JSONファイルのパス

        Returns:
            読み込み済みのOpeningDatabase
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            logger.warning(
                "定跡JSON: ルートが配列ではありません: %s",
                path,
            )
            return cls()

        custom_entries: list[OpeningEntry] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                logger.warning(
                    "定跡JSON: エントリ %d はdictではないためスキップ",
                    i,
                )
                continue
            if "moves" not in item or "name" not in item:
                logger.warning(
                    "定跡JSON: エントリ %d に必須キー"
                    "(moves/name)がないためスキップ",
                    i,
                )
                continue
            if not isinstance(item["moves"], list):
                logger.warning(
                    "定跡JSON: エントリ %d の moves が"
                    "リストではないためスキップ",
                    i,
                )
                continue
            if not all(
                isinstance(m, str) for m in item["moves"]
            ):
                logger.warning(
                    "定跡JSON: エントリ %d の moves に"
                    "文字列でない要素が含まれるためスキップ",
                    i,
                )
                continue
            custom_entries.append(
                OpeningEntry(
                    moves=tuple(item["moves"]),
                    name=item["name"],
                    category=item.get("category", ""),
                )
            )

        # カスタムパターンをデフォルトの前に配置(優先)
        merged = custom_entries + list(_DEFAULT_OPENINGS)
        return cls(entries=merged)
