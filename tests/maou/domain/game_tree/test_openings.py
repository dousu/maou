"""定跡データベースのテスト．"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from maou.domain.game_tree.openings import (
    OpeningDatabase,
    OpeningEntry,
    OpeningInfo,
)


class TestOpeningDatabase:
    """OpeningDatabase のテスト．"""

    def test_find_exact_match(self) -> None:
        """完全一致するパターンを検索する．"""
        entries = [
            OpeningEntry(
                moves=("7g7f", "3c3d", "6g6f"),
                name="振り飛車模様",
                category="振り飛車",
            ),
        ]
        db = OpeningDatabase(entries)
        result = db.find_opening(["7g7f", "3c3d", "6g6f"])
        assert result is not None
        assert result.name == "振り飛車模様"
        assert result.category == "振り飛車"

    def test_find_prefix_match(self) -> None:
        """前方一致するパターンを検索する．"""
        entries = [
            OpeningEntry(
                moves=("7g7f", "3c3d"),
                name="相角道",
                category="相居飛車",
            ),
        ]
        db = OpeningDatabase(entries)
        result = db.find_opening(
            ["7g7f", "3c3d", "6g6f", "8c8d"]
        )
        assert result is not None
        assert result.name == "相角道"

    def test_longest_prefix_wins(self) -> None:
        """最長一致パターンが優先される．"""
        entries = [
            OpeningEntry(
                moves=("7g7f",),
                name="角道",
                category="不明",
            ),
            OpeningEntry(
                moves=("7g7f", "3c3d", "6g6f"),
                name="振り飛車模様",
                category="振り飛車",
            ),
        ]
        db = OpeningDatabase(entries)
        result = db.find_opening(
            ["7g7f", "3c3d", "6g6f", "8c8d"]
        )
        assert result is not None
        assert result.name == "振り飛車模様"

    def test_no_match(self) -> None:
        """一致するパターンがない場合はNoneを返す．"""
        entries = [
            OpeningEntry(
                moves=("7g7f", "3c3d", "6g6f"),
                name="振り飛車模様",
                category="振り飛車",
            ),
        ]
        db = OpeningDatabase(entries)
        result = db.find_opening(["2g2f", "8c8d"])
        assert result is None

    def test_empty_moves(self) -> None:
        """空の指し手列はNoneを返す．"""
        db = OpeningDatabase()
        result = db.find_opening([])
        assert result is None

    def test_default_openings_loaded(self) -> None:
        """デフォルト定跡が読み込まれる．"""
        db = OpeningDatabase()
        # 振り飛車模様: 7g7f, 3c3d, 6g6f
        result = db.find_opening(["7g7f", "3c3d", "6g6f"])
        assert result is not None
        assert result.name == "振り飛車模様"

    def test_default_yagura(self) -> None:
        """デフォルト定跡の矢倉を検索する．"""
        db = OpeningDatabase()
        result = db.find_opening(["7g7f", "8c8d", "7i6h"])
        assert result is not None
        assert result.name == "矢倉"

    def test_default_aigakari(self) -> None:
        """デフォルト定跡の相掛かりを検索する．"""
        db = OpeningDatabase()
        result = db.find_opening(
            ["2g2f", "8c8d", "2f2e", "8d8e"]
        )
        assert result is not None
        assert result.name == "相掛かり"


class TestOpeningDatabaseLoadJson:
    """OpeningDatabase.load_from_json のテスト．"""

    def test_load_custom_json(self, tmp_path: Path) -> None:
        """JSONファイルからカスタムパターンを読み込む．"""
        data = [
            {
                "moves": ["7g7f", "3c3d"],
                "name": "カスタム定跡",
                "category": "テスト",
            }
        ]
        path = tmp_path / "openings.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        db = OpeningDatabase.load_from_json(path)
        result = db.find_opening(["7g7f", "3c3d"])
        assert result is not None
        assert result.name == "カスタム定跡"

    def test_custom_overrides_default(
        self, tmp_path: Path
    ) -> None:
        """カスタムパターンがデフォルトより優先される．"""
        data = [
            {
                "moves": ["7g7f", "3c3d", "6g6f"],
                "name": "カスタム振り飛車",
                "category": "テスト",
            }
        ]
        path = tmp_path / "openings.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        db = OpeningDatabase.load_from_json(path)
        result = db.find_opening(["7g7f", "3c3d", "6g6f"])
        assert result is not None
        assert result.name == "カスタム振り飛車"

    def test_default_patterns_still_available(
        self, tmp_path: Path
    ) -> None:
        """JSONロード後もデフォルトパターンが利用できる．"""
        data = [
            {
                "moves": ["1g1f"],
                "name": "端歩",
                "category": "テスト",
            }
        ]
        path = tmp_path / "openings.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        db = OpeningDatabase.load_from_json(path)
        # デフォルトの矢倉パターンが残っている
        result = db.find_opening(["7g7f", "8c8d", "7i6h"])
        assert result is not None
        assert result.name == "矢倉"


class TestOpeningInfo:
    """OpeningInfo のテスト．"""

    def test_frozen_dataclass(self) -> None:
        """OpeningInfoはイミュータブルである．"""
        info = OpeningInfo(name="矢倉", category="相居飛車")
        assert info.name == "矢倉"
        assert info.category == "相居飛車"
        with pytest.raises(FrozenInstanceError):
            info.name = "角換わり"  # type: ignore[misc]
