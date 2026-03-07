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


class TestOpeningDatabaseLoadJsonErrors:
    """OpeningDatabase.load_from_json のエラーパスのテスト．"""

    def test_file_not_found_returns_default(
        self, tmp_path: Path
    ) -> None:
        """存在しないファイルはデフォルトパターンで返す．"""
        path = tmp_path / "nonexistent.json"
        db = OpeningDatabase.load_from_json(path)
        # デフォルトパターンは利用可能
        result = db.find_opening(["7g7f", "8c8d", "7i6h"])
        assert result is not None
        assert result.name == "矢倉"

    def test_invalid_json_returns_default(
        self, tmp_path: Path
    ) -> None:
        """不正なJSON形式はデフォルトパターンで返す．"""
        path = tmp_path / "invalid.json"
        path.write_text("{broken", encoding="utf-8")
        db = OpeningDatabase.load_from_json(path)
        result = db.find_opening(["7g7f", "8c8d", "7i6h"])
        assert result is not None
        assert result.name == "矢倉"

    def test_non_list_root_returns_default(
        self, tmp_path: Path
    ) -> None:
        """ルートが配列でないJSONはデフォルトパターンで返す．"""
        path = tmp_path / "obj.json"
        path.write_text('{"key": "value"}', encoding="utf-8")
        db = OpeningDatabase.load_from_json(path)
        result = db.find_opening(["7g7f", "8c8d", "7i6h"])
        assert result is not None

    def test_moves_string_skipped(
        self, tmp_path: Path
    ) -> None:
        """moves が文字列のエントリはスキップされる．"""
        data = [
            {"moves": "7g7f", "name": "不正"},
            {"moves": ["2g2f"], "name": "正常"},
        ]
        path = tmp_path / "openings.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        db = OpeningDatabase.load_from_json(path)
        assert db.find_opening(["7g7f"]) is None
        assert db.find_opening(["2g2f"]) is not None

    def test_moves_with_non_string_elements_skipped(
        self, tmp_path: Path
    ) -> None:
        """moves の要素に数値が含まれるエントリはスキップされる．"""
        data = [
            {"moves": [7, "g7f"], "name": "不正"},
            {"moves": ["2g2f"], "name": "正常"},
        ]
        path = tmp_path / "openings.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        db = OpeningDatabase.load_from_json(path)
        assert db.find_opening(["2g2f"]) is not None

    def test_empty_moves_entry(
        self, tmp_path: Path
    ) -> None:
        """moves が空リストのエントリは全手順にマッチする．"""
        data = [{"moves": [], "name": "空手順"}]
        path = tmp_path / "openings.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        db = OpeningDatabase.load_from_json(path)
        # 空の moves は 0 長の前方一致 → 任意の手順にマッチ
        # ただしデフォルトの長いパターンが優先される
        result = db.find_opening(["7g7f", "8c8d", "7i6h"])
        assert result is not None
        assert result.name == "矢倉"  # 長いパターンが優先
        # デフォルトに該当しない手順では空手順がマッチ
        result2 = db.find_opening(["1g1f"])
        assert result2 is not None
        assert result2.name == "空手順"

    def test_long_custom_overrides_shorter_default(
        self, tmp_path: Path
    ) -> None:
        """デフォルトより長いカスタムパターンが優先される．"""
        data = [
            {
                "moves": [
                    "7g7f", "8c8d", "7i6h",
                    "3c3d", "6h7g",
                ],
                "name": "矢倉カスタム",
                "category": "テスト",
            }
        ]
        path = tmp_path / "openings.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        db = OpeningDatabase.load_from_json(path)
        # 長いパターンが優先
        result = db.find_opening(
            ["7g7f", "8c8d", "7i6h", "3c3d", "6h7g"]
        )
        assert result is not None
        assert result.name == "矢倉カスタム"
        # 短い手順ではデフォルトの矢倉がマッチ
        result2 = db.find_opening(
            ["7g7f", "8c8d", "7i6h"]
        )
        assert result2 is not None
        assert result2.name == "矢倉"


class TestOpeningInfo:
    """OpeningInfo のテスト．"""

    def test_frozen_dataclass(self) -> None:
        """OpeningInfoはイミュータブルである．"""
        info = OpeningInfo(name="矢倉", category="相居飛車")
        assert info.name == "矢倉"
        assert info.category == "相居飛車"
        with pytest.raises(FrozenInstanceError):
            info.name = "角換わり"  # type: ignore[misc]
