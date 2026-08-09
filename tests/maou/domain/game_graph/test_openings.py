"""定跡データベースのテスト．"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from maou.domain.game_graph.openings import (
    OpeningDatabase,
    OpeningInfo,
)


class TestOpeningDatabase:
    """OpeningDatabase のテスト．"""

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

    def test_no_match(self) -> None:
        """一致するパターンがない場合はNoneを返す．"""
        db = OpeningDatabase()
        result = db.find_opening(["1g1f"])
        assert result is None

    def test_longest_prefix_wins(self) -> None:
        """デフォルト定跡で最長一致パターンが優先される．"""
        db = OpeningDatabase()
        # 7g7f だけではどのデフォルト定跡にもならない
        # 7g7f, 8c8d, 7i6h で矢倉がマッチ
        result = db.find_opening(
            ["7g7f", "8c8d", "7i6h", "3c3d"]
        )
        assert result is not None
        assert result.name == "矢倉"

    def test_origin_precondition_is_not_enforced_here(
        self,
    ) -> None:
        """平手起点という前提条件はドメイン側では検査しない．

        ``find_opening`` は手順の前方一致だけを見るので，起点が
        平手でなくても一致すれば定跡名を返す．起点の検査は
        interface 層の ``_root_is_startpos()``
        (``game_graph_visualization.py``) が担う，という設計を固定
        する．ここに二重のガードを入れないこと．
        """
        db = OpeningDatabase()
        # 中盤局面から指された「5g5f」でも先手中飛車として一致する．
        # 手順だけからは起点が分からないため，これは仕様どおり．
        result = db.find_opening(["5g5f"])
        assert result is not None
        assert result.name == "先手中飛車"


class TestOpeningInfo:
    """OpeningInfo のテスト．"""

    def test_frozen_dataclass(self) -> None:
        """OpeningInfoはイミュータブルである．"""
        info = OpeningInfo(name="矢倉", category="相居飛車")
        assert info.name == "矢倉"
        assert info.category == "相居飛車"
        with pytest.raises(FrozenInstanceError):
            info.name = "角換わり"  # type: ignore[misc]
