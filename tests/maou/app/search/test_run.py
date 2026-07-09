"""SearchRunner (MCTS 探索ユースケース) のテスト．

Rust 拡張 (maou._rust.maou_search) を mock 評価器 (model_path=None) で駆動する．
"""

import pytest

from maou.app.search.run import SearchRunner

MATE_IN_1 = "4k4/9/4P4/9/9/9/9/9/9 b G 1"
PERPETUAL = "8k/9/7R1/9/9/9/9/9/4K4 b - 1"


class TestSearchRunner:
    def test_mate_in_1_is_proven(self) -> None:
        option = SearchRunner.SearchOption(
            sfen=MATE_IN_1,
            max_playouts=3000,
            threads=1,
            batch_size=4,
        )
        result = SearchRunner().run(option)
        assert result["Bestmove"] == "G*5b"
        assert result["WinRate"] == "1.0000"
        # 勝敗がほぼ決している水準 (evaluate.md: eval >= 3000)
        assert float(result["Eval"]) > 3000
        assert "stop=root_proven" in result["Stats"]
        # ルート評価 (エンジンビルド) の所要は warmup_ms として別掲される
        # (計測区間の nps/elapsed_ms には含まれない)
        assert "warmup_ms=" in result["Stats"]
        # best_move は候補リストの先頭に必ず含まれる
        assert (
            result["Candidates"]
            .splitlines()[0]
            .startswith("G*5b ")
        )
        assert "Board" in result

    def test_history_marks_perpetual_check_as_loss(
        self,
    ) -> None:
        # 王手往復 1 循環を対局履歴として渡すと，連続王手の千日手を
        # 完成させる 2c1c は負け評価になり best_move に選ばれない
        option = SearchRunner.SearchOption(
            sfen=PERPETUAL,
            moves=("2c1c", "1a2a", "1c2c", "2a1a"),
            max_playouts=500,
            threads=1,
        )
        result = SearchRunner().run(option)
        assert result["Bestmove"] != "2c1c"
        assert "repetitions=" in result["Stats"]

    def test_illegal_move_raises(self) -> None:
        option = SearchRunner.SearchOption(
            sfen=MATE_IN_1, moves=("1a1b",), max_playouts=10
        )
        with pytest.raises(ValueError):
            SearchRunner().run(option)
