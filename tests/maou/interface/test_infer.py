"""`maou.interface.infer` のテスト．

推論は Rust (`maou._rust.maou_search.evaluate`) が担うため，ここでは
Python 側に残った責務 (引数の受け渡し・表示整形・入力検証) を検証する．
"""

import math

import pytest

from maou._rust.maou_search import evaluate as rust_evaluate
from maou._rust.maou_search import winrate_to_eval
from maou.interface.infer import infer

HIRATE = (
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/"
    "PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
)
CHECKMATED = "4k4/4G4/4G4/9/9/9/9/9/4K4 w - 1"


class TestInfer:
    def test_returns_all_sections(self) -> None:
        out = infer(num_moves=5, sfen=HIRATE)
        assert "Policy:" in out
        assert "Eval:" in out
        assert "WinRate:" in out
        assert "先手の持駒" in out

    def test_negative_num_moves_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            infer(num_moves=-1, sfen=HIRATE)

    def test_invalid_sfen_raises(self) -> None:
        with pytest.raises(Exception):
            infer(num_moves=5, sfen="not-a-sfen")

    def test_eval_matches_winrate_conversion(self) -> None:
        """表示される Eval が WinRate の Ponanza 変換と一致する．"""
        result = rust_evaluate(HIRATE, num_moves=1)
        expected = winrate_to_eval(result.winrate)
        out = infer(num_moves=1, sfen=HIRATE)
        eval_line = next(
            line
            for line in out.splitlines()
            if line.startswith("Eval:")
        )
        assert eval_line == f"Eval: {expected:.2f}"


class TestRustEvaluate:
    def test_priors_are_normalized_over_legal_moves(
        self,
    ) -> None:
        result = rust_evaluate(HIRATE, num_moves=30)
        assert result.legal_move_count == 30
        assert len(result.moves) == 30
        total = sum(m.prior for m in result.moves)
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_moves_are_sorted_by_prior_descending(
        self,
    ) -> None:
        result = rust_evaluate(HIRATE, num_moves=30)
        priors = [m.prior for m in result.moves]
        assert priors == sorted(priors, reverse=True)

    def test_num_moves_truncates_but_keeps_legal_count(
        self,
    ) -> None:
        result = rust_evaluate(HIRATE, num_moves=3)
        assert len(result.moves) == 3
        assert result.legal_move_count == 30

    def test_num_moves_over_legal_count_is_clamped(
        self,
    ) -> None:
        result = rust_evaluate(HIRATE, num_moves=1000)
        assert len(result.moves) == 30

    def test_checkmated_position_returns_no_moves(
        self,
    ) -> None:
        result = rust_evaluate(CHECKMATED, num_moves=5)
        assert result.moves == []
        assert result.legal_move_count == 0
        assert 0.0 <= result.winrate <= 1.0

    def test_winrate_and_eval_agree(self) -> None:
        result = rust_evaluate(HIRATE, num_moves=1)
        assert result.eval_score == pytest.approx(
            winrate_to_eval(result.winrate)
        )


class TestWinrateToEval:
    def test_midpoint_is_zero(self) -> None:
        assert winrate_to_eval(0.5) == 0.0

    def test_sign_follows_winrate(self) -> None:
        assert winrate_to_eval(0.9) > 0
        assert winrate_to_eval(0.1) < 0

    def test_ponanza_scale(self) -> None:
        # sigmoid(1) に対応する勝率が eval = 600
        winrate = 1 / (1 + math.exp(-1.0))
        assert winrate_to_eval(winrate) == pytest.approx(600.0)

    def test_saturates_without_diverging(self) -> None:
        assert math.isfinite(winrate_to_eval(0.0))
        assert math.isfinite(winrate_to_eval(1.0))
