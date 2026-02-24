"""evaluate コマンドの --engine-path 関連テスト．"""

import pytest
from click.testing import CliRunner

from maou.infra.console.evaluate_board import evaluate_board


class TestEvaluateBoardEnginePath:
    """evaluate コマンドの --engine-path オプションテスト．"""

    def test_evaluate_has_engine_path_option(self) -> None:
        """--engine-path オプションが存在すること．"""
        param_names = [p.name for p in evaluate_board.params]
        assert "engine_path" in param_names

    def test_evaluate_model_path_not_required(self) -> None:
        """--model-path が required=False であること．"""
        for param in evaluate_board.params:
            if param.name == "model_path":
                assert param.required is False
                break
        else:
            pytest.fail("model_path param not found")

    def test_evaluate_requires_model_path_or_engine_path(
        self,
    ) -> None:
        """--model-path と --engine-path の両方が未指定でエラーになること．

        handle_exception デコレータが全例外をキャッチするため，
        click.UsageError が logger.exception 経由で記録される．
        CliRunner の result.output には正常な推論結果が含まれないことで確認する．
        """
        runner = CliRunner()
        result = runner.invoke(
            evaluate_board,
            [
                "--sfen",
                "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            ],
        )
        # handle_exception が例外をキャッチするため exit_code=0 だが，
        # 正常な推論結果（Policy, Eval, WinRate）は出力されない
        assert "Policy" not in result.output
        assert "Eval" not in result.output
        assert "WinRate" not in result.output
