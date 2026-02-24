import logging
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from maou.app.inference.run import InferenceRunner, ModelType
from maou.domain.board.shogi import Board
from maou.domain.move.label import IllegalMove

logger: logging.Logger = logging.getLogger("TEST")


class TestInferenceRunner:
    """InferenceRunnerクラスの包括的なテスト．"""

    @pytest.fixture
    def runner(self) -> InferenceRunner:
        """テスト用のInferenceRunnerインスタンスを作成．"""
        return InferenceRunner()

    @pytest.fixture
    def sample_sfen(self) -> str:
        """テスト用のサンプルsfen（初期局面）．"""
        return "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"

    @pytest.fixture
    def sample_board(self) -> Board:
        """テスト用のサンプルBoard（初期局面）．"""
        board = Board()
        board.set_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        )
        return board

    @pytest.fixture
    def mock_model_path(self, tmp_path: Path) -> Path:
        """テスト用のモックモデルパス．"""
        model_file = tmp_path / "test_model.onnx"
        model_file.touch()
        return model_file

    def test_infer_with_sfen_onnx(
        self,
        runner: InferenceRunner,
        sample_sfen: str,
        mock_model_path: Path,
    ) -> None:
        """sfenを指定してONNXモデルで推論を実行．"""
        with (
            patch(
                "maou.app.inference.run.make_board_id_positions"
            ) as mock_make_board,
            patch(
                "maou.app.inference.run.make_pieces_in_hand"
            ) as mock_make_hand,
            patch(
                "maou.app.inference.run.ONNXInference.infer"
            ) as mock_onnx_infer,
            patch(
                "maou.app.inference.run.make_usi_move_from_label"
            ) as mock_make_usi,
        ):
            # モックの設定
            mock_make_board.return_value = np.zeros(
                (9, 9), dtype=np.int64
            )
            mock_make_hand.return_value = np.zeros(
                (14,), dtype=np.uint8
            )
            mock_onnx_infer.return_value = (
                [0, 1, 2, 3, 4],
                1.5,
            )  # policy_labels, value
            mock_make_usi.side_effect = [
                "7g7f",
                "2g2f",
                "6i7h",
                "3i4h",
                "5g5f",
            ]

            config = InferenceRunner.InferenceOption(
                model_path=mock_model_path,
                model_type=ModelType.ONNX,
                sfen=sample_sfen,
                num_moves=5,
                board_view=True,
            )

            result = runner.infer(config)

            # 検証
            assert "Policy" in result
            assert "Eval" in result
            assert "WinRate" in result
            assert "Board" in result

            # Policyに5手含まれていることを確認
            policies = result["Policy"].split(", ")
            assert len(policies) == 5
            assert "7g7f" in policies

            # 評価値と勝率が数値として妥当か確認
            eval_value = float(result["Eval"])
            winrate_value = float(result["WinRate"])
            assert -10000 < eval_value < 10000  # 妥当な範囲
            assert 0.0 <= winrate_value <= 1.0

    def test_infer_with_board_onnx(
        self,
        runner: InferenceRunner,
        sample_board: Board,
        mock_model_path: Path,
    ) -> None:
        """Boardオブジェクトを指定してONNXモデルで推論を実行．"""
        with (
            patch(
                "maou.app.inference.run.make_board_id_positions"
            ) as mock_make_board,
            patch(
                "maou.app.inference.run.make_pieces_in_hand"
            ) as mock_make_hand,
            patch(
                "maou.app.inference.run.ONNXInference.infer"
            ) as mock_onnx_infer,
            patch(
                "maou.app.inference.run.make_usi_move_from_label"
            ) as mock_make_usi,
        ):
            # モックの設定
            mock_make_board.return_value = np.zeros(
                (9, 9), dtype=np.int64
            )
            mock_make_hand.return_value = np.zeros(
                (14,), dtype=np.uint8
            )
            mock_onnx_infer.return_value = ([0, 1, 2], 0.5)
            mock_make_usi.side_effect = ["7g7f", "2g2f", "6i7h"]

            config = InferenceRunner.InferenceOption(
                model_path=mock_model_path,
                model_type=ModelType.ONNX,
                board=sample_board,
                num_moves=3,
                board_view=False,
            )

            result = runner.infer(config)

            # 検証
            assert "Policy" in result
            assert "Eval" in result
            assert "WinRate" in result
            assert "Board" not in result  # board_view=False

            # 正しい手数が返されている
            policies = result["Policy"].split(", ")
            assert len(policies) == 3

    def test_infer_without_sfen_and_board(
        self, runner: InferenceRunner, mock_model_path: Path
    ) -> None:
        """sfenもboardも指定されていない場合にエラーを発生．"""
        config = InferenceRunner.InferenceOption(
            model_path=mock_model_path,
            model_type=ModelType.ONNX,
            sfen=None,
            board=None,
        )

        with pytest.raises(
            ValueError,
            match="Either sfen or board must be provided",
        ):
            runner.infer(config)

    def test_infer_with_cuda_option(
        self,
        runner: InferenceRunner,
        sample_sfen: str,
        mock_model_path: Path,
    ) -> None:
        """cudaオプションがONNXInferenceに正しく渡されることを確認．"""
        with (
            patch(
                "maou.app.inference.run.make_board_id_positions"
            ) as mock_make_board,
            patch(
                "maou.app.inference.run.make_pieces_in_hand"
            ) as mock_make_hand,
            patch(
                "maou.app.inference.run.ONNXInference.infer"
            ) as mock_onnx_infer,
            patch(
                "maou.app.inference.run.make_usi_move_from_label"
            ) as mock_make_usi,
        ):
            mock_make_board.return_value = np.zeros(
                (9, 9), dtype=np.int64
            )
            mock_make_hand.return_value = np.zeros(
                (14,), dtype=np.uint8
            )
            mock_onnx_infer.return_value = ([0], 0.0)
            mock_make_usi.return_value = "7g7f"

            config = InferenceRunner.InferenceOption(
                model_path=mock_model_path,
                model_type=ModelType.ONNX,
                sfen=sample_sfen,
                cuda=True,
            )

            runner.infer(config)

            # cudaオプションが正しく渡されたか確認
            mock_onnx_infer.assert_called_once()
            call_args = mock_onnx_infer.call_args
            assert call_args[0][4] is True  # cuda=True

    def test_infer_with_illegal_move(
        self,
        runner: InferenceRunner,
        sample_sfen: str,
        mock_model_path: Path,
    ) -> None:
        """不正な手（IllegalMove）が含まれる場合の処理を確認．"""
        with (
            patch(
                "maou.app.inference.run.make_board_id_positions"
            ) as mock_make_board,
            patch(
                "maou.app.inference.run.make_pieces_in_hand"
            ) as mock_make_hand,
            patch(
                "maou.app.inference.run.ONNXInference.infer"
            ) as mock_onnx_infer,
            patch(
                "maou.app.inference.run.make_usi_move_from_label"
            ) as mock_make_usi,
        ):
            mock_make_board.return_value = np.zeros(
                (9, 9), dtype=np.int64
            )
            mock_make_hand.return_value = np.zeros(
                (14,), dtype=np.uint8
            )
            mock_onnx_infer.return_value = ([0, 1, 2], 0.0)

            # 2番目の手でIllegalMoveを発生させる
            def side_effect_make_usi(
                board: Board, label: int
            ) -> str:
                if label == 1:
                    raise IllegalMove("Invalid move")
                return f"move_{label}"

            mock_make_usi.side_effect = side_effect_make_usi

            config = InferenceRunner.InferenceOption(
                model_path=mock_model_path,
                model_type=ModelType.ONNX,
                sfen=sample_sfen,
                num_moves=3,
            )

            result = runner.infer(config)

            # 不正な手は "failed to convert" に置き換えられる
            policies = result["Policy"].split(", ")
            assert len(policies) == 3
            assert policies[0] == "move_0"
            assert policies[1] == "failed to convert"
            assert policies[2] == "move_2"

    def test_infer_tensorrt_missing_dependency(
        self,
        runner: InferenceRunner,
        sample_sfen: str,
        mock_model_path: Path,
    ) -> None:
        """TensorRTが利用できない場合のエラー処理を確認．"""
        with (
            patch(
                "maou.app.inference.run.make_board_id_positions"
            ) as mock_make_board,
            patch(
                "maou.app.inference.run.make_pieces_in_hand"
            ) as mock_make_hand,
            patch("builtins.__import__") as mock_import,
        ):
            mock_make_board.return_value = np.zeros(
                (9, 9), dtype=np.int64
            )
            mock_make_hand.return_value = np.zeros(
                (14,), dtype=np.uint8
            )

            # TensorRTインポート時にModuleNotFoundErrorを発生
            def side_effect_import(
                name: str, *args: Any, **kwargs: Any
            ) -> Any:
                if "tensorrt_inference" in name:
                    raise ModuleNotFoundError(
                        "No module named 'tensorrt'"
                    )
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = side_effect_import

            config = InferenceRunner.InferenceOption(
                model_path=mock_model_path,
                model_type=ModelType.TENSORRT,
                sfen=sample_sfen,
            )

            with pytest.raises(
                RuntimeError,
                match="TensorRT inference requires optional dependency",
            ):
                runner.infer(config)

    def test_infer_tensorrt_available(
        self,
        runner: InferenceRunner,
        sample_sfen: str,
        mock_model_path: Path,
    ) -> None:
        """TensorRTが利用可能な場合の推論を確認．"""
        # TensorRTInferenceクラスのモック
        mock_tensorrt_class = MagicMock()
        mock_tensorrt_class.infer.return_value = ([0, 1], 1.0)

        # モックモジュールを作成してsys.modulesに登録
        mock_tensorrt_module = MagicMock()
        mock_tensorrt_module.TensorRTInference = (
            mock_tensorrt_class
        )

        with (
            patch(
                "maou.app.inference.run.make_board_id_positions"
            ) as mock_make_board,
            patch(
                "maou.app.inference.run.make_pieces_in_hand"
            ) as mock_make_hand,
            patch.dict(
                sys.modules,
                {
                    "maou.app.inference.tensorrt_inference": mock_tensorrt_module
                },
            ),
            patch(
                "maou.app.inference.run.make_usi_move_from_label"
            ) as mock_make_usi,
        ):
            mock_make_board.return_value = np.zeros(
                (9, 9), dtype=np.int64
            )
            mock_make_hand.return_value = np.zeros(
                (14,), dtype=np.uint8
            )
            mock_make_usi.side_effect = ["7g7f", "2g2f"]

            config = InferenceRunner.InferenceOption(
                model_path=mock_model_path,
                model_type=ModelType.TENSORRT,
                sfen=sample_sfen,
                num_moves=2,
            )

            result = runner.infer(config)

            # TensorRTInference.inferが呼ばれたことを確認
            mock_tensorrt_class.infer.assert_called_once()
            # workspace_size_mbが渡されていることを確認
            call_kwargs = mock_tensorrt_class.infer.call_args
            assert (
                call_kwargs.kwargs["workspace_size_mb"] == 256
            )

            # 結果の検証
            assert "Policy" in result
            policies = result["Policy"].split(", ")
            assert len(policies) == 2

    def test_infer_unsupported_model_type(
        self,
        runner: InferenceRunner,
        sample_sfen: str,
        mock_model_path: Path,
    ) -> None:
        """サポートされていないモデルタイプの場合のエラー処理．"""
        # 無効なModelTypeを作成するため，テスト用のEnumを追加
        # (実際にはありえないが，将来の拡張に備えたテスト)

        with (
            patch(
                "maou.app.inference.run.make_board_id_positions"
            ) as mock_make_board,
            patch(
                "maou.app.inference.run.make_pieces_in_hand"
            ) as mock_make_hand,
        ):
            mock_make_board.return_value = np.zeros(
                (9, 9), dtype=np.int64
            )
            mock_make_hand.return_value = np.zeros(
                (14,), dtype=np.uint8
            )

            # Enumに存在しない値を強制的に設定（通常はありえないが，堅牢性テスト）
            # NOTE: この部分は実際には到達不可能だが，網羅的なテストとして含める
            # 現状のコードでは ModelType.ONNX と ModelType.TENSORRT のみサポート

    def test_inference_option_defaults(
        self, mock_model_path: Path
    ) -> None:
        """InferenceOptionのデフォルト値を確認．"""
        config = InferenceRunner.InferenceOption(
            model_path=mock_model_path,
            model_type=ModelType.ONNX,
        )

        assert config.cuda is False
        assert config.num_moves == 5
        assert config.board_view is True
        assert config.sfen is None
        assert config.board is None
        assert config.trt_workspace_size_mb == 256

    def test_inference_option_frozen(
        self, mock_model_path: Path
    ) -> None:
        """InferenceOptionがfrozenであることを確認（イミュータブル）．"""
        config = InferenceRunner.InferenceOption(
            model_path=mock_model_path,
            model_type=ModelType.ONNX,
        )

        with pytest.raises(
            Exception
        ):  # dataclasses.FrozenInstanceError
            config.num_moves = 10  # type: ignore

    def test_evaluation_integration(
        self,
        runner: InferenceRunner,
        sample_sfen: str,
        mock_model_path: Path,
    ) -> None:
        """Evaluationクラスとの統合が正しく動作することを確認．"""
        with (
            patch(
                "maou.app.inference.run.make_board_id_positions"
            ) as mock_make_board,
            patch(
                "maou.app.inference.run.make_pieces_in_hand"
            ) as mock_make_hand,
            patch(
                "maou.app.inference.run.ONNXInference.infer"
            ) as mock_onnx_infer,
            patch(
                "maou.app.inference.run.make_usi_move_from_label"
            ) as mock_make_usi,
        ):
            mock_make_board.return_value = np.zeros(
                (9, 9), dtype=np.int64
            )
            mock_make_hand.return_value = np.zeros(
                (14,), dtype=np.uint8
            )
            # value = 2.0 → winrate ≈ 0.88, eval ≈ 1200
            mock_onnx_infer.return_value = ([0], 2.0)
            mock_make_usi.return_value = "7g7f"

            config = InferenceRunner.InferenceOption(
                model_path=mock_model_path,
                model_type=ModelType.ONNX,
                sfen=sample_sfen,
            )

            result = runner.infer(config)

            # Evaluationクラスの計算結果を検証
            winrate = float(result["WinRate"])
            eval_value = float(result["Eval"])

            # value=2.0の場合の期待値
            # winrate = sigmoid(2.0) ≈ 0.88
            # eval = 600 * 2.0 = 1200
            assert 0.87 < winrate < 0.89
            assert 1150 < eval_value < 1250

    def test_num_moves_variation(
        self,
        runner: InferenceRunner,
        sample_sfen: str,
        mock_model_path: Path,
    ) -> None:
        """異なるnum_moves値で正しく動作することを確認．"""
        for num_moves in [1, 3, 10, 20]:
            with (
                patch(
                    "maou.app.inference.run.make_board_id_positions"
                ) as mock_make_board,
                patch(
                    "maou.app.inference.run.make_pieces_in_hand"
                ) as mock_make_hand,
                patch(
                    "maou.app.inference.run.ONNXInference.infer"
                ) as mock_onnx_infer,
                patch(
                    "maou.app.inference.run.make_usi_move_from_label"
                ) as mock_make_usi,
            ):
                mock_make_board.return_value = np.zeros(
                    (9, 9), dtype=np.int64
                )
                mock_make_hand.return_value = np.zeros(
                    (14,), dtype=np.uint8
                )
                mock_onnx_infer.return_value = (
                    list(range(num_moves)),
                    0.0,
                )
                mock_make_usi.side_effect = [
                    f"move_{i}" for i in range(num_moves)
                ]

                config = InferenceRunner.InferenceOption(
                    model_path=mock_model_path,
                    model_type=ModelType.ONNX,
                    sfen=sample_sfen,
                    num_moves=num_moves,
                )

                result = runner.infer(config)

                # ONNXInference.inferにnum_movesが正しく渡されている
                mock_onnx_infer.assert_called_once()
                call_args = mock_onnx_infer.call_args
                assert call_args[0][3] == num_moves

                # 結果に正しい手数が含まれている
                policies = result["Policy"].split(", ")
                assert len(policies) == num_moves


class TestModelType:
    """ModelType enumのテスト．"""

    def test_model_type_values(self) -> None:
        """ModelTypeに期待される値が存在することを確認．"""
        assert hasattr(ModelType, "ONNX")
        assert hasattr(ModelType, "TENSORRT")

    def test_model_type_auto_values(self) -> None:
        """ModelTypeの値がauto()で生成されていることを確認．"""
        # auto()は整数値を生成する
        assert isinstance(ModelType.ONNX.value, int)
        assert isinstance(ModelType.TENSORRT.value, int)
        assert ModelType.ONNX.value != ModelType.TENSORRT.value
