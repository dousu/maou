from pathlib import Path
from typing import Any, Generator, Literal

import numpy as np
import pytest

from maou.app.common.data_io_service import (
    load_hcpe_array,
    load_preprocessing_array,
)
from maou.app.pre_process import hcpe_transform
from maou.app.pre_process.transform import Transform
from maou.domain.board import shogi
from maou.domain.data.schema import (
    create_empty_hcpe_array,
    get_preprocessing_dtype,
    validate_preprocessing_array,
)


class MockHCPEDataSource(hcpe_transform.DataSource):
    """模擬HCPE データソース．テスト用に作成されたデータを提供する．"""

    def __init__(self, mock_data: np.ndarray) -> None:
        self.mock_data = mock_data

    def __len__(self) -> int:
        return len(self.mock_data)

    def iter_batches(
        self,
    ) -> Generator[tuple[str, np.ndarray], None, None]:
        yield "mock_batch", self.mock_data


class TempDataSource(hcpe_transform.DataSource):
    def __init__(
        self,
        file_paths: list[Path],
        array_type: Literal["hcpe", "preprocessing"],
        bit_pack: bool,
    ) -> None:
        self.data = {}
        for file_path in file_paths:
            if file_path.exists():
                if array_type == "hcpe":
                    self.data[file_path.name] = load_hcpe_array(
                        file_path=file_path, mmap_mode="r"
                    )
                elif array_type == "preprocessing":
                    self.data[file_path.name] = (
                        load_preprocessing_array(
                            file_path=file_path,
                            bit_pack=bit_pack,
                            mmap_mode="r",
                        )
                    )
                else:
                    raise ValueError(
                        f"Unsupported array type: {array_type}"
                    )
            else:
                raise FileNotFoundError(
                    f"File not found: {file_path}"
                )

    def __len__(self) -> int:
        return sum(len(arr) for arr in self.data.values())

    def iter_batches(
        self,
    ) -> Generator[tuple[str, np.ndarray], None, None]:
        for file_name, arr in self.data.items():
            yield file_name, arr


class TestHCPEConverter:
    @pytest.fixture(autouse=True)
    def clean_up_after_test(self) -> Generator[None, Any, Any]:
        output_dir = Path(
            "tests/maou/app/pre_process/resources/test_dir/output"
        )
        yield
        self.clean_up_dir(output_dir)

    def clean_up_dir(self, dir: Path) -> None:
        if dir.exists() and dir.is_dir():
            for f in dir.glob("**/*"):
                if f.name != ".gitkeep":
                    f.unlink()

    def test_successfull_transformation(self) -> None:
        input_paths = [
            Path(
                "tests/maou/app/pre_process/resources/test_dir/input/test_data_1.npy"
            ),
            Path(
                "tests/maou/app/pre_process/resources/test_dir/input/test_data_2.npy"
            ),
            Path(
                "tests/maou/app/pre_process/resources/test_dir/input/test_data_3.npy"
            ),
        ]
        output_dir = Path(
            "tests/maou/app/pre_process/resources/test_dir/output"
        )
        option: hcpe_transform.PreProcess.PreProcessOption = (
            hcpe_transform.PreProcess.PreProcessOption(
                output_dir=output_dir,
                output_filename="transformed",
                max_workers=1,
            )
        )
        self.clean_up_dir(output_dir)
        datasource = TempDataSource(
            file_paths=input_paths,
            array_type="hcpe",
            bit_pack=False,
        )
        transformer = hcpe_transform.PreProcess(
            datasource=datasource
        )
        transformer.transform(option)
        # 出力ファイルのチェック
        assert output_dir.exists()
        assert (
            (output_dir / "transformed")
            .with_suffix(".npy")
            .exists()
        )

    def test_preprocess_transform_comprehensive(self) -> None:
        """PreProcess.transformメソッドの包括的テスト．

        出力されるnumpy arrayの各種検証を行う．
        """
        # 1. テストデータ準備
        mock_hcpe_data = self._create_mock_hcpe_data()
        datasource = MockHCPEDataSource(mock_hcpe_data)

        # 2. PreProcessの設定と実行
        import tempfile
        from pathlib import Path

        from maou.domain.data.array_io import (
            load_preprocessing_array,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            preprocessor = hcpe_transform.PreProcess(
                datasource=datasource
            )
            option = hcpe_transform.PreProcess.PreProcessOption(
                output_dir=output_path,
                output_filename="test",
                max_workers=1,
            )
            preprocessor.transform(option)

            # 実際の前処理済みデータを取得
            preprocessed_array = load_preprocessing_array(
                output_path / "test.npy", bit_pack=True
            )

            # 3. 基本整合性検証
            self._test_basic_data_integrity(preprocessed_array)

            # 4. 確率分布検証
            self._test_probability_distributions(
                preprocessed_array
            )

            # 5. データ変換正確性検証（元データとの一致性）
            self._test_data_transformation_accuracy(
                preprocessed_array, mock_hcpe_data
            )

            # 6. 追加整合性検証
            self._test_additional_validations(
                preprocessed_array
            )

    def _create_mock_hcpe_data(self) -> np.ndarray:
        """テスト用の模擬HCPEデータを作成する．SFENから現実的な局面を作成．"""
        # テスト用の局面SFEN（既知の正しい局面）
        test_sfens = [
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",  # 初期局面
            "6k2/8P/9/9/5N3/9/9/9/4K4 b 2R2B4G4S3N4L16P 1",
            "6k2/8P/9/9/5N3/9/9/9/4K4 b 2R2B4G4S3N4L16P 1",
            "6k2/8P/9/9/5N3/9/9/9/4K4 b 2R2B4G4S3N4L16P 1",
            "6k2/8P/9/9/5N3/9/9/9/4K4 b 2R2B4G4S3N4L16P 1",
            "6k2/8P/9/9/5N3/9/9/9/4K4 b 2R2B4G4S3N4L16P 1",
            "6k2/8P/9/9/5N3/9/9/9/4K4 b 2R2B4G4S3N4L16P 1",
            "l5B1l/4+P2G1/2n2sn2/p1pr1pkpp/7P1/P1P1R3P/1PN+p1PN2/2G2s3/L+p1K4L b 1G4P1b1g2s 99",
            "l6nl/1r3bk2/p1np1gPp1/2p1p3p/3Psp1P1/1PP5P/P1BG1sN2/2S6/LNKG4L w 1G1S3P1r1p 62",
        ]

        positions = []
        game_results = [
            shogi.Result.BLACK_WIN,
            shogi.Result.WHITE_WIN,
            shogi.Result.DRAW,
        ]

        for sfen in test_sfens:
            board = shogi.Board()
            board.set_sfen(sfen)
            hcp_array = board.get_hcp()
            hcp = hcp_array[0][
                "hcp"
            ]  # 'hcp'フィールドにアクセス

            # この局面の実際の合法手を取得
            legal_moves = list(board.get_legal_moves())[
                :3
            ]  # 最初の3つだけ使用

            # 同じ局面で複数の手と結果を作成（集約テスト用）
            for i, move in enumerate(legal_moves):
                # move16に変換
                move16 = shogi.move16(move)
                result = game_results[i % len(game_results)]

                positions.append(
                    {
                        "hcp": hcp,
                        "bestMove16": move16,
                        "gameResult": result,
                    }
                )

        # HCPEデータ配列を作成
        mock_data = create_empty_hcpe_array(len(positions))

        for i, pos in enumerate(positions):
            mock_data["hcp"][i] = pos["hcp"]
            mock_data["bestMove16"][i] = np.int16(
                pos["bestMove16"]
            )
            mock_data["gameResult"][i] = pos["gameResult"]
            mock_data["eval"][i] = 100 + i * 50
            mock_data["id"][i] = f"test_id_{i}"

        return mock_data

    def _test_basic_data_integrity(
        self, preprocessed_array: np.ndarray
    ) -> None:
        """基本的なデータ整合性をテストする．"""
        # データ型検証
        assert (
            preprocessed_array.dtype
            == get_preprocessing_dtype()
        )

        # スキーマ検証
        validate_preprocessing_array(preprocessed_array)

        # ID一意性テスト
        ids = preprocessed_array["id"]
        assert len(np.unique(ids)) == len(ids), "IDが一意でない"

        # ID-features一対一関係テスト
        for i in range(len(preprocessed_array)):
            for j in range(i + 1, len(preprocessed_array)):
                if (
                    preprocessed_array["id"][i]
                    == preprocessed_array["id"][j]
                ):
                    assert np.array_equal(
                        preprocessed_array["features"][i],
                        preprocessed_array["features"][j],
                    ), "同じIDに対して異なるfeaturesが存在"

        # データが空でないことを確認
        assert len(preprocessed_array) > 0, (
            "前処理済みデータが空"
        )

    def _test_probability_distributions(
        self, preprocessed_array: np.ndarray
    ) -> None:
        """確率分布に関する検証を行う．"""
        for i, record in enumerate(preprocessed_array):
            # moveLabel総和が1であることを確認（小数点誤差を考慮）
            move_label_sum = np.sum(record["moveLabel"])
            assert np.isclose(
                move_label_sum, 1.0, rtol=1e-5, atol=1e-8
            ), (
                f"Record {i}: moveLabel総和が1でない ({move_label_sum})"
            )

            # resultValue範囲検証 (0-1)
            result_value = record["resultValue"]
            assert 0.0 <= result_value <= 1.0, (
                f"Record {i}: resultValueが範囲外 ({result_value})"
            )

            # moveLabelは非負値であることを確認
            assert np.all(record["moveLabel"] >= 0), (
                f"Record {i}: moveLabelに負の値が存在"
            )

    def _test_data_transformation_accuracy(
        self,
        preprocessed_array: np.ndarray,
        original_hcpe: np.ndarray,
    ) -> None:
        """データ変換の正確性を検証する（元データとの整合性）．"""
        # 各前処理済みレコードについて元データとの整合性を確認
        for processed_record in preprocessed_array:
            processed_id = processed_record["id"]

            # IDが有効な値であることを確認
            assert processed_id > 0, (
                f"IDが無効 ({processed_id})"
            )

            # 元データから同じハッシュのレコードを抽出
            matching_records = []
            for orig_record in original_hcpe:
                try:
                    orig_hash = Transform.board_hash(
                        orig_record["hcp"]
                    )
                    if orig_hash == processed_id:
                        matching_records.append(orig_record)
                except (ValueError, OverflowError, TypeError):
                    # ハッシュ計算に失敗した場合はスキップ
                    continue

            if matching_records:
                # 期待されるresultValueを計算（勝率）
                win_values = []
                for r in matching_records:
                    try:
                        win_value = Transform.board_game_result(
                            r["hcp"], r["gameResult"]
                        )
                        win_values.append(win_value)
                    except Exception:
                        # 変換に失敗した場合はスキップ
                        continue

                if win_values:
                    expected_result_value = np.mean(win_values)

                    # resultValueの検証（小数点誤差を考慮）
                    actual_result_value = float(
                        processed_record["resultValue"]
                    )
                    assert np.isclose(
                        actual_result_value,
                        expected_result_value,
                        rtol=1e-1,
                        atol=1e-2,
                    ), (
                        f"resultValue不一致: "
                        f"期待値={expected_result_value}, "
                        f"実際={actual_result_value}, "
                        f"ID={processed_id}, "
                        f"matching_records={len(matching_records)}"
                    )

                # 期待されるmoveLabelの分布を計算
                expected_move_counts = np.zeros(
                    len(processed_record["moveLabel"]),
                    dtype=np.int32,
                )
                for r in matching_records:
                    try:
                        move_label = Transform.board_move_label(
                            r["hcp"], r["bestMove16"]
                        )
                        expected_move_counts[move_label] += 1
                    except Exception:
                        # 変換に失敗した場合はスキップ
                        continue

                # 期待される確率分布を計算
                total_moves = np.sum(expected_move_counts)
                if total_moves > 0:
                    expected_move_probs = (
                        expected_move_counts / total_moves
                    )

                    # moveLabelの検証（小数点誤差を考慮）
                    actual_move_probs = processed_record[
                        "moveLabel"
                    ]
                    assert np.allclose(
                        actual_move_probs,
                        expected_move_probs,
                        rtol=1e-2,
                        atol=1e-3,
                    ), (
                        f"moveLabel不一致: "
                        f"ID={processed_id}, "
                        f"最大差分="
                        f"{np.max(np.abs(actual_move_probs - expected_move_probs))}"
                    )

    def _test_additional_validations(
        self, preprocessed_array: np.ndarray
    ) -> None:
        """追加的な検証項目をテストする．"""
        from maou.app.pre_process.label import MOVE_LABELS_NUM
        from maou.domain.board.shogi import FEATURES_NUM

        for i, record in enumerate(preprocessed_array):
            # features形状検証
            expected_features_shape = (FEATURES_NUM, 9, 9)
            actual_features_shape = record["features"].shape
            assert (
                actual_features_shape == expected_features_shape
            ), (
                f"Record {i}: features形状が不正 "
                f"(期待={expected_features_shape}, 実際={actual_features_shape})"
            )

            # moveLabel形状検証
            expected_move_label_shape = (MOVE_LABELS_NUM,)
            actual_move_label_shape = record["moveLabel"].shape
            assert (
                actual_move_label_shape
                == expected_move_label_shape
            ), (
                f"Record {i}: moveLabel形状が不正 "
                f"(期待={expected_move_label_shape}, 実際={actual_move_label_shape})"
            )

            # featuresが設定されていることを確認（全て0でないこと）
            features_sum = np.sum(record["features"])
            assert features_sum > 0, (
                f"Record {i}: featuresが全て0（設定されていない）"
            )
