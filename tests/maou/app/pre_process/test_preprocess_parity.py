"""前処理 (_process_single_array) の golden 一致検証．

cshogi 遺構リファクタの parity gate．golden はリファクタ前実装の
出力を hash 昇順に正規化して固定したもの (npz)．
生成器: scratchpad/gen_refactor_golden.py (リファクタ前実装で実行)．

入力は converter golden (CSA 3 種: 先手勝ち/後手勝ち/引き分け) から
production 経路と同じ convert_hcpe_df_to_numpy で構築する．

NOTE: golden の resultValue 相当 (win_counts/move_win_values) は
gameResult 規約バグ (cshogi 規約 0=draw/1=black/2=white を
shogi.Result で解釈) を含む現行挙動を固定している．
バグ修正コミットで期待値を明示的に更新する．
"""

from pathlib import Path

import numpy as np
import polars as pl

from maou.app.pre_process.hcpe_transform import PreProcess
from maou.domain.data.schema import convert_hcpe_df_to_numpy

GOLDEN_CONV = Path("tests/maou/app/converter/resources/golden")
GOLDEN_NPZ = Path(
    "tests/maou/app/pre_process/resources/golden/preprocess_golden.npz"
)


def _build_input() -> np.ndarray:
    dfs = [
        pl.read_ipc(GOLDEN_CONV / "csa_test_data_1.feather"),
        pl.read_ipc(GOLDEN_CONV / "csa_test_data_2.feather"),
        pl.read_ipc(GOLDEN_CONV / "csa_test_data_3.feather"),
    ]
    arrays = [convert_hcpe_df_to_numpy(df) for df in dfs]
    return np.concatenate(arrays)


class TestPreprocessParity:
    def test_process_single_array_matches_golden(self) -> None:
        """_process_single_array の全出力が golden と bit-exact に一致する．"""
        data = _build_input()
        result = PreProcess._process_single_array(data)
        golden = np.load(GOLDEN_NPZ)

        hashes = np.array(
            sorted(result.keys()), dtype=np.uint64
        )
        np.testing.assert_array_equal(hashes, golden["hashes"])

        offsets = golden["offsets"]
        for i, h in enumerate(hashes):
            d = result[int(h)]
            lo, hi = offsets[i], offsets[i + 1]
            assert d["count"] == golden["counts"][i], (
                f"count mismatch at hash {h}"
            )
            # win 値は {0, 0.5, 1} の和で f32 で正確に表現できるため
            # 完全一致で比較する
            assert (
                np.float32(d["winCount"])
                == golden["win_counts"][i]
            ), f"winCount mismatch at hash {h}"
            np.testing.assert_array_equal(
                d["moveLabelIndices"],
                golden["move_label_indices"][lo:hi],
                err_msg=f"moveLabelIndices mismatch at hash {h}",
            )
            np.testing.assert_array_equal(
                d["moveLabelValues"],
                golden["move_label_values"][lo:hi],
                err_msg=f"moveLabelValues mismatch at hash {h}",
            )
            np.testing.assert_array_equal(
                np.asarray(
                    d["moveWinValues"], dtype=np.float32
                ),
                golden["move_win_values"][lo:hi],
                err_msg=f"moveWinValues mismatch at hash {h}",
            )
            np.testing.assert_array_equal(
                d["boardIdPositions"],
                golden["board_id_positions"][i],
                err_msg=f"boardIdPositions mismatch at hash {h}",
            )
            np.testing.assert_array_equal(
                d["piecesInHand"],
                golden["pieces_in_hand"][i],
                err_msg=f"piecesInHand mismatch at hash {h}",
            )
