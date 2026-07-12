"""前処理 (_process_single_array) の golden 一致検証．

cshogi 遺構リファクタの parity gate．golden はリファクタ前実装の
出力を hash 昇順に正規化して固定したもの (npz)．
生成器: scratchpad/gen_refactor_golden.py (リファクタ前実装で実行)．

入力は converter golden (CSA 3 種: 先手勝ち/後手勝ち/引き分け) から
production 経路と同じ convert_hcpe_df_to_numpy で構築する．

NOTE: golden の resultValue 相当 (win_counts/move_win_values) は
gameResult 規約バグ (旧 shogi.Result 定義による誤読) の修正に伴い
修正後の値で再生成済み (hashes/labels/特徴量はリファクタ前実装と
bit-exact のまま)．規約の正しさ自体は
test_label.py::test_make_result_value_hcpe_raw_values と Rust 側
test_result_value_hcpe_convention が生値で固定している．
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


class TestRustBulkPreprocessParity:
    """Rust 一括 API (maou._rust.maou_search) と Python 実装の交差検証．

    前処理 hot path を Rust に置換する前の parity gate．
    Python 側実装が Rust 委譲になった後も，このテストは
    「per-position の意味論が golden 時点と同じ」ことの回帰検証として機能する．
    """

    def test_preprocess_hcpes_matches_python_loop(self) -> None:
        """Rust preprocess_hcpes が Python の per-position 計算と一致する．"""
        from maou._rust.maou_search import preprocess_hcpes
        from maou.app.pre_process.transform import Transform
        from maou.domain.board import shogi

        data = _build_input()
        n = len(data)

        hashes, labels, results = preprocess_hcpes(
            np.ascontiguousarray(data["hcp"]),
            np.ascontiguousarray(data["bestMove16"]),
            np.ascontiguousarray(data["gameResult"]).astype(
                np.int8
            ),
        )
        assert hashes.dtype == np.uint64
        assert labels.dtype == np.uint16
        assert results.dtype == np.float32

        board = shogi.Board()
        for i in range(n):
            board.set_hcp(data["hcp"][i])
            assert hashes[i] == board.hash(), (
                f"hash mismatch at {i}"
            )
            assert labels[
                i
            ] == Transform.board_move_label_from_board(
                board,
                int(data["bestMove16"][i]),
            ), f"move label mismatch at {i}"
            assert results[
                i
            ] == Transform.board_game_result_from_board(
                board,
                int(data["gameResult"][i]),
            ), f"result value mismatch at {i}"
