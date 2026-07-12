"""stage2 データ生成のループ本体の golden 一致検証．

cshogi 遺構リファクタの parity gate．golden はリファクタ前実装で
unique 局面ごとの (boardIdPositions, piecesInHand, legalMovesLabel) を
固定したもの (npz)．
生成器: scratchpad/gen_refactor_golden.py (リファクタ前実装で実行)．

NOTE: このテストは stage2_data_generation._generate_labels の
チャンク内ループ本体と同一のコードパスを再現している (ループ本体が
メソッド内に埋め込まれており直接呼べないため)．前処理 hot path を
Rust 一括 API に置換するコミットで，本テストの計算部を新 API 呼び出しに
書き換えて golden 比較を維持すること．
"""

from pathlib import Path

import numpy as np

from maou.app.pre_process.feature import (
    make_board_id_positions,
    make_pieces_in_hand,
)
from maou.app.utility.stage2_data_generation import (
    Stage2DataGenerationUseCase,
)
from maou.domain.board import shogi
from maou.domain.move.label import (
    MOVE_LABELS_NUM,
    make_move_label,
)

GOLDEN_NPZ = Path(
    "tests/maou/app/utility/resources/golden/stage2_golden.npz"
)


class TestStage2Golden:
    def test_label_generation_matches_golden(self) -> None:
        """unique 局面ごとの特徴量と合法手ラベルが golden と一致する．"""
        golden = np.load(GOLDEN_NPZ)
        hcps = golden["hcps"]
        n = len(hcps)

        board = shogi.Board()
        for i in range(n):
            board.set_hcp(hcps[i])
            board_positions = make_board_id_positions(board)
            pieces_in_hand = make_pieces_in_hand(board)
            legal = np.zeros(MOVE_LABELS_NUM, dtype=np.uint8)
            if board.get_turn() == shogi.Turn.BLACK:
                for move in board.get_legal_moves():
                    legal[
                        make_move_label(shogi.Turn.BLACK, move)
                    ] = 1
            else:
                normalized_board = Stage2DataGenerationUseCase._reconstruct_normalized_board(
                    board_positions, pieces_in_hand
                )
                for move in normalized_board.get_legal_moves():
                    legal[
                        make_move_label(shogi.Turn.BLACK, move)
                    ] = 1

            np.testing.assert_array_equal(
                board_positions,
                golden["board_id_positions"][i],
                err_msg=f"boardIdPositions mismatch at index {i}",
            )
            np.testing.assert_array_equal(
                pieces_in_hand,
                golden["pieces_in_hand"][i],
                err_msg=f"piecesInHand mismatch at index {i}",
            )
            np.testing.assert_array_equal(
                legal,
                golden["legal_moves_label"][i],
                err_msg=f"legalMovesLabel mismatch at index {i}",
            )


class TestRustBulkStage2Golden:
    """Rust 一括 API (encode_hcp_features / legal_move_masks) の golden 検証．

    golden はリファクタ前の Python 実装 (make_board_id_positions /
    make_pieces_in_hand / 後手番の盤面再構築 + make_move_label) の出力．
    Rust 側は後手番の合法手を「元盤面の合法手を後手視点でラベル化」で
    計算しており，このテストが盤面再構築方式との同値性を実データで検証する．
    """

    def test_bulk_features_and_masks_match_golden(self) -> None:
        from maou._rust.maou_search import (
            encode_hcp_features,
            legal_move_masks,
        )

        golden = np.load(GOLDEN_NPZ)
        hcps = np.ascontiguousarray(golden["hcps"])

        boards, hands = encode_hcp_features(hcps)
        np.testing.assert_array_equal(
            boards, golden["board_id_positions"]
        )
        np.testing.assert_array_equal(
            hands, golden["pieces_in_hand"]
        )

        masks = legal_move_masks(hcps)
        np.testing.assert_array_equal(
            masks, golden["legal_moves_label"]
        )
