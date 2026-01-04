#!/usr/bin/env python3
"""実際の表示内容を確認"""

import numpy as np
from maou.domain.board.shogi import Board
from maou.domain.visualization.piece_mapping import get_piece_name_ja
from maou.app.pre_process.feature import make_board_id_positions

# 初期局面
board = Board()
board.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")

# board_id_positions取得
board_id_positions = make_board_id_positions(board)

print("初期局面のboard_id_positions:")
print(board_id_positions)
print()

print("重要な位置の駒名:")
print(f"[7, 1] (8筋2段) = ID {board_id_positions[7, 1]} → {get_piece_name_ja(board_id_positions[7, 1])} (期待: 角)")
print(f"[7, 7] (2筋2段) = ID {board_id_positions[7, 7]} → {get_piece_name_ja(board_id_positions[7, 7])} (期待: 飛)")
print()
print(f"[1, 1] (8筋8段) = ID {board_id_positions[1, 1]} → {get_piece_name_ja(board_id_positions[1, 1])} (期待: 飛)")
print(f"[1, 7] (2筋8段) = ID {board_id_positions[1, 7]} → {get_piece_name_ja(board_id_positions[1, 7])} (期待: 角)")
print()

print("結果:")
if get_piece_name_ja(board_id_positions[7, 1]) == "角" and get_piece_name_ja(board_id_positions[7, 7]) == "飛":
    print("✓ 先手の駒は正しい（8筋=角、2筋=飛）")
else:
    print("✗ 先手の駒が逆（8筋=飛、2筋=角）")

if get_piece_name_ja(board_id_positions[1, 1]) == "飛" and get_piece_name_ja(board_id_positions[1, 7]) == "角":
    print("✓ 後手の駒は正しい（8筋=飛、2筋=角）")
else:
    print("✗ 後手の駒が逆（8筋=角、2筋=飛）")
