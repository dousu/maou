#!/usr/bin/env python3
"""PVの各手が王手かどうかを確認するスクリプト."""

from maou._rust.maou_shogi import PyBoard

SFEN = "9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1"

PV_MOVES = [
    "7b6b", "5b4c", "8b9c", "4c3d", "1b2c", "3d2c",
    "N*1e", "2c3b", "N*2d", "3b2b", "2d1b+", "2b3b",
    "1b2b", "3b2b", "4f1c", "2b1c", "9c3c", "1c1d",
    "3c2c", "1d1e", "P*1f", "1e1f", "P*1g", "1f1g",
    "5g6f", "1g1h", "2c2g", "1h1i", "8g8i", "S*6i",
    "8i6i", "6h6i+", "S*2h", "1i2i", "2h3g", "2i3i",
    "2g2h", "3i4i", "2h4h",
]


def main() -> None:
    """PVの各手の情報を表示."""
    board = PyBoard()
    board.set_sfen(SFEN)

    print(f"{'手数':>4} {'USI':>8} {'手番':>4} {'SFEN (局面後)'}")
    print("-" * 100)

    for i, usi_move in enumerate(PV_MOVES):
        move_num = i + 1
        turn = "攻め" if i % 2 == 0 else "玉方"
        m = board.move_from_usi(usi_move)
        board.push(m)
        sfen = board.sfen()
        print(f"{move_num:>4} {usi_move:>8} {turn:>4} {sfen[:70]}")

    print()
    print("=" * 80)
    print("「残りN手」の局面(攻め方手番)で解けなかった理由を調べる")
    print("残り17手 = PV[22]以降, 残り19手 = PV[20]以降, etc.")
    print()

    # 解けなかった局面: 残り17, 19, 21, 23手
    # これらの局面で合法な王手がどれだけあるか確認
    from maou._rust.maou_shogi import solve_tsume

    for moves_played in [16, 18, 20, 22]:
        remaining = len(PV_MOVES) - moves_played
        board2 = PyBoard()
        board2.set_sfen(SFEN)
        for i in range(moves_played):
            m = board2.move_from_usi(PV_MOVES[i])
            board2.push(m)

        sfen = board2.sfen()
        next_move = PV_MOVES[moves_played]
        print(f"残り{remaining}手の局面: {sfen}")
        print(f"  正解PVの次の手: {next_move}")

        # この局面で短い詰みを探す (depth=3, 少ないノード)
        r1 = solve_tsume(sfen, depth=3, nodes=100000, timeout_secs=5)
        print(f"  3手以内の詰み探索: {r1.status} (nodes={r1.nodes_searched})")

        # もう少し深く
        r2 = solve_tsume(sfen, depth=remaining + 2, nodes=1_000_000, timeout_secs=10)
        print(f"  {remaining+2}手以内 (1Mノード): {r2.status} (nodes={r2.nodes_searched})")
        print()


if __name__ == "__main__":
    main()
