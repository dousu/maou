"""cshogiをリファレンス実装としてテストフィクスチャを生成するスクリプト．

Usage:
    uv run python rust/maou_shogi/generate_fixtures.py

Output:
    rust/maou_shogi/tests/fixtures/*.json
"""

import json
import os
from pathlib import Path

import cshogi  # type: ignore

FIXTURES_DIR = Path(__file__).parent / "tests" / "fixtures"


def generate_sfen_fixtures() -> None:
    """各種局面のSFEN + pieces配列 + hand配列を生成する．"""
    test_cases = [
        {
            "name": "hirate",
            "sfen": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        },
        {
            "name": "mid_game",
            "sfen": "lnsg1gsnl/1r5b1/ppppkpppp/4p4/9/4P4/PPPP1PPPP/1B5R1/LNSGKGSNL b - 5",
        },
        {
            "name": "with_hand",
            "sfen": "lnsgkgsnl/1r5b1/pppppp1pp/6p2/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 3",
        },
        {
            "name": "promoted_pieces",
            "sfen": "lnsgk1snl/1r3g1b1/pppppp+Bpp/9/9/9/PPPPPPPPP/7R1/LNSGKGSNL w - 10",
        },
        {
            "name": "endgame",
            "sfen": "4k4/9/9/9/9/9/9/9/4K4 b 2r2b4g4s4n4l18p 1",
        },
    ]

    fixtures = []
    for tc in test_cases:
        board = cshogi.Board(tc["sfen"])
        pieces = list(board.pieces)
        hand_black = list(board.pieces_in_hand[0])
        hand_white = list(board.pieces_in_hand[1])

        fixtures.append(
            {
                "name": tc["name"],
                "sfen": tc["sfen"],
                "pieces": pieces,
                "hand_black": hand_black,
                "hand_white": hand_white,
                "turn": int(board.turn),
            }
        )

    save_fixture("sfen_fixtures.json", fixtures)


def generate_move_fixtures() -> None:
    """各種指し手のmove16/to/from/USI変換結果を生成する．"""
    board = cshogi.Board()
    fixtures = []

    for move in board.legal_moves:
        fixtures.append(
            {
                "move": int(move),
                "move16": int(cshogi.move16(move)),
                "to_sq": int(cshogi.move_to(move)),
                "from_sq": int(cshogi.move_from(move)),
                "usi": cshogi.move_to_usi(move),
                "is_drop": bool(cshogi.move_is_drop(move)),
                "is_promotion": bool(cshogi.move_is_promotion(move)),
            }
        )

    save_fixture("move_fixtures.json", fixtures)


def generate_legal_move_fixtures() -> None:
    """各局面の全合法手リスト(sorted)を生成する．"""
    test_positions = [
        {
            "name": "hirate",
            "sfen": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        },
        {
            "name": "mid_game",
            "sfen": "l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RsijFFn2teleportp 1",
        },
        {
            "name": "only_king",
            "sfen": "4k4/9/9/9/9/9/9/9/4K4 b 2r2b4g4s4n4l18p 1",
        },
        {
            "name": "in_check",
            "sfen": "4k4/9/9/9/9/9/9/4r4/4K4 b - 1",
        },
        {
            "name": "pawn_drop_positions",
            "sfen": "4k4/9/9/9/9/9/4P4/9/4K4 b P 1",
        },
        # 詰将棋風の局面(両玉あり，片方は攻め駒のみ)
        # 注: cshogiは片玉局面でバグがあるため(存在しない銀の打ちを生成)，
        # fixture testでは両玉局面を使用する．片玉はunit testで検証する．
        {
            "name": "tsume_like_attacker",
            "sfen": "4k4/9/4G4/9/9/9/9/9/4K4 b G 1",
        },
        {
            "name": "tsume_like_defender_turn",
            "sfen": "4k4/4G4/9/9/9/9/9/9/4K4 w - 1",
        },
        # 詰み局面 — 有名な詰みパターン(合法手0)
        # 頭金: 玉の頭に金を打つ最も基本的な詰み
        {
            "name": "checkmate_atama_kin",
            "sfen": "7k1/7G1/9/9/9/9/9/9/K6R1 w - 1",
        },
        # 腹金: 端の玉の横に金を打つ詰み
        {
            "name": "checkmate_hara_kin",
            "sfen": "9/7Gk/9/9/9/9/9/9/K6RL w - 1",
        },
        # 尻金: 龍(前)と金(後=尻)で玉を挟撃する詰み
        {
            "name": "checkmate_shiri_kin",
            "sfen": "R6+R1/7k1/7G1/9/9/9/9/9/K8 w - 1",
        },
        # 吊るし桂: 桂馬の遠隔王手で詰み(後手自駒が逃げ場を塞ぐ)
        {
            "name": "checkmate_tsurushi_kei",
            "sfen": "7gk/8p/6BN1/9/9/9/9/9/K8 w - 1",
        },
        # 都詰め: 盤の中央5五で王が詰む珍しい詰み
        {
            "name": "checkmate_miyako_zume",
            "sfen": "9/9/9/3+R1+R3/4k4/4G4/3B5/9/4K4 w - 1",
        },
        # 雪隠詰め: 隅に追い詰められた玉の詰み
        {
            "name": "checkmate_setchin_zume",
            "sfen": "8k/7S1/9/9/9/9/9/9/K6RL w - 1",
        },
    ]

    fixtures = []
    for tc in test_positions:
        try:
            board = cshogi.Board(tc["sfen"])
        except Exception:
            continue

        move16_list = sorted(int(cshogi.move16(m)) for m in board.legal_moves)
        usi_list = sorted(cshogi.move_to_usi(m) for m in board.legal_moves)

        fixtures.append(
            {
                "name": tc["name"],
                "sfen": tc["sfen"],
                "legal_moves_count": len(move16_list),
                "legal_moves_move16": move16_list,
                "legal_moves_usi": usi_list,
            }
        )

    save_fixture("legal_move_fixtures.json", fixtures)


def generate_special_rule_fixtures() -> None:
    """二歩・打ち歩詰め・行き所のない駒のテスト局面を生成する．"""
    fixtures = []

    # 二歩テスト: 5筋に先手の歩がある状態
    sfen_nifu = "4k4/9/9/9/4P4/9/9/9/4K4 b P 1"
    board = cshogi.Board(sfen_nifu)
    moves_usi = [cshogi.move_to_usi(m) for m in board.legal_moves]
    # 5筋(col=4)に歩を打てないことを確認
    pawn_drops_col4 = [
        m for m in moves_usi if m.startswith("P*5")
    ]
    fixtures.append(
        {
            "name": "nifu_check",
            "sfen": sfen_nifu,
            "description": "5筋に歩があるので5筋に歩を打てない",
            "pawn_drops_on_file5": pawn_drops_col4,
            "total_legal_moves": len(moves_usi),
        }
    )

    # 打ち歩詰めテスト
    sfen_uchifuzume = "4k4/4P4/9/9/9/9/9/9/4K4 b P 1"
    board = cshogi.Board(sfen_uchifuzume)
    moves_usi = [cshogi.move_to_usi(m) for m in board.legal_moves]
    # P*5aが合法手に含まれないことを確認
    has_p_drop_5a = "P*5a" in moves_usi
    fixtures.append(
        {
            "name": "uchifuzume_check",
            "sfen": sfen_uchifuzume,
            "description": "P*5aは打ち歩詰めなので合法手に含まれない",
            "p_drop_5a_is_legal": has_p_drop_5a,
            "total_legal_moves": len(moves_usi),
            "all_moves_usi": sorted(moves_usi),
        }
    )

    # 注: cshogiは片玉局面で存在しない駒の打ちを生成するバグがあるため，
    # fixture testでは片玉局面を使用しない．片玉はRust unit testで検証する．

    save_fixture("special_rule_fixtures.json", fixtures)


def generate_feature_fixtures() -> None:
    """piece_planes出力を生成する(平手のみ，サイズ制限)．"""
    board = cshogi.Board()

    import numpy as np

    planes = np.zeros((104, 9, 9), dtype=np.float32)
    board.piece_planes(planes)
    # cshogi順のまま保存(Python側での並べ替えはRustテストでは不要)

    planes_rotate = np.zeros((104, 9, 9), dtype=np.float32)
    board.piece_planes_rotate(planes_rotate)

    fixture = {
        "name": "hirate",
        "sfen": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        "piece_planes": planes.tolist(),
        "piece_planes_rotate": planes_rotate.tolist(),
    }

    save_fixture("feature_fixtures.json", [fixture])


def save_fixture(filename: str, data: list | dict) -> None:
    """フィクスチャをJSONファイルに保存する．"""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = FIXTURES_DIR / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {filepath}")


def main() -> None:
    print("Generating test fixtures from cshogi...")
    generate_sfen_fixtures()
    generate_move_fixtures()
    generate_legal_move_fixtures()
    generate_special_rule_fixtures()
    generate_feature_fixtures()
    print("Done!")


if __name__ == "__main__":
    main()
