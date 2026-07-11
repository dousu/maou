//! 千日手 (同一局面の再出現) 検出 — 経路ハッシュの後方走査．
//!
//! # maou_shogi 既存実装との対応 (語彙・意味論を揃えている)
//!
//! - **連続王手の分類**は `Position::is_perpetual_check_move`
//!   (maou_shogi/src/position.rs) と同じ「王手フラグの手番 parity 別
//!   区間全称判定」．同メソッドは指し手側の連続王手のみを判定するが，
//!   探索では検出がどちらの手番の局面で起きるかが経路次第なので，
//!   本モジュールは両側 ([`RepetitionOutcome::Loss`] /
//!   [`RepetitionOutcome::Win`]) + 通常千日手 (Draw) に一般化する．
//! - **検出のタイミング**は dfpn (maou_shogi/src/dfpn/) の on-path 検出
//!   (`path_depths`) と同じ「経路上の最初の再出現で終端」．実ルールの
//!   「同一局面 4 回」(position.rs は `hash_count` で 4-fold を数える) は
//!   待たない — 探索内の標準近似で dfpn も同じ扱い．王手が続く限り
//!   後続の循環も全て王手なので，初回循環での分類は 4-fold まで指し
//!   続けた場合の結論と一致する．
//! - **ハッシュ**は両実装と同じフル Zobrist (盤 + 持駒 + 手番,
//!   `Board::hash`)．
//!
//! dfpn 側で連続王手の分類が不要なのは OR ノード (攻め方) の着手生成が
//! 王手限定のため — 経路上の再出現が構造的に「連続王手の千日手 = 攻め方の
//! 失敗 (不詰)」になる．MCTS は任意の手を探索するため本モジュールの分類が
//! 必要になる．
//!
//! # 探索側の使い方
//!
//! MCTS の木には合流 (transposition) が無く root への経路はノード毎に
//! 一意なので，「対局履歴 + root からの経路」に対する千日手判定の結果は
//! ノードに対して不変になる．探索側はこの性質を使い，初回検出時にノードを
//! 終端状態 ([`crate::tree::node_state`]) へ焼き付けて以後の走査を省く．
//!
//! 未実装 (将来のレバー): 優越局面による一般化 — 盤面同一で持駒が優越/劣位
//! の局面の刈り込み (dfpn の `DomPathStack` / `hand_gte` 相当)．

use maou_shogi::board::Board;

/// 対局履歴・探索経路上の 1 局面 (千日手検出用)．
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoryEntry {
    /// 局面のフル Zobrist ハッシュ (盤 + 持ち駒 + 手番)．
    pub hash: u64,
    /// この局面で手番側が王手を受けているか (= この局面に至った指し手が
    /// 王手だったか)．position.rs の `StateInfo::gives_check` と同じ
    /// ビットを「指し手」でなく「局面」側に付けたもの．
    pub in_check: bool,
}

impl HistoryEntry {
    /// 局面からエントリを作る (position.rs の `do_move` と同じ
    /// `is_in_check(turn)` イディオム)．
    pub fn from_board(board: &Board) -> HistoryEntry {
        HistoryEntry {
            hash: board.hash(),
            in_check: board.is_in_check(board.turn()),
        }
    }
}

/// 千日手判定の結果 (現局面の手番側から見た値)．
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepetitionOutcome {
    /// 通常の千日手 — 引き分け (0.5)．
    Draw,
    /// 手番側が王手をかけ続けて同一局面に戻した — 手番側の負け (0.0)．
    Loss,
    /// 相手が王手をかけ続けて同一局面に戻した — 手番側の勝ち (1.0)．
    Win,
}

/// 対局履歴 + 探索経路の末尾局面が千日手かを判定する．
///
/// `game_history` は root より前の対局履歴 (古い順，root を含まない)，
/// `path` は root から現局面まで (先頭が root，末尾が現局面)．
///
/// フルハッシュは手番を含むため同一局面は必ず偶数距離に現れる．また最短の
/// 繰り返しは 4 手 (双方が動かして戻す) なので，距離 4 から 2 手刻みで後方
/// 走査し，最初に一致した局面との区間を
/// `Position::is_perpetual_check_move` と同じ parity 全称判定で分類する:
///
/// - 相手手番の局面が全て王手 → 手番側が王手をかけ続けた
///   ([`RepetitionOutcome::Loss`])
/// - 手番側の局面が全て王手 → 相手が王手をかけ続けた
///   ([`RepetitionOutcome::Win`])
/// - どちらでもない → [`RepetitionOutcome::Draw`]
///
/// 双方が王手をかけ続けた場合 (逆王手の応酬，実戦ではほぼ生じない) は
/// Loss を優先する．
pub fn find_repetition(
    game_history: &[HistoryEntry],
    path: &[HistoryEntry],
) -> Option<RepetitionOutcome> {
    let gh = game_history.len();
    let total = gh + path.len();
    let get = |i: usize| {
        if i < gh {
            game_history[i]
        } else {
            path[i - gh]
        }
    };
    let k = total - 1;
    let cur_hash = get(k).hash;
    let mut dist = 4;
    while dist <= k {
        let m = k - dist;
        if get(m).hash == cur_hash {
            // 区間 (m, k] の王手状況で分類する．m からの距離 j が偶数の局面は
            // 現局面と同じ手番 (王手を受けていれば相手が王手をかけた)，
            // 奇数の局面は相手番 (王手を受けていれば手番側が王手をかけた)
            let mut self_all_checked = true;
            let mut opp_all_checked = true;
            for j in 1..=dist {
                let checked = get(m + j).in_check;
                if j % 2 == 0 {
                    self_all_checked &= checked;
                } else {
                    opp_all_checked &= checked;
                }
            }
            return Some(if opp_all_checked {
                RepetitionOutcome::Loss
            } else if self_all_checked {
                RepetitionOutcome::Win
            } else {
                RepetitionOutcome::Draw
            });
        }
        dist += 2;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use maou_shogi::movegen::generate_legal_moves;

    fn e(hash: u64, in_check: bool) -> HistoryEntry {
        HistoryEntry { hash, in_check }
    }

    /// 後手玉 1a，先手飛 2c．先手が 1c/2c への飛車往復で王手をかけ続けられる．
    const PERPETUAL: &str = "8k/9/7R1/9/9/9/9/9/4K4 b - 1";

    /// SFEN から指し手列 (USI) を合法手照合しながら適用し，
    /// 各局面のエントリ列 (開始局面を含む) を返す．
    fn replay(sfen: &str, moves: &[&str]) -> Vec<HistoryEntry> {
        let mut board = Board::empty();
        board.set_sfen(sfen).expect("正当な SFEN");
        let mut entries = vec![HistoryEntry::from_board(&board)];
        for usi in moves {
            let mut probe = board.clone();
            let mv = generate_legal_moves(&mut probe)
                .into_iter()
                .find(|m| m.to_usi() == *usi)
                .unwrap_or_else(|| panic!("{usi} は合法手のはず"));
            board.do_move(mv);
            entries.push(HistoryEntry::from_board(&board));
        }
        entries
    }

    #[test]
    fn test_no_repetition() {
        let path = [
            e(1, false),
            e(2, false),
            e(3, false),
            e(4, false),
            e(5, false),
        ];
        assert_eq!(find_repetition(&[], &path), None);
    }

    #[test]
    fn test_simple_draw() {
        let path = [
            e(1, false),
            e(2, false),
            e(3, false),
            e(4, false),
            e(1, false),
        ];
        assert_eq!(find_repetition(&[], &path), Some(RepetitionOutcome::Draw));
    }

    #[test]
    fn test_distance_2_not_matched() {
        // 距離 2 の同一局面は実局面では生じ得ない (照合は距離 4 から)
        let path = [e(1, false), e(2, false), e(1, false)];
        assert_eq!(find_repetition(&[], &path), None);
    }

    #[test]
    fn test_perpetual_check_by_current_side_is_loss() {
        // 相手手番の局面 (奇数距離) が全て王手 = 手番側が王手をかけ続けた
        let path = [
            e(1, false),
            e(2, true),
            e(3, false),
            e(4, true),
            e(1, false),
        ];
        assert_eq!(find_repetition(&[], &path), Some(RepetitionOutcome::Loss));
    }

    #[test]
    fn test_perpetual_check_by_opponent_is_win() {
        // 手番側の局面 (偶数距離) が全て王手 = 相手が王手をかけ続けた
        // (同一局面は王手状態も同一なので両端の in_check を揃える)
        let path = [e(1, true), e(2, false), e(3, true), e(4, false), e(1, true)];
        assert_eq!(find_repetition(&[], &path), Some(RepetitionOutcome::Win));
    }

    #[test]
    fn test_both_perpetual_prefers_loss() {
        let path = [e(1, true), e(2, true), e(3, true), e(4, true), e(1, true)];
        assert_eq!(find_repetition(&[], &path), Some(RepetitionOutcome::Loss));
    }

    #[test]
    fn test_partial_checks_is_draw() {
        // 王手が途切れる往復は通常の千日手
        let path = [
            e(1, false),
            e(2, true),
            e(3, false),
            e(4, false),
            e(1, false),
        ];
        assert_eq!(find_repetition(&[], &path), Some(RepetitionOutcome::Draw));
    }

    #[test]
    fn test_nearest_occurrence_is_used() {
        // 距離 4 と距離 8 の両方に一致がある場合は近い方の区間で分類する
        let path = [
            e(1, false),
            e(9, true),
            e(3, false),
            e(9, true),
            e(1, false),
            e(6, false),
            e(7, false),
            e(8, false),
            e(1, false),
        ];
        assert_eq!(find_repetition(&[], &path), Some(RepetitionOutcome::Draw));
    }

    #[test]
    fn test_match_in_game_history() {
        // 一致相手が対局履歴側にあっても検出できる (結合列 [1,2,3,4,1])
        let gh = [e(1, false), e(2, false)];
        let path = [e(3, false), e(4, false), e(1, false)];
        assert_eq!(find_repetition(&gh, &path), Some(RepetitionOutcome::Draw));
    }

    #[test]
    fn test_real_board_perpetual_check_loss() {
        // 先手が 4 手で同一局面に戻し，その間先手の指し手が全て王手
        let entries = replay(PERPETUAL, &["2c1c", "1a2a", "1c2c", "2a1a"]);
        assert_eq!(entries[4].hash, entries[0].hash, "4 手で同一局面に戻る");
        assert_eq!(
            find_repetition(&[], &entries),
            Some(RepetitionOutcome::Loss)
        );
    }

    #[test]
    fn test_real_board_perpetual_check_win() {
        // 同じ循環をもう 1 手進める: 現局面は王手を受けている後手番で，
        // 王手をかけ続けているのは相手 (先手) → 手番側 (後手) から見て勝ち
        let entries = replay(PERPETUAL, &["2c1c", "1a2a", "1c2c", "2a1a", "2c1c"]);
        assert_eq!(entries[5].hash, entries[1].hash);
        assert_eq!(find_repetition(&[], &entries), Some(RepetitionOutcome::Win));
    }

    #[test]
    fn test_real_board_kings_shuffle_draw() {
        let entries = replay(
            "k8/9/9/9/9/9/9/9/8K b - 1",
            &["1i1h", "9a9b", "1h1i", "9b9a"],
        );
        assert_eq!(entries[4].hash, entries[0].hash);
        assert_eq!(
            find_repetition(&[], &entries),
            Some(RepetitionOutcome::Draw)
        );
    }
}
