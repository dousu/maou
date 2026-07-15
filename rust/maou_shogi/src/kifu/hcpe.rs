//! GameRecord → HCPE 行列変換 (局面再生)．
//!
//! Python 実装 `src/maou/app/converter/hcpe_converter.py` の per-move ループ
//! (`board.set_sfen` → 各手で `to_hcp` / `move16` / `push_move`) を Rust に
//! 移植したもの．初期局面から指し手を 1 手ずつ再生しながら，各局面 (指し手
//! 適用前) の HCP と手番側視点の評価値，`bestMove16` を列で生成する．
//!
//! 対局不変の情報 (勝敗・レーティング・終局状態・手数・partitioningKey) は
//! ここでは扱わず，呼び出し側 (`maou_convert`) が GameRecord のフィールドと
//! var_info から付与する．移植の正しさは HCPE 変換 golden fixture
//! (`tests/maou/app/converter/resources/golden/`) との bit-exact 一致で検証．

use crate::board::Board;
use crate::hcp::{self, Hcp};
use crate::kifu::record::GameRecord;
use crate::moves::{self, Move};
use crate::types::Color;

/// HCPE 変換エラー．
#[derive(Debug, thiserror::Error)]
pub enum HcpeError {
    /// 初期局面 SFEN のパース失敗．
    #[error("invalid initial SFEN: {0}")]
    Sfen(#[from] crate::board::SfenError),
    /// HCP エンコード失敗 (ply 番目の局面)．
    #[error("HCP encode failed at ply {ply}: {source}")]
    Hcp {
        ply: usize,
        #[source]
        source: crate::hcp::HcpError,
    },
}

/// HCPE 1 行 (1 局面) の可変部分．
///
/// 対局不変の列 (gameResult / ratings / endgameStatus / moves /
/// partitioningKey) は呼び出し側で付与する．
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HcpeRow {
    /// 局面 (指し手適用**前**) の HCP．
    pub hcp: Hcp,
    /// 手番側視点の評価値 (後手番は符号反転)．
    pub eval: i16,
    /// この局面で指された手の 16-bit 表現．
    pub best_move16: u16,
    /// id 用の連番 (`{stem}.hcpe_{ply}`)．exclude された手の分は欠番になる．
    pub ply: u16,
}

/// GameRecord を初期局面から再生し，各局面の HCPE 行を生成する．
///
/// `exclude_moves` に含まれる指し手はスキップする (Python の
/// `exclude_moves` 挙動を bit-exact 移植: 該当手は行を出力せず盤面も進めず，
/// `ply` 連番のみ消費して id を欠番にする)．
pub fn game_to_hcpe_rows(
    record: &GameRecord,
    exclude_moves: &[u32],
) -> Result<Vec<HcpeRow>, HcpeError> {
    let mut board = Board::empty();
    board.set_sfen(&record.sfen)?;

    let mut rows = Vec::with_capacity(record.moves.len());
    for (idx, &m) in record.moves.iter().enumerate() {
        // Python の `if exclude_moves and move in exclude_moves: continue`
        // 相当．continue は行出力も push もスキップする (idx は enumerate で進む)．
        if exclude_moves.contains(&m) {
            continue;
        }

        // HCP は指し手適用前の局面
        let hcp = hcp::to_hcp(&board).map_err(|e| HcpeError::Hcp {
            ply: idx,
            source: e,
        })?;

        // 評価値: [-32767, 32767] にクランプ後，後手番は符号反転
        let score = record.scores.get(idx).copied().unwrap_or(0);
        let clamped = score.clamp(-32767, 32767) as i16;
        let eval = match board.turn() {
            Color::Black => clamped,
            Color::White => -clamped,
        };

        let mv = Move::from_raw_u32(m);
        rows.push(HcpeRow {
            hcp,
            eval,
            best_move16: moves::move16(mv),
            ply: idx as u16,
        });

        // Board::do_move は bits 0-14 のみ読むため cshogi 互換 move を
        // そのまま適用できる (record.rs のエンコーディング doc 参照)．
        board.do_move(mv);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kifu::parse_csa_str;

    #[test]
    fn test_hirate_single_move() {
        // 平手初期局面から +7776FU の 1 手．
        let mut record = GameRecord {
            sfen: "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1".to_string(),
            ..Default::default()
        };
        // +7776FU の cshogi 互換 move (record.rs テストの 0x00011e3b)
        record.moves = vec![0x00011e3b];
        record.scores = vec![123];

        let rows = game_to_hcpe_rows(&record, &[]).expect("変換成功");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].ply, 0);
        // 先手番なので eval は符号そのまま
        assert_eq!(rows[0].eval, 123);
        // move16 = move & 0xFFFF
        assert_eq!(rows[0].best_move16, (0x00011e3b_u32 & 0xFFFF) as u16);
    }

    #[test]
    fn test_white_turn_eval_sign_flip() {
        // 1 手進めた局面 (後手番) で score の符号が反転する
        let record = parse_csa_single("V2.2\nPI\n+\n+7776FU\n'** 100\n-3334FU\n'** 200\n%TORYO\n");
        let rows = game_to_hcpe_rows(&record, &[]).expect("変換成功");
        assert_eq!(rows.len(), 2);
        // ply 0: 先手番 score=100 → 100
        assert_eq!(rows[0].eval, 100);
        // ply 1: 後手番 score=200 → -200
        assert_eq!(rows[1].eval, -200);
    }

    #[test]
    fn test_exclude_moves_creates_id_gap() {
        let record = parse_csa_single("V2.2\nPI\n+\n+7776FU\n-3334FU\n+2726FU\n%TORYO\n");
        let all = game_to_hcpe_rows(&record, &[]).expect("変換成功");
        assert_eq!(all.len(), 3);
        let excluded_move = record.moves[1];
        let rows = game_to_hcpe_rows(&record, &[excluded_move]).expect("変換成功");
        // 2 手目を除外 → 行は 2 つ，ply は 0 と 2 (1 が欠番)
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].ply, 0);
        assert_eq!(rows[1].ply, 2);
    }

    fn parse_csa_single(content: &str) -> GameRecord {
        parse_csa_str(content).expect("CSA parse")
    }
}
