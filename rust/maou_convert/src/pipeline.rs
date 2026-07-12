//! 変換パイプライン: パース → フィルタ → 局面再生 → RecordBatch → feather 書き出し．
//!
//! Python `hcpe_converter.py` の `_process_single_file` / `game_filter` /
//! `convert` を移植．CSA は複数対局を全て変換し (Python は先頭 1 局のみだった)，
//! game 0 は従来 id (`{prefix}_{ply}`)，game≥1 は `{prefix}_g{g}_{ply}` を用いる．

use std::path::{Path, PathBuf};

use maou_shogi::kifu::{game_to_hcpe_rows, parse_csa_multi, parse_kif_str, GameRecord, HcpeError};
use rayon::prelude::*;

use crate::batch::build_record_batch;
use crate::date::{csa_start_date, kif_start_date, DateParseError};
use crate::decode::{decode_kifu_bytes, DecodeError};

/// 入力棋譜フォーマット．
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    Csa,
    Kif,
}

impl InputFormat {
    /// Python CLI の `--input-format` 文字列からパースする．
    pub fn parse(s: &str) -> Option<InputFormat> {
        match s {
            "csa" => Some(InputFormat::Csa),
            "kif" => Some(InputFormat::Kif),
            _ => None,
        }
    }
}

/// 品質フィルタ + 除外手のオプション．
#[derive(Debug, Clone, Default)]
pub struct ConvertOptions {
    pub min_rating: Option<i32>,
    pub min_moves: Option<usize>,
    pub max_moves: Option<usize>,
    pub allowed_endgame_status: Option<Vec<String>>,
    pub exclude_moves: Vec<u32>,
}

/// HCPE 1 行 (batch 組み立て用の中間表現)．
#[derive(Debug, Clone)]
pub struct HcpeRecord {
    pub hcp: [u8; 32],
    pub eval: i16,
    pub best_move16: i16,
    pub game_result: i8,
    pub id: String,
    pub partitioning_key: Option<i32>,
    pub ratings: Vec<u16>,
    pub endgame_status: Option<String>,
    pub moves: i16,
}

/// 1 ファイル分の変換結果 (status 文字列導出用)．
#[derive(Debug)]
pub enum ConvertOutcome {
    /// 変換成功 (1 行以上)．
    Success(Vec<HcpeRecord>),
    /// フィルタで全局スキップ．
    Skipped,
    /// 指し手が無い．
    SkippedNoMoves,
}

/// 変換パイプラインのエラー (ファイル単位の "error: ..." status に対応)．
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("undefined format {0}")]
    UndefinedFormat(String),
    #[error("decode error: {0}")]
    Decode(#[from] DecodeError),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("HCPE conversion error: {0}")]
    Hcpe(#[from] HcpeError),
    #[error("date parse error: {0}")]
    Date(#[from] DateParseError),
    #[error("io error: {0}")]
    Io(String),
    /// KIF に min_rating を指定した (レーティング情報が無い)．
    /// Python は `min([])` が ValueError を送出していた挙動を再現する．
    #[error("min() arg is an empty sequence")]
    EmptyRatings,
}

/// フォーマット別のレーティング列 (フィルタ + ratings カラム両用)．
///
/// CSA は `[先手, 後手]` (2 要素)．KIF はレーティング情報が無く空リスト
/// (Python `KifParser.ratings()` が常に `[]` を返していた挙動を再現)．
fn ratings_for(format: InputFormat, record: &GameRecord) -> Vec<u16> {
    match format {
        InputFormat::Csa => record.ratings.iter().map(|&r| r as u16).collect(),
        InputFormat::Kif => Vec::new(),
    }
}

/// Python `game_filter` 相当．変換対象なら `Ok(true)`．
fn passes_filter(
    format: InputFormat,
    record: &GameRecord,
    ratings: &[u16],
    opts: &ConvertOptions,
) -> Result<bool, PipelineError> {
    let move_count = record.moves.len();

    if let Some(min_rating) = opts.min_rating {
        // Python: min(parser.ratings())．KIF は空 → min([]) が ValueError
        let min = ratings
            .iter()
            .copied()
            .min()
            .ok_or(PipelineError::EmptyRatings)?;
        let _ = format; // ratings は format 依存で既に構築済み
        if (min as i32) < min_rating {
            return Ok(false);
        }
    }
    if let Some(min_moves) = opts.min_moves {
        if move_count < min_moves {
            return Ok(false);
        }
    }
    if let Some(max_moves) = opts.max_moves {
        if move_count > max_moves {
            return Ok(false);
        }
    }
    if let Some(allowed) = &opts.allowed_endgame_status {
        if !allowed.is_empty() {
            // Python: parser.endgame() で比較．CSA は None→""，KIF は None のまま．
            let endgame = endgame_status(format, record);
            let cur = endgame.as_deref().unwrap_or("");
            // KIF で endgame=None のときは "" が allowed に含まれなければ弾く
            let contained = match &endgame {
                Some(s) => allowed.iter().any(|a| a == s),
                None => allowed.iter().any(|a| a == cur), // None は "" 扱いにはしない
            };
            if !contained {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// endgameStatus カラムの値 (CSA は None→Some("")，KIF は None のまま)．
fn endgame_status(format: InputFormat, record: &GameRecord) -> Option<String> {
    match format {
        InputFormat::Csa => Some(record.endgame.clone().unwrap_or_default()),
        InputFormat::Kif => record.endgame.clone(),
    }
}

/// partitioningKey (Date32 日数) を求める．
fn partitioning_key(
    format: InputFormat,
    record: &GameRecord,
) -> Result<Option<i32>, PipelineError> {
    let key = match format {
        InputFormat::Csa => "START_TIME",
        InputFormat::Kif => "開始日時",
    };
    let value = record
        .var_info
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str());
    match (format, value) {
        (_, None) => Ok(None),
        (InputFormat::Csa, Some(v)) => Ok(Some(csa_start_date(v)?)),
        (InputFormat::Kif, Some(v)) => Ok(kif_start_date(v)),
    }
}

/// 1 局を HCPE 行に変換する (フィルタ通過済み前提)．
fn game_to_records(
    format: InputFormat,
    record: &GameRecord,
    id_prefix: &str,
    game_index: usize,
    ratings: Vec<u16>,
    opts: &ConvertOptions,
) -> Result<Vec<HcpeRecord>, PipelineError> {
    let rows = game_to_hcpe_rows(record, &opts.exclude_moves)?;
    let game_result = record.win.unwrap_or(0) as i8;
    let pkey = partitioning_key(format, record)?;
    let endgame = endgame_status(format, record);
    let moves_count = record.moves.len() as i16;

    let out = rows
        .into_iter()
        .map(|row| {
            let id = if game_index == 0 {
                format!("{id_prefix}_{}", row.ply)
            } else {
                format!("{id_prefix}_g{game_index}_{}", row.ply)
            };
            HcpeRecord {
                hcp: row.hcp,
                eval: row.eval,
                best_move16: row.best_move16 as i16,
                game_result,
                id,
                partitioning_key: pkey,
                ratings: ratings.clone(),
                endgame_status: endgame.clone(),
                moves: moves_count,
            }
        })
        .collect();
    Ok(out)
}

/// 棋譜文字列を HCPE レコード列に変換する．
///
/// `id_prefix` は id の接頭辞 (`{id_prefix}_{ply}`)．CSA は全対局を変換し，
/// game≥1 は `{id_prefix}_g{g}_{ply}`．
pub fn convert_content(
    content: &str,
    format: InputFormat,
    id_prefix: &str,
    opts: &ConvertOptions,
) -> Result<ConvertOutcome, PipelineError> {
    let games: Vec<GameRecord> = match format {
        InputFormat::Csa => {
            parse_csa_multi(content).map_err(|e| PipelineError::Parse(e.to_string()))?
        }
        InputFormat::Kif => {
            vec![parse_kif_str(content).map_err(|e| PipelineError::Parse(e.to_string()))?]
        }
    };

    let mut all_rows: Vec<HcpeRecord> = Vec::new();
    let mut saw_no_moves = false;
    let mut saw_filtered = false;

    for (g, record) in games.iter().enumerate() {
        let ratings = ratings_for(format, record);
        if !passes_filter(format, record, &ratings, opts)? {
            saw_filtered = true;
            continue;
        }
        if record.moves.is_empty() {
            saw_no_moves = true;
            continue;
        }
        let rows = game_to_records(format, record, id_prefix, g, ratings, opts)?;
        all_rows.extend(rows);
    }

    if !all_rows.is_empty() {
        Ok(ConvertOutcome::Success(all_rows))
    } else if saw_no_moves && !saw_filtered {
        Ok(ConvertOutcome::SkippedNoMoves)
    } else {
        Ok(ConvertOutcome::Skipped)
    }
}

/// ファイルの stem (拡張子なしの名前)．
fn file_stem(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// 1 ファイルを変換し (path, status) を返す．出力 feather は `output_dir` に書く．
fn convert_one_file(
    path: &Path,
    format: InputFormat,
    output_dir: &Path,
    opts: &ConvertOptions,
) -> (String, String) {
    let path_str = path.to_string_lossy().into_owned();
    let status = convert_one_file_inner(path, format, output_dir, opts);
    let status_str = match status {
        Ok(ConvertOutcome::Success(rows)) => format!("success {} rows", rows.len()),
        Ok(ConvertOutcome::Skipped) => "skipped".to_string(),
        Ok(ConvertOutcome::SkippedNoMoves) => "skipped (no moves)".to_string(),
        Err(e) => format!("error: {e}"),
    };
    (path_str, status_str)
}

fn convert_one_file_inner(
    path: &Path,
    format: InputFormat,
    output_dir: &Path,
    opts: &ConvertOptions,
) -> Result<ConvertOutcome, PipelineError> {
    let bytes = std::fs::read(path).map_err(|e| PipelineError::Io(e.to_string()))?;
    let content = decode_kifu_bytes(&bytes)?;
    let stem = file_stem(path);
    let id_prefix = format!("{stem}.hcpe");
    let outcome = convert_content(&content, format, &id_prefix, opts)?;

    if let ConvertOutcome::Success(rows) = &outcome {
        let batch = build_record_batch(rows).map_err(|e| PipelineError::Io(e.to_string()))?;
        let out_path = output_dir.join(format!("{stem}.feather"));
        maou_io::arrow_io::save_feather_batches(&[batch], &out_path.to_string_lossy())
            .map_err(|e| PipelineError::Io(e.to_string()))?;
    }
    Ok(outcome)
}

/// 複数ファイルを rayon で並列変換する．
///
/// `threads` が `Some(n)` なら専用 ThreadPool (n スレッド) で実行し，`None` は
/// rayon のグローバルプールを使う．戻り値は入力順の (path, status) リスト．
pub fn convert_files(
    paths: &[PathBuf],
    format: InputFormat,
    output_dir: &Path,
    opts: &ConvertOptions,
    threads: Option<usize>,
) -> Vec<(String, String)> {
    let run = || {
        paths
            .par_iter()
            .map(|p| convert_one_file(p, format, output_dir, opts))
            .collect::<Vec<_>>()
    };
    match threads {
        Some(n) if n >= 1 => rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map(|pool| pool.install(run))
            .unwrap_or_else(|_| run()),
        _ => run(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MULTI_CSA: &str = "V2.2\nN+g1black\nN-g1white\nPI\n+\n+7776FU\nT1\n%TORYO\n/\nV2.2\nN+g2black\nN-g2white\nPI\n+\n+2726FU\nT2\n-3334FU\n%CHUDAN\n";

    #[test]
    fn test_multi_game_all_converted() {
        let outcome = convert_content(
            MULTI_CSA,
            InputFormat::Csa,
            "m.hcpe",
            &ConvertOptions::default(),
        )
        .expect("ok");
        match outcome {
            ConvertOutcome::Success(rows) => {
                // game0: 1 手, game1: 2 手 = 3 行
                assert_eq!(rows.len(), 3);
                assert_eq!(rows[0].id, "m.hcpe_0");
                assert_eq!(rows[1].id, "m.hcpe_g1_0");
                assert_eq!(rows[2].id, "m.hcpe_g1_1");
                // game0 は先手勝ち (%TORYO), game1 は中断 (%CHUDAN → draw)
                assert_eq!(rows[0].game_result, 1);
                assert_eq!(rows[1].game_result, 0);
            }
            other => panic!("expected Success, got {other:?}"),
        }
    }

    #[test]
    fn test_min_rating_on_kif_errors() {
        let kif = "手合割：平手\n開始日時：2025/01/05\n手数----指手---------消費時間--\n   1 ７六歩(77)   ( 0:00/00:00:00)\n";
        let opts = ConvertOptions {
            min_rating: Some(2000),
            ..Default::default()
        };
        let err = convert_content(kif, InputFormat::Kif, "k.hcpe", &opts);
        assert!(matches!(err, Err(PipelineError::EmptyRatings)));
    }

    #[test]
    fn test_no_moves_skipped() {
        let csa = "V2.2\nPI\n+\n%TORYO\n";
        let outcome = convert_content(csa, InputFormat::Csa, "n.hcpe", &ConvertOptions::default())
            .expect("ok");
        assert!(matches!(outcome, ConvertOutcome::SkippedNoMoves));
    }

    #[test]
    fn test_filtered_skipped() {
        let csa =
            "V2.2\n'black_rate:test:1000.0\n'white_rate:test:1000.0\nPI\n+\n+7776FU\n%TORYO\n";
        let opts = ConvertOptions {
            min_rating: Some(3000),
            ..Default::default()
        };
        let outcome = convert_content(csa, InputFormat::Csa, "f.hcpe", &opts).expect("ok");
        assert!(matches!(outcome, ConvertOutcome::Skipped));
    }
}
