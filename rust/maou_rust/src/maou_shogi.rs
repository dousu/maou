//! maou_shogi submodule - PyO3 bindings for Shogi board operations

use numpy::{PyArray3, PyArrayMethods};
use pyo3::prelude::*;

use maou_shogi::board::Board;
use maou_shogi::dfpn;
use maou_shogi::feature;
use maou_shogi::hcp;
use maou_shogi::movegen;
use maou_shogi::moves::{self, Move};
use maou_shogi::types::{Color, Piece, Square, FEATURES_NUM};

/// PyArray3<f32> の feature planes の形状を検証する．
fn validate_feature_array_shape(arr: &Bound<'_, PyArray3<f32>>) -> PyResult<()> {
    let shape = arr.dims();
    let expected = [FEATURES_NUM, 9, 9];
    if shape != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected array shape {expected:?}, got {shape:?}"
        )));
    }
    Ok(())
}

/// 将棋盤面の Python バインディング．
///
/// SFEN / HCP による局面設定，合法手生成，特徴平面抽出などを提供する．
#[pyclass]
struct PyBoard {
    board: Board,
    undo_stack: Vec<(u32, u8)>, // (move raw_u32, captured piece raw_u8)
}

#[pymethods]
impl PyBoard {
    /// 初期局面 (平手) で盤面を生成する．
    #[new]
    fn new() -> Self {
        PyBoard {
            board: Board::new(),
            undo_stack: Vec::new(),
        }
    }

    /// SFEN 文字列から局面を設定する．
    fn set_sfen(&mut self, sfen: &str) -> PyResult<()> {
        self.board
            .set_sfen(sfen)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.undo_stack.clear();
        Ok(())
    }

    /// 現在の局面を SFEN 文字列で返す．
    fn sfen(&self) -> String {
        self.board.sfen()
    }

    /// 現在の手番 (0=先手, 1=後手)．
    #[getter]
    fn turn(&self) -> u8 {
        self.board.turn() as u8
    }

    /// 手番を設定する (0=先手, 1=後手)．
    fn set_turn(&mut self, turn: u8) -> PyResult<()> {
        let color = Color::from_u8(turn)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("turn must be 0 or 1"))?;
        self.board.set_turn(color);
        Ok(())
    }

    /// 32 バイトの HCP データから局面を設定する．
    fn set_hcp(&mut self, data: &[u8]) -> PyResult<()> {
        if data.len() < 32 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "HCP data must be at least 32 bytes",
            ));
        }
        let mut hcp_buf = [0u8; 32];
        hcp_buf.copy_from_slice(&data[..32]);
        self.board = hcp::from_hcp(&hcp_buf)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.undo_stack.clear();
        Ok(())
    }

    /// 局面を 32 バイトの HCP データにエンコードする．
    fn to_hcp(&self) -> PyResult<Vec<u8>> {
        let hcp = hcp::to_hcp(&self.board)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(hcp.to_vec())
    }

    /// 合法手のリストを返す (各要素は 32-bit move)．
    fn legal_moves(&mut self) -> Vec<u32> {
        movegen::generate_legal_moves(&mut self.board)
            .iter()
            .map(|m| m.raw_u32())
            .collect()
    }

    /// 16-bit move から 32-bit move に変換する．
    fn move_from_move16(&self, m16: u16) -> PyResult<u32> {
        self.board
            .move_from_move16(m16)
            .map(|m| m.raw_u32())
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("invalid move16"))
    }

    /// USI 形式の文字列から 32-bit move に変換する．
    fn move_from_usi(&self, usi: &str) -> PyResult<u32> {
        self.board
            .move_from_usi(usi)
            .map(|m| m.raw_u32())
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("invalid USI move"))
    }

    /// 指し手を実行して局面を進める．
    fn push(&mut self, m: u32) -> PyResult<()> {
        if m == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "invalid move: Move::NONE (0)",
            ));
        }
        let mv = Move::from_raw_u32(m);
        let captured = self.board.do_move(mv);
        self.undo_stack.push((m, captured.raw_u8()));
        Ok(())
    }

    /// 直前の指し手を取り消す．
    fn pop(&mut self) -> PyResult<()> {
        let (m_raw, cap_raw) = self
            .undo_stack
            .pop()
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("no moves to undo"))?;
        self.board
            .undo_move(Move::from_raw_u32(m_raw), Piece::from_raw_u8(cap_raw));
        Ok(())
    }

    /// 先手視点の駒特徴平面 (104x9x9) を書き込む．
    fn piece_planes<'py>(&self, arr: &Bound<'py, PyArray3<f32>>) -> PyResult<()> {
        validate_feature_array_shape(arr)?;
        // SAFETY: The caller must ensure no other Python reference aliases `arr`
        // during this call. `as_array_mut()` bypasses Python-level aliasing checks;
        // if another ndarray view shares the same buffer, this is undefined behavior.
        // In practice, `to_piece_planes()` in shogi.py creates a new local array
        // that is not shared, satisfying this invariant.
        let mut rw = unsafe { arr.as_array_mut() };
        let slice = rw
            .as_slice_mut()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("array must be contiguous"))?;
        feature::piece_planes(&self.board, slice);
        Ok(())
    }

    /// 後手視点 (180度回転) の駒特徴平面 (104x9x9) を書き込む．
    fn piece_planes_rotate<'py>(&self, arr: &Bound<'py, PyArray3<f32>>) -> PyResult<()> {
        validate_feature_array_shape(arr)?;
        // SAFETY: Same invariant as `piece_planes` — caller must ensure
        // exclusive access to `arr`. See comment in `piece_planes` for details.
        let mut rw = unsafe { arr.as_array_mut() };
        let slice = rw
            .as_slice_mut()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("array must be contiguous"))?;
        feature::piece_planes_rotate(&self.board, slice);
        Ok(())
    }

    /// 持ち駒を (先手, 後手) のタプルで返す．
    ///
    /// 各リストは [歩, 香, 桂, 銀, 金, 角, 飛] の順．
    fn pieces_in_hand(&self) -> (Vec<u32>, Vec<u32>) {
        let (black, white) = self.board.pieces_in_hand();
        (
            black.iter().map(|&x| x as u32).collect(),
            white.iter().map(|&x| x as u32).collect(),
        )
    }

    /// 指定マスの駒 ID を返す (0=空)．
    fn piece(&self, sq: u8) -> PyResult<u32> {
        if sq >= 81 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "square index out of range: {sq} (must be 0..80)"
            )));
        }
        Ok(self.board.piece_at(Square::from_raw_u8(sq)) as u32)
    }

    /// 盤面の駒配列 (81 要素) を返す．
    fn pieces(&self) -> Vec<u32> {
        self.board.pieces().iter().map(|&x| x as u32).collect()
    }

    /// Zobrist ハッシュ値を返す．
    fn zobrist_hash(&self) -> u64 {
        self.board.hash()
    }

    /// 盤面が妥当かどうかを検証する．
    fn is_ok(&self) -> bool {
        self.board.is_ok()
    }

    /// 人間可読な盤面表現を返す．
    fn __str__(&self) -> String {
        format!("{}", self.board)
    }
}

// Free functions (cshogi-compatible move utilities)

/// 32-bit move を 16-bit に圧縮する．
#[pyfunction]
fn move16(m: u32) -> u16 {
    moves::move16(Move::from_raw_u32(m))
}

/// 指し手から移動先マス番号 (0-80) を取得する．
#[pyfunction]
fn move_to(m: u32) -> u8 {
    moves::move_to(Move::from_raw_u32(m))
}

/// 指し手から移動元情報を取得する．
///
/// 通常手: マス番号 (0-80)，駒打ち: 内部エンコード値．
#[pyfunction]
fn move_from(m: u32) -> u8 {
    moves::move_from(Move::from_raw_u32(m))
}

/// 指し手を USI 形式の文字列に変換する．
#[pyfunction]
fn move_to_usi(m: u32) -> String {
    moves::move_to_usi(Move::from_raw_u32(m))
}

/// 指し手が駒打ちかどうかを判定する．
#[pyfunction]
fn move_is_drop(m: u32) -> bool {
    moves::move_is_drop(Move::from_raw_u32(m))
}

/// 指し手が成りを含むかどうかを判定する．
#[pyfunction]
fn move_is_promotion(m: u32) -> bool {
    moves::move_is_promotion(Move::from_raw_u32(m))
}

/// 駒打ちの駒種を取得する．
#[pyfunction]
fn move_drop_hand_piece(m: u32) -> u8 {
    moves::move_drop_hand_piece(Move::from_raw_u32(m))
}

/// 詰将棋の探索結果．
///
/// - `status`: `"checkmate"` / `"checkmate_no_pv"` / `"no_checkmate"` / `"unknown"` のいずれか．
/// - `moves`: 詰みの場合は手順(USI形式のリスト)．それ以外は空リスト．
/// - `nodes_searched`: 探索ノード数．
#[pyclass(frozen)]
struct TsumeResult {
    #[pyo3(get)]
    status: String,
    #[pyo3(get)]
    moves: Vec<String>,
    #[pyo3(get)]
    nodes_searched: u64,
}

#[pymethods]
impl TsumeResult {
    fn __repr__(&self) -> String {
        if self.status == "checkmate" {
            format!(
                "TsumeResult(status='checkmate', moves={:?}, nodes_searched={})",
                self.moves, self.nodes_searched
            )
        } else {
            format!(
                "TsumeResult(status='{}', nodes_searched={})",
                self.status, self.nodes_searched
            )
        }
    }

    /// `status == "checkmate"` のとき `True`．
    ///
    /// `checkmate_no_pv` は手順 (`moves`) が空のため `False` を返す．
    /// 手順の有無に関わらず詰みが証明されたかを判定するには
    /// `is_proven` プロパティを使用すること．
    fn __bool__(&self) -> bool {
        self.status == "checkmate"
    }

    /// 詰みが証明されたかどうかを返す．
    ///
    /// `status` が `"checkmate"` または `"checkmate_no_pv"` のとき `True`．
    /// `__bool__` と異なり，PV 復元に失敗した場合でも `True` を返す．
    ///
    /// ```python
    /// result = solve_tsume(sfen)
    /// if result.is_proven:
    ///     print("詰みが証明された")
    /// ```
    #[getter]
    fn is_proven(&self) -> bool {
        self.status == "checkmate" || self.status == "checkmate_no_pv"
    }
}

/// 詰将棋を解く(Df-Pn アルゴリズム)．
///
/// 返り値: `TsumeResult` オブジェクト．
///   - `status`: `"checkmate"` / `"checkmate_no_pv"` / `"no_checkmate"` / `"unknown"`
///   - `moves`: 詰みの場合は手順(USI形式のリスト)
///   - `nodes_searched`: 探索ノード数
///
/// # 引数
///
/// - `sfen` (str): 局面のSFEN文字列．
/// - `depth` (int, optional): 最大探索手数(デフォルト 31)．範囲: 1〜数百程度．
/// - `nodes` (int, optional): 最大ノード数(デフォルト 1,048,576 = 2^20)．
///   u64 範囲(0〜2^64-1)．推奨: 100,000〜100,000,000．
/// - `draw_ply` (int, optional): 引き分け手数(デフォルト 32767)．
/// - `timeout_secs` (int, optional): 実行時間制限(秒)(デフォルト 300)．
/// - `find_shortest` (bool, optional): 最短手数探索を行うか(デフォルト true)．
///   false にすると追加探索をスキップし高速化するが，
///   返される手順が最短とは限らない．
#[pyfunction]
#[pyo3(signature = (sfen, depth=31, nodes=1048576, draw_ply=32767, timeout_secs=300, find_shortest=true))]
fn solve_tsume(
    sfen: &str,
    depth: u32,
    nodes: u64,
    draw_ply: u32,
    timeout_secs: u64,
    find_shortest: bool,
) -> PyResult<TsumeResult> {
    let result = dfpn::solve_tsume_with_timeout(
        sfen,
        Some(depth),
        Some(nodes),
        Some(draw_ply),
        Some(timeout_secs),
        Some(find_shortest),
    )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    match result {
        dfpn::TsumeResult::Checkmate { moves, nodes_searched } => {
            Ok(TsumeResult {
                status: "checkmate".to_string(),
                moves: moves.iter().map(|m| m.to_usi()).collect(),
                nodes_searched,
            })
        }
        dfpn::TsumeResult::CheckmateNoPv { nodes_searched } => {
            Ok(TsumeResult {
                status: "checkmate_no_pv".to_string(),
                moves: Vec::new(),
                nodes_searched,
            })
        }
        dfpn::TsumeResult::NoCheckmate { nodes_searched } => {
            Ok(TsumeResult {
                status: "no_checkmate".to_string(),
                moves: Vec::new(),
                nodes_searched,
            })
        }
        dfpn::TsumeResult::Unknown { nodes_searched } => {
            Ok(TsumeResult {
                status: "unknown".to_string(),
                moves: Vec::new(),
                nodes_searched,
            })
        }
    }
}

/// Create maou_shogi submodule
pub fn create_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "maou_shogi")?;

    m.add_class::<PyBoard>()?;
    m.add_class::<TsumeResult>()?;
    m.add_function(wrap_pyfunction!(move16, &m)?)?;
    m.add_function(wrap_pyfunction!(move_to, &m)?)?;
    m.add_function(wrap_pyfunction!(move_from, &m)?)?;
    m.add_function(wrap_pyfunction!(move_to_usi, &m)?)?;
    m.add_function(wrap_pyfunction!(move_is_drop, &m)?)?;
    m.add_function(wrap_pyfunction!(move_is_promotion, &m)?)?;
    m.add_function(wrap_pyfunction!(move_drop_hand_piece, &m)?)?;
    m.add_function(wrap_pyfunction!(solve_tsume, &m)?)?;

    Ok(m)
}
