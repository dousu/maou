//! maou_shogi submodule - PyO3 bindings for Shogi board operations

use numpy::{PyArray3, PyArrayMethods};
use pyo3::prelude::*;

use maou_shogi::board::Board;
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

#[pyclass]
struct PyBoard {
    board: Board,
    undo_stack: Vec<(u32, u8)>, // (move raw_u32, captured piece raw_u8)
}

#[pymethods]
impl PyBoard {
    #[new]
    fn new() -> Self {
        PyBoard {
            board: Board::new(),
            undo_stack: Vec::new(),
        }
    }

    fn set_sfen(&mut self, sfen: &str) -> PyResult<()> {
        self.board
            .set_sfen(sfen)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.undo_stack.clear();
        Ok(())
    }

    fn sfen(&self) -> String {
        self.board.sfen()
    }

    #[getter]
    fn turn(&self) -> u8 {
        self.board.turn() as u8
    }

    fn set_turn(&mut self, turn: u8) -> PyResult<()> {
        let color = Color::from_u8(turn)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("turn must be 0 or 1"))?;
        self.board.set_turn(color);
        Ok(())
    }

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

    fn to_hcp(&self) -> PyResult<Vec<u8>> {
        let hcp = hcp::to_hcp(&self.board)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(hcp.to_vec())
    }

    fn legal_moves(&mut self) -> Vec<u32> {
        movegen::generate_legal_moves(&mut self.board)
            .iter()
            .map(|m| m.raw_u32())
            .collect()
    }

    fn move_from_move16(&self, m16: u16) -> PyResult<u32> {
        self.board
            .move_from_move16(m16)
            .map(|m| m.raw_u32())
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("invalid move16"))
    }

    fn move_from_usi(&self, usi: &str) -> PyResult<u32> {
        self.board
            .move_from_usi(usi)
            .map(|m| m.raw_u32())
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("invalid USI move"))
    }

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

    fn pop(&mut self) -> PyResult<()> {
        let (m_raw, cap_raw) = self.undo_stack.pop().ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err("no moves to undo")
        })?;
        self.board
            .undo_move(Move::from_raw_u32(m_raw), Piece::from_raw_u8(cap_raw));
        Ok(())
    }

    fn piece_planes<'py>(&self, arr: &Bound<'py, PyArray3<f32>>) -> PyResult<()> {
        validate_feature_array_shape(arr)?;
        let mut rw = unsafe { arr.as_array_mut() };
        let slice = rw
            .as_slice_mut()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("array must be contiguous"))?;
        feature::piece_planes(&self.board, slice);
        Ok(())
    }

    fn piece_planes_rotate<'py>(&self, arr: &Bound<'py, PyArray3<f32>>) -> PyResult<()> {
        validate_feature_array_shape(arr)?;
        let mut rw = unsafe { arr.as_array_mut() };
        let slice = rw
            .as_slice_mut()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("array must be contiguous"))?;
        feature::piece_planes_rotate(&self.board, slice);
        Ok(())
    }

    fn pieces_in_hand(&self) -> (Vec<u32>, Vec<u32>) {
        let (black, white) = self.board.pieces_in_hand();
        (
            black.iter().map(|&x| x as u32).collect(),
            white.iter().map(|&x| x as u32).collect(),
        )
    }

    fn piece(&self, sq: u8) -> u32 {
        self.board.piece_at(Square::from_raw_u8(sq)) as u32
    }

    fn pieces(&self) -> Vec<u32> {
        self.board.pieces().iter().map(|&x| x as u32).collect()
    }

    fn zobrist_hash(&self) -> u64 {
        self.board.hash()
    }

    fn is_ok(&self) -> bool {
        self.board.is_ok()
    }

    fn __str__(&self) -> String {
        format!("{}", self.board)
    }
}

// Free functions (cshogi-compatible move utilities)

#[pyfunction]
fn move16(m: u32) -> u16 {
    moves::move16(Move::from_raw_u32(m))
}

#[pyfunction]
fn move_to(m: u32) -> u8 {
    moves::move_to(Move::from_raw_u32(m))
}

#[pyfunction]
fn move_from(m: u32) -> u8 {
    moves::move_from(Move::from_raw_u32(m))
}

#[pyfunction]
fn move_to_usi(m: u32) -> String {
    moves::move_to_usi(Move::from_raw_u32(m))
}

#[pyfunction]
fn move_is_drop(m: u32) -> bool {
    moves::move_is_drop(Move::from_raw_u32(m))
}

#[pyfunction]
fn move_is_promotion(m: u32) -> bool {
    moves::move_is_promotion(Move::from_raw_u32(m))
}

#[pyfunction]
fn move_drop_hand_piece(m: u32) -> u8 {
    moves::move_drop_hand_piece(Move::from_raw_u32(m))
}

/// Create maou_shogi submodule
pub fn create_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "maou_shogi")?;

    m.add_class::<PyBoard>()?;
    m.add_function(wrap_pyfunction!(move16, &m)?)?;
    m.add_function(wrap_pyfunction!(move_to, &m)?)?;
    m.add_function(wrap_pyfunction!(move_from, &m)?)?;
    m.add_function(wrap_pyfunction!(move_to_usi, &m)?)?;
    m.add_function(wrap_pyfunction!(move_is_drop, &m)?)?;
    m.add_function(wrap_pyfunction!(move_is_promotion, &m)?)?;
    m.add_function(wrap_pyfunction!(move_drop_hand_piece, &m)?)?;

    Ok(m)
}
