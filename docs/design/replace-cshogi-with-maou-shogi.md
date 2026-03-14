# Plan: cshogi → maou_shogi 置き換え

## 概要

Python側で使用している外部ライブラリ `cshogi` を，自前の Rust クレート `maou_shogi` の PyO3 バインディングに置き換える．
**CSA/KIF パーサーは対象外**とし，盤面操作・合法手生成・特徴量抽出・HCP エンコードのみを対象とする．

## スコープ

### 置き換え対象

| カテゴリ | cshogi API | maou_shogi 対応 |
|---------|-----------|----------------|
| 盤面初期化 | `cshogi.Board()` | `Board::new()` |
| SFEN | `board.set_sfen()`, `board.sfen()` | `Board::set_sfen()`, `Board::to_sfen()` |
| 合法手生成 | `board.legal_moves` | `movegen::legal_moves()` |
| 手の適用/取消 | `board.push()`, `board.pop()` | `Board::do_move()`, `Board::undo_move()` |
| 手の変換 | `board.move_from_move16()`, `board.move_from_usi()` | **未実装 → 追加必要** |
| 手のユーティリティ | `cshogi.move16()`, `move_to()`, `move_from()` 等 | `Move` のメソッド群 |
| 特徴量抽出 | `board.piece_planes()`, `piece_planes_rotate()` | `feature::piece_planes()`, `piece_planes_rotate()` |
| HCP | `board.to_hcp()`, `board.set_hcp()` | `hcp::to_hcp()`, `hcp::from_hcp()` |
| 駒情報 | `board.piece()`, `board.pieces`, `board.pieces_in_hand` | `Board::squares`, `Board::pieces_in_hand()` |
| ハッシュ | `board.zobrist_hash()` | `Board::hash` |
| 整合性検証 | `board.is_ok()` | `Board::is_ok()` |
| 定数 | `cshogi.BLACK`, `WHITE`, `MAX_PIECES_IN_HAND` 等 | `types.rs` の定数 |

### 対象外(パーサー)

- `src/maou/domain/parser/csa_parser.py` — `cshogi.CSA.Parser` を引き続き使用
- `src/maou/domain/parser/kif_parser.py` — `cshogi.KIF.Parser` を引き続き使用

## アーキテクチャ決定

### PyO3 サブモジュールとして追加

既存の `maou_rust` クレートに `maou_shogi` サブモジュールを追加する．

```
maou._rust (既存)
├── maou._rust.maou_io      (既存)
├── maou._rust.maou_index   (既存)
└── maou._rust.maou_shogi   (新規)
    ├── PyBoard (class)
    ├── move16()
    ├── move_to()
    ├── move_from()
    ├── move_to_usi()
    ├── move_is_drop()
    ├── move_is_promotion()
    └── move_drop_hand_piece()
```

**理由:**
- 既存パターン(`maou_io`, `maou_index`)と一致
- 単一 `cdylib` で maturin ビルドがシンプル
- maou_shogi は既にワークスペースメンバー

## 実装ステップ

### Step 1: maou_shogi に不足メソッドを追加

**ファイル:** `rust/maou_shogi/src/board.rs`

以下のメソッドを `impl Board` に追加:

1. **`move_from_move16(&self, move16: u16) -> Option<Move>`**
   - 16bit エンコードから from/to/promote を解析
   - 打ちの場合: `Move::new_drop(to, piece_type)` を構築
   - 通常手の場合: `self.squares[to]` で捕獲駒を取得し `Move` を構築

2. **`move_from_usi(&self, usi: &str) -> Option<Move>`**
   - USI 文字列をパースし，盤面状態から完全な Move を構築
   - `Move::from_usi()` が既にある場合はそれを活用

### Step 2: PyO3 ラッパーモジュール作成

**ファイル:** `rust/maou_rust/Cargo.toml` — 依存追加:

```toml
maou_shogi = { path = "../maou_shogi" }
numpy = "0.28"
```

**ファイル:** `rust/maou_rust/src/maou_shogi.rs` — 新規作成

#### PyBoard クラス

```rust
#[pyclass]
struct PyBoard {
    board: maou_shogi::Board,
    undo_stack: Vec<(maou_shogi::Move, maou_shogi::Piece)>,
}
```

**重要な設計判断 — push/pop のセマンティクス:**
- cshogi は内部ヒストリで `push()`/`pop()` を管理
- maou_shogi の `do_move`/`undo_move` は captured piece の受け渡しが必要
- PyO3 ラッパー側で `Vec<(Move, Piece)>` の undo スタックを保持

#### 公開メソッド一覧

| メソッド | 引数 | 戻り値 | 備考 |
|---------|------|--------|------|
| `__new__()` | - | PyBoard | 平手初期局面 |
| `set_sfen(sfen)` | `&str` | `PyResult<()>` | |
| `sfen()` | - | `String` | |
| `turn` (property) | - | `u8` | 0=Black, 1=White |
| `set_turn(turn)` | `u8` | - | |
| `set_hcp(data)` | `&[u8]` | `PyResult<()>` | 32バイト |
| `to_hcp()` | - | `PyResult<Vec<u8>>` | 32バイト返却 |
| `legal_moves()` | - | `Vec<u32>` | Move の raw 値 |
| `move_from_move16(m16)` | `u16` | `PyResult<u32>` | Step 1 で追加するメソッド使用 |
| `move_from_usi(usi)` | `&str` | `PyResult<u32>` | Step 1 で追加するメソッド使用 |
| `push(m)` | `u32` | - | do_move + undo_stack に push |
| `pop()` | - | - | undo_stack から pop + undo_move |
| `piece_planes(arr)` | `PyArray3<f32>` | - | in-place 書き込み |
| `piece_planes_rotate(arr)` | `PyArray3<f32>` | - | in-place 書き込み |
| `pieces_in_hand()` | - | `(Vec<u8>, Vec<u8>)` | (black, white) |
| `piece(sq)` | `u8` | `u8` | cshogi 互換 ID |
| `pieces()` | - | `Vec<u8>` | 81 要素 |
| `zobrist_hash()` | - | `u64` | |
| `is_ok()` | - | `bool` | |
| `__str__()` | - | `String` | Display trait 使用 |

#### フリー関数

| 関数 | 引数 | 戻り値 |
|------|------|--------|
| `move16(m)` | `u32` | `u16` |
| `move_to(m)` | `u32` | `u8` |
| `move_from(m)` | `u32` | `u8` |
| `move_to_usi(m)` | `u32` | `String` |
| `move_is_drop(m)` | `u32` | `bool` |
| `move_is_promotion(m)` | `u32` | `bool` |
| `move_drop_hand_piece(m)` | `u32` | `u8` |

### Step 3: lib.rs にサブモジュール登録

**ファイル:** `rust/maou_rust/src/lib.rs`

```rust
mod maou_shogi;

// _rust 関数内:
m.add_submodule(&maou_shogi::create_module(py)?)?;
sys_modules.set_item("maou._rust.maou_shogi", m.getattr("maou_shogi")?)?;
```

### Step 4: Python Board ラッパー修正

**ファイル:** `src/maou/domain/board/shogi.py`

#### 変更点

1. **インポート変更:**
   ```python
   # Before
   import cshogi

   # After
   from maou._rust.maou_shogi import (
       PyBoard,
       move16 as _move16,
       move_to as _move_to,
       move_from as _move_from,
       move_to_usi as _move_to_usi,
       move_is_drop as _move_is_drop,
       move_is_promotion as _move_is_promotion,
       move_drop_hand_piece as _move_drop_hand_piece,
   )
   ```

2. **定数:** `cshogi.BLACK` → リテラル `0`，`cshogi.WHITE` → リテラル `1`(既に `Turn` enum で定義済み)

3. **Board クラス:** `self.board = cshogi.Board()` → `self.board = PyBoard()`

4. **HCP dtype:** `cshogi.HuffmanCodedPos` numpy dtype の使用箇所を `self.board.to_hcp()` の `bytes` 返却に変更

5. **特徴量抽出:** `_reorder_piece_planes_cshogi_to_pieceid()` はそのまま維持(maou_shogi も cshogi 互換の駒順序を使用)

6. **cshogi インポートを残す箇所:** パーサーが `cshogi` をインポートするため，`pyproject.toml` の依存からは削除しない

### Step 5: Rust ビルド & テスト

```bash
# Rust クレートのビルド
uv run maturin develop

# Rust テスト
cargo test --manifest-path rust/maou_shogi/Cargo.toml

# Python テスト
uv run pytest tests/
```

### Step 6: バージョンバンプ

`src/` 配下を変更するため `pyproject.toml` のバージョンを `feat:` (minor) でバンプ．

## 互換性リスクと対策

| リスク | 影響度 | 対策 |
|-------|-------|------|
| Move エンコーディング不一致 | 高 | maou_shogi は cshogi 互換ビットレイアウト(検証済み) |
| Piece ID 不一致 | 高 | cshogi 互換 ID 使用(0=空, 1-14=先手, 17-30=後手) |
| 特徴量チャネル順序 | 高 | cshogi 互換順序 + 既存 reorder ロジック維持 |
| HCP バイナリ互換性 | 高 | Apery 互換実装．既存 fixture テストで検証 |
| push/pop セマンティクス | 中 | PyO3 ラッパーで undo スタック実装 |
| numpy interop | 中 | `numpy` クレート v0.28 で PyArray3 対応 |

## 影響ファイル一覧

### 変更するファイル

| ファイル | 変更内容 |
|---------|---------|
| `rust/maou_shogi/src/board.rs` | `move_from_move16`, `move_from_usi` 追加 |
| `rust/maou_rust/Cargo.toml` | `maou_shogi`, `numpy` 依存追加 |
| `rust/maou_rust/src/lib.rs` | maou_shogi サブモジュール登録 |
| `rust/maou_rust/src/maou_shogi.rs` | 新規: PyO3 ラッパーモジュール |
| `src/maou/domain/board/shogi.py` | cshogi → maou_shogi 置き換え |
| `pyproject.toml` | バージョンバンプ |

### 変更しないファイル(パーサー)

| ファイル | 理由 |
|---------|------|
| `src/maou/domain/parser/csa_parser.py` | cshogi.CSA.Parser を継続使用 |
| `src/maou/domain/parser/kif_parser.py` | cshogi.KIF.Parser を継続使用 |

### テストファイル(修正が必要な可能性)

| ファイル | 理由 |
|---------|------|
| `tests/maou/domain/board/test_shogi_polars.py` | Board API 経由のため変更不要の見込み |
| `tests/maou/domain/move/test_label.py` | cshogi 直接参照があれば修正 |
| `tests/maou/app/pre_process/test_feature.py` | cshogi 直接参照があれば修正 |
| `tests/maou/test_cross_stage_consistency.py` | cshogi 直接参照があれば修正 |
