# maou_shogi クレート設計資料

## 概要

`maou_shogi`は，maouプロジェクトで使用しているcshogiの機能をRustで自前実装したクレートである．
cshogiはC++実装のPythonバインディングであり，ビルド環境の制約やカスタマイズの難しさがある．
cshogiの依存部分をRustで再実装し，既存Rustワークスペースに統合することで，
ビルドの一貫性と将来の拡張性を確保する．

## スコープ

### 対象

- Board(局面表現): Mailbox + Bitboardハイブリッド
- Move(指し手): 16-bit/32-bitエンコーディング，USI変換
- 合法手生成: 疑似合法手生成 + 合法性フィルタリング
- Position(履歴付き局面): 連続王手の千日手判定
- piece_planes(特徴平面): 104×9×9のNN入力特徴量生成
- SFEN: パース/シリアライズ
- Zobrist hashing: 局面のハッシュ値計算

### 対象外

- KIF/CSAパーサー
- HuffmanCodedPos (HCP): SFENベースのシリアライズで代替
- GUI/通信プロトコル

## 座標系

cshogi互換の座標系を使用する:

- `square = col * 9 + row` (column-major)
- col: 0-8 (0=1筋, 8=9筋)
- row: 0-8 (0=一段目, 8=九段目)
- 先手の敵陣: row 0-2 (1段目〜3段目)
- 後手の敵陣: row 6-8 (7段目〜9段目)

**注意**: SFEN文字列は左から右に9筋→1筋の順で記述されるため，
パース時に列の反転(`actual_col = 8 - string_position`)が必要．

## 駒ID体系

cshogiと同じID体系を使用する:

| 駒 | 先手(Black) | 後手(White) |
|----|------------|------------|
| 空 | 0 | - |
| 歩 | 1 | 17 |
| 香 | 2 | 18 |
| 桂 | 3 | 19 |
| 銀 | 4 | 20 |
| 角 | 5 | 21 |
| 飛 | 6 | 22 |
| 金 | 7 | 23 |
| 王 | 8 | 24 |
| と | 9 | 25 |
| 成香 | 10 | 26 |
| 成桂 | 11 | 27 |
| 成銀 | 12 | 28 |
| 馬 | 13 | 29 |
| 龍 | 14 | 30 |

白駒オフセット: +16

maouプロジェクトでは別途domain PieceIdが存在する(金・角・飛の順序が異なる)．
`piece.rs`の`CSHOGI_TO_DOMAIN`テーブルで変換する．

## Moveエンコーディング

### 16-bit (HCPE互換)

```
Bit 15:   drop flag (1=駒打ち)
Bit 14:   promotion flag (1=成り)
Bits 13-7: from_sq (0-80) or drop piece type (駒打ちの場合)
Bits 6-0:  to_sq (0-80)
```

### 32-bit (内部拡張)

```
Bits 0-15:  16-bit move (上記)
Bits 16-20: captured piece (取った駒のcshogi ID, 0=なし)
Bits 21-24: moving piece type (動かした駒種)
```

## データ構造

### Board

```rust
pub struct Board {
    pub squares: [Piece; 81],           // Mailbox
    pub piece_bb: [[Bitboard; 15]; 2],  // [color][piece_type]
    pub occupied: [Bitboard; 2],        // [color]
    pub hand: [[u8; 7]; 2],            // [color][hand_piece_index]
    pub turn: Color,
    pub ply: u16,
    pub hash: u64,                      // Zobrist hash
}
```

### Bitboard

```rust
pub struct Bitboard {
    pub lo: u64,  // squares 0-62 (63ビット)
    pub hi: u64,  // squares 63-80 (18ビット)
}
```

81マスを2つのu64で表現する．BitAnd, BitOr, BitXor, Not等の演算を実装．

### Position

```rust
pub struct Position {
    pub board: Board,
    history: Vec<StateInfo>,
}

struct StateInfo {
    hash: u64,
    captured: Piece,
    last_move: Move,
    in_check: bool,
}
```

## 合法手生成

### 処理フロー

1. **疑似合法手生成** (`generate_pseudo_legal_moves`)
   - 盤上の自駒の移動手(成り/不成の両方)
   - 駒打ち(空きマスへ)

2. **合法性フィルタリング** (`is_legal`)
   - 自玉の王手放置チェック(ピン含む)
   - 打ち歩詰めチェック

3. **疑似合法手生成時の制約**
   - 二歩: 同じ筋に自分の歩がある場合，歩打ちを除外
   - 行き所のない駒: 歩・香の最奥段，桂の最奥2段への打ちを除外
   - 強制成り: 行き所がなくなる場合は成りのみ生成

### 特殊ルール

#### 打ち歩詰め

歩を打って相手玉に王手をかけ，かつ相手に合法手がない場合は不合法．
歩打ちが王手になる場合のみ判定を実行する(パフォーマンス最適化)．

#### 連続王手の千日手

`Position`レベルで判定する．手を指した結果，同一局面が4回出現し，
その間の自分の手が全て王手だった場合，その手を合法手から除外する．

## 特徴平面 (piece_planes)

### 形状

`(104, 9, 9)`, dtype: `float32`

### チャネル配置

| チャネル | 内容 |
|---------|------|
| 0-7 | 先手盤上駒 (歩,香,桂,銀,角,飛,金,王) |
| 8-13 | 先手成駒 (と,成香,成桂,成銀,馬,龍) |
| 14-21 | 後手盤上駒 |
| 22-27 | 後手成駒 |
| 28-65 | 先手持ち駒 (歩×18, 香×4, 桂×4, 銀×4, 金×4, 角×2, 飛×2) |
| 66-103 | 後手持ち駒 |

- `piece_planes()`: 先手視点(そのまま)
- `piece_planes_rotate()`: 後手視点(盤面180度回転 + 先後入替)

## クレート構造

```
rust/maou_shogi/
├── Cargo.toml
├── src/
│   ├── lib.rs           # Public API re-exports
│   ├── types.rs         # Color, PieceType, Piece, Square
│   ├── piece.rs         # 駒ID変換 (cshogi↔domain)
│   ├── bitboard.rs      # Bitboard型と演算
│   ├── board.rs         # Board(局面表現)
│   ├── movegen.rs       # 合法手生成
│   ├── moves.rs         # Moveエンコーディング/USI変換
│   ├── position.rs      # Position(履歴付き局面)
│   ├── sfen.rs          # SFENパース/シリアライズ
│   ├── zobrist.rs       # Zobristハッシュ
│   ├── feature.rs       # piece_planes生成
│   └── attack.rs        # 利き計算
├── tests/
│   ├── fixtures/        # JSON test fixtures (cshogiから生成)
│   └── fixture_tests.rs # 統合テスト
└── generate_fixtures.py # cshogiからテストデータ生成
```

## テスト戦略

### Fixture-based testing

cshogiをリファレンス実装として，`generate_fixtures.py`でJSON形式のテストデータを事前生成する．
Rustの統合テストでこれらのfixtureを読み込み，出力が一致するか検証する．

### テスト種別

| Fixture | 検証内容 |
|---------|---------|
| sfen_fixtures.json | SFEN→pieces配列/hand配列/turn + roundtrip |
| move_fixtures.json | Move→move16/to_sq/is_drop/is_promotion/USI roundtrip |
| legal_move_fixtures.json | 各局面の全合法手リスト(sorted) |
| special_rule_fixtures.json | 二歩/打ち歩詰めの特殊ルール検証 |
| feature_fixtures.json | piece_planes/piece_planes_rotate出力の一致 |

### 単体テスト

各モジュールに`#[cfg(test)]`ブロックで単体テストを配置．
座標系，駒の利き，Zobristハッシュ等の基本動作を検証する．

## 今後の拡張

### Phase 6: PyO3バインディング

`maou_rust`クレートからmaou_shogiを呼び出すPyO3ラッパーを作成し，
Python側の`shogi.py`をmaou_shogi呼び出しに切り替える．
