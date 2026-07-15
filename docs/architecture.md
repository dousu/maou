# アーキテクチャ

## 基本的な設計方針

基本的にはクリーンアーキテクチャ inspired な形にする．

基本的な依存関係は以下でおおまかなフォルダも同名になっている．

```mermaid
graph TD;
infra --> interface
interface --> app
app --> domain
```

## infra

- DBやUI等のアプリケーション外部との接続の一番外側なコード
- ここにはビジネスロジックは一切いれずに純粋に外部のDB等とコミュケーションするために一般的に必要なものだけが入っている
- いわゆるプレゼンテーション層といわれるのもここにいれる

## interface

- presenter，controller，gatewayがあるが，要するにここではinfraの取り扱う一般的な形で解釈できるように変換する
- interfaceではentityを取り扱わずに言語の組み込み型だけ使う．これにより，entityへの依存がなくなり，interfaceを読むためにentityを読む必要がなくなる．
- 入力
    - infraを純粋な技術だけにして，appからinfraごとに必要な変換を取り除くといった上下の役割の凝集度をあげる効果がある
      - validation系もinfraから切り離してinterfaceに書いてしまったほうがよさそう
- 出力
    - infraは組み込みの言語の型で返せばよく，変な変換を入れることもしなくていい (変な変換とは1が返ってきたらtrueにするとか)
    - interfaceがappに返すときは，entityでラップするとかはしなくていいが，1が返ってきたらtrueにする等の抽象化は行う．

## app

- 各ユースケースを取り扱う．インプットはinterfaceから受け取って，domainを操作して，結果を出す．必要であればinterface経由でデータを取得してentityを作成して操作する (entityにすることでドメインルールが適用されるのでデータをそのまま使わない)．
- 依存関係の一方向性を持たせるために，domainサービスは存在せず，appでサービスを作る
    - ドメインサービスとは，ある一つのentityに依存しないルールを記載する時に使う．例えば，IDの一意性チェックだと個別のID entityの機能としてあると不自然なのでドメインサービスで実装する．
        - 一意性チェックであればinterfaceを使ってDBへのアクセスが必要

## domain

- ここではentityだけが存在する

## Shogi Engine (Rust) Encapsulation

The project uses the in-house Rust engine `maou_shogi` (exposed via the
`maou._rust` PyO3 extension) for Shogi game logic. Board-level logic is
**encapsulated within the domain layer** following Clean Architecture.
See [docs/rust-backend.md](rust-backend.md) for the crate structure.

### Encapsulation Rules

**Board-level logic (single position operations):**
- ✅ `src/maou/domain/board/shogi.py` - `Board` wrapper around
  `maou._rust.maou_shogi.PyBoard` (PRIMARY abstraction point) plus
  move utility functions and piece ID conversions
- ❌ `src/maou/app/**`, `src/maou/interface/**`, `src/maou/infra/**` -
  MUST use the domain `Board` class, not `maou._rust.maou_shogi.PyBoard`
  directly
- ✅ `tests/**` - Direct usage allowed for test simplicity (but Board
  usage preferred)

**Bulk data-pipeline APIs** (`maou._rust.maou_search`,
`maou._rust.maou_convert`, `maou._rust.maou_io`): app-layer use cases
and domain data modules call these directly by design — batching in
Rust avoids per-position Python loops (HCPE conversion, stage2
generation, preprocessing).

### Piece ID Mapping (CRITICAL)

The engine's raw piece IDs and the domain `PieceId` enum use
**DIFFERENT orderings**:

| Piece | raw ID (engine) | PieceId enum | Conversion |
|-------|-----------------|--------------|------------|
| 金(GOLD) | 7 | 5 (KI) | Reordered |
| 角(BISHOP) | 5 | 6 (KA) | Reordered |
| 飛(ROOK) | 6 | 7 (HI) | Reordered |
| 白(WHITE) | black+16 | black+14 | Offset difference |

**Conversion (single source of truth):**
- `shogi.RAW_PIECE_TO_PIECEID` - lookup table (module-level, numpy)
- `Board.raw_piece_to_piece_id()` - scalar conversion helper

**IMPORTANT:** All piece ID conversions MUST go through the
centralized table. Never implement conversion logic elsewhere.

### Anti-Patterns (DO NOT DO THIS)

```python
# ❌ BAD: Direct PyBoard usage in app layer
from maou._rust.maou_shogi import PyBoard

# ✅ GOOD: Use domain Board wrapper
from maou.domain.board.shogi import Board
```

```python
# ❌ BAD: Duplicate piece ID conversion logic
def my_converter(piece):
    if piece == 5:
        return 6  # BISHOP
    # ...

# ✅ GOOD: Use centralized conversion
piece_id = Board.raw_piece_to_piece_id(raw_piece)
```

## Data I/O Architecture

### Centralized Schema Management
```python
from maou.domain.data.schema import get_hcpe_dtype, get_preprocessing_dtype
from maou.domain.data.io import save_hcpe_array, load_hcpe_array

# Standardized data types
hcpe_dtype = get_hcpe_dtype()
preprocessing_dtype = get_preprocessing_dtype()

# High-performance I/O
save_hcpe_array(array, "output.hcpe.npy", validate=True)
loaded_array = load_hcpe_array("input.hcpe.npy", validate=True)
```

### Explicit Array Type System
**CRITICAL**: Always specify `array_type` parameter:

```python
# File system data source
datasource = FileDataSource(
    file_paths=paths,
    array_type="hcpe"  # REQUIRED
)

# S3 data source
datasource = S3DataSource(
    bucket_name="my-bucket",
    array_type="preprocessing"  # REQUIRED
)
```

Available types: `"hcpe"` (game records), `"preprocessing"` (training features)

## 追加資料

- [HCPE変換コマンドのデータフロー](commands/hcpe_convert.md)
- [学習コマンドとデータフロー (CLIとシーケンス図)](commands/learn_model.md)
