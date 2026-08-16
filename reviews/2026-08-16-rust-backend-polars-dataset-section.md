---
status: applied
applied_in: 654986c
date: 2026-08-16
target: [docs/rust-backend.md]
risk: low
reversibility: trivial
---

# `docs/rust-backend.md` の「PyTorch Dataset with Polars DataFrames (Phase 5)」節を削除する

## Trigger

`/audit-backlog` (2026-08-16, `bg1urj`)．backlog 行 D13 の消化中に
`PolarsDataFrameSource` (`src/maou/app/learning/polars_datasource.py:54`)
が **ABC を継承しない duck-typed なソース**であり，学習側 `DataSource`
に列アクセサを足すにあたって `KifDataset` に `isinstance` ガードを
要求する唯一の理由になっていることが表面化した．

step 3d でユーザに扱いを問い，**「削除する」**が選択された
(却下: 「`DataSource` を実装させる」「現状のまま残す」)．

## 検証結果 (HEAD `b3fe0a2`)

- `PolarsDataFrameSource` の production からの呼び出しは**ゼロ**．
  `src/` 全体でのヒットは定義ファイル自身のみ
  (`benchmark_polars_io.py:386` の `polars_datasource` は**ローカル変数名**
  であって `FileDataSource` のインスタンスである — 同名の別物)．
- 参照は `tests/maou/app/learning/test_polars_datasource.py` と
  **`docs/rust-backend.md:740`** の 2 箇所だけ．

## 変更

`docs/rust-backend.md` の `### PyTorch Dataset with Polars DataFrames (Phase 5)`
節を**丸ごと削除する** (`## File Format Migration` の直前まで，61 行)．

### before

````markdown
### PyTorch Dataset with Polars DataFrames (Phase 5)

The project now supports using Polars DataFrames directly with PyTorch Dataset and DataLoader:

```python
import polars as pl
from torch.utils.data import DataLoader

from maou.app.learning.polars_datasource import PolarsDataFrameSource
from maou.app.learning.dataset import KifDataset
from maou.domain.data.rust_io import load_preprocessing_df
...
```

**Benefits of Polars Dataset:**
...

**Supported Data Types:**

| array_type | Schema | Dataset Class | Status |
|------------|--------|---------------|--------|
| `"preprocessing"` | Full training data | `KifDataset` | ✅ Tested |
...
````

### after

(節ごと削除．直前の `iter_batches()` に関する段落と，直後の
`## File Format Migration` が隣り合う．)

## P2 判定 — なぜ standing approval で適用できるか

CLAUDE.md § "Standing approval — drift corrections only" の判定基準は
**「訂正後の本文が現行コードから一意に決まるか」**である．

この節は存在しないモジュールの import 例と，そのクラスの利点・対応表で
**構成が丸ごと `PolarsDataFrameSource` に従属している**．クラスが無く
なった以上，節を残す書き方は存在しない．**削除は一意に決まる．**

**置き換えの本文は書かない．** 「Polars DataFrame を学習に載せる正しい
入口は `FileDataSource` である」といった案内文は，あれば有用だが
**現行コードから一意には決まらない** (どこに何行で書くか，どの経路を
推すかに複数の書き方がある)．それは新しい指針であって drift correction
ではないので，この提案には含めない．必要なら別提案とする．

## 影響

- `docs/rust-backend.md` の他の節は不変．削除した節は Polars の I/O 性能
  (`iter_batches_df()` ほか) には触れておらず，そちらの記述は前後に
  独立して残っている．
- `docs/` 内の他ファイルからこの節への参照は無い (`grep` 済み)．
