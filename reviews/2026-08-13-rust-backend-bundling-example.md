---
status: pending
date: 2026-08-13
target:
  - docs/rust-backend.md
risk: low
reversibility: trivial
---

# `docs/rust-backend.md` の S3DataSource 例が死んだノブを「効く引数」として見せている

## Trigger

`audits/coverage.md` § Out-of-scope backlog の **O5(c)** 行
([2026-08-10 infra/file_system](../audits/2026-08-10-src-maou-infra-file-system.md))．
`/audit-backlog` (2026-08-13, `cb21490`) で HEAD に対して再検証した．

本件は **2026-08-13 の先行提案
[`bundling-knobs-are-no-ops`](2026-08-13-bundling-knobs-are-no-ops.md)
(status: applied, `a1ce41c`) の取りこぼし**である．
その提案は `docs/commands/` の 3 本に「受理するが効果なし」と明記したが，
`docs/rust-backend.md` の**コード例**が同じノブを引数として書いており，
そちらは直っていなかった．grep で 4 箇所目として surface した．

## 現行コードが何をするか (訂正文の一意性の根拠)

先行提案が示した事実は今も成立し，今回さらに強く確認できた:

| 位置 | 事実 |
|---|---|
| `src/maou/infra/object_storage/data_source.py:197-201` | `__download_all_to_local` の docstring が *"enable_bundling and bundle_size_gb are ignored for .feather files"* と明言 |
| 同 `:303-323` | ダウンロード完了ハンドラは各オブジェクトを `feather_path.write_bytes(byte_data)` で無条件に個別保存する |
| `grep -rn "if.*enable_bundling\|not enable_bundling\|self\.enable_bundling" src/ tests/ scripts/` | **ヒットは docstring の 1 行のみ** (`:199`)．リポジトリ全体で値を判定に使う場所が存在しない |

`enable_bundling` / `bundle_size_gb` は CLI → `DataSourceSpliter` →
`PageManager` と受け渡されるだけで，属性にすら保存されず捨てられる．

したがって訂正後の本文は現行コードから一意に決まる．
**書き方の選択肢は先行提案が既に確定させている** —
姉妹 doc 3 本が採った「受理するが現状は効果がない」という文言を
4 箇所目にも適用するだけであり，新しい指針でも節の再構成でもない．
P2 = drift correction．

## Before

`docs/rust-backend.md:691-701`

```markdown
# S3 DataSource
s3_datasource = S3DataSource(
    bucket_name="my-bucket",
    prefix="training-data",
    data_name="hcpe-202412",
    local_cache_dir="./cache",
    array_type="hcpe",
    max_workers=16,
    enable_bundling=True,
    bundle_size_gb=1.5,
)
```

## After

```markdown
# S3 DataSource
s3_datasource = S3DataSource(
    bucket_name="my-bucket",
    prefix="training-data",
    data_name="hcpe-202412",
    local_cache_dir="./cache",
    array_type="hcpe",
    max_workers=16,
    # enable_bundling / bundle_size_gb は受理されるが現状は効果がない
    # (.feather は常に個別保存される)．docs/commands/pre_process.md 参照．
    enable_bundling=True,
    bundle_size_gb=1.5,
)
```

## なぜ削除ではなく注記か

姉妹 doc 3 本 (`docs/commands/pre_process.md` ほか) が採った形と揃える．
引数自体は API として今も受理されるので，例から消すと「渡せない」と
読めてしまう．「渡せるが効かない」が事実であり，先行提案が確定させた
表現でもある．

## 影響

なし (コード変更を伴わない doc のみ)．O5 行そのもの — ノブを削除するか
残すかの設計判断 (P6 + G4) — には踏み込まない．
