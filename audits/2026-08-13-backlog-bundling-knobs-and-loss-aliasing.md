---
path:
  - src/maou/infra/object_storage
  - src/maou/app/learning
  - docs/rust-backend.md
  - scripts
kind: backlog
scope: python
level: medium
status: done
started: 2026-08-13
last_sha: cb21490
---

# `/audit-backlog` — bundling ノブの不整合 (O5(b)(c))，loss テンソルの所有権 (N8)，`.npy` 遺物スクリプト

## Summary

`audits/coverage.md` の 2 表にある **14 行**を HEAD (`cb21490`) に対して
再検証した．stale 0 / changed shape 4 / confirmed 10．

自動帯は **P2 1 件 + P3 2 件**，判断帯は **N8 (P4)** と検証中に見つけた
**新規所見 1 件**．残る 11 行はすべてゲート付き (G1/G2/G3/G4) で，文言を
鋭くして残した．

**質問は 1 回だけ上げた** — 新規所見の `scripts/benchmark_file_datasource.py`
が「削除」と「`.feather` 版へ書き直し」で共有する行がゼロだったため，
one-check の split test の両半分を満たした．ユーザは「削除」を選択．

## Classification

| ID | 由来行 | クラス | クラスを決めたテスト | ゲート |
|---|---|---|---|---|
| P1-1 | (この run) | **P1** | `audits/` / `reviews/` は出荷ファイルに触れない → bump 不要 | — |
| P2-1 | O5(c) | **P2** | 訂正後の本文が現行コードから一意に決まる．姉妹 doc 3 本が 2026-08-13 の提案 `bundling-knobs-are-no-ops` (applied `a1ce41c`) で既に文言を確定させており，その 4 箇所目の取りこぼしを同じ文言で埋めるだけ — 新しい指針でも節の再構成でもない | — |
| P3-1 | O5(b) | **P3** | docstring のみの変更．返り値も生成物も不変 | — |
| P3-2 | O5(c) | **P3** | 既定値変更だが，`enable_bundling`/`bundle_size_gb` は**リポジトリ全体で一度も判定に使われていない**ため挙動不変と証明できる (AST テストで固定) | — |
| P4-1 | 新規 N9 系 | **P4** | 観測可能な挙動が変わる (スクリプトが落ちる→存在しなくなる)．`scripts/` は非パッケージ (`python-source = "src"`) なので bump 不要 | — |
| P4-2 | N8 | **P4** | `reset()` が呼び出し側テンソルを 0 にしなくなる．production では dormant だが「観測可能な変化」であることは変わらないのでフェイルセーフ側に倒した | — |

### 判断帯に残した 11 行とそのゲート

| 行 | クラス | ゲート | 一言 |
|---|---|---|---|
| Deferred 2 | P4 | **G3** + G4 | ~600 行の訓練経路リファクタ．この環境で等価性を示せない |
| Deferred 3 | P4 | **G2** + G4 | 統合はテスト 2 本の書き換えと不可分 |
| Deferred 5 | P4 | **G1** | dormant (`legal_move_mask` は `training_loop.py:507` で `None` 固定，産出者ゼロ) |
| Deferred 6 | P4 | **G1** | GPU が要る |
| Deferred 7 | P4 | **G1** | GPU + 数値等価性 |
| D5 | P6 | **G4** | ノブ廃止は O5 と一体 |
| D10+D11 (1) | P4 | **G4** | 「ファイル数」と「yield 数」のどちらを意味させるかの決め |
| D13 | P4/P6 | **G2** + G4 | 根治は `app/learning` と ABC を触る |
| D14 (b) | P6 | **G2** + G4 | ABC を外すとテスト 3 ファイルが壊れる |
| O5 (残り (a)(d) + 削除) | P6 | **G4** | bool flag と dir のどちらがキャッシュを有効にするかの決め |
| O9 | P4/P6 | **G1** + G4 | BigQuery が無い．修正の向きが 3 つに割れている |
| N4 | P4 | **G4** | 薄いテストへの切り出し vs CPU extra 必須化 |
| N6-2 | P6 | **G4** | 「HCPE 専用と明記して改名」は P6 を含む |

## Consumed

**行の削除はゼロ．** 指定ブランチ (`claude/audit-backlog-kpm6a7`) 1 本に
集約する制約があり，単一 PR が未マージのため，6a により消化した行も
in flight として残る．

| 行 | 対象 | 出荷したもの | commit |
|---|---|---|---|
| O5(c) の doc drift 部分 | `docs/rust-backend.md` | S3DataSource 例に「受理するが効果なし」の注記 | `70602b2` |
| O5(b) | `infra/object_storage` | `max_cached_bytes` docstring の訂正 | `10fe752` |
| O5(c) の既定値部分 | `infra/object_storage` | `enable_bundling` 既定を `False` に統一 + AST characterization test | `10fe752` |
| N8 | `app/learning` | `_last_batch_loss.copy_()` + 罠を突くテスト | `bd8b8b9` |
| (新規) | `scripts/` | 実行不能な `benchmark_file_datasource.py` を削除 | `901ad55` |

## Applied

- `docs/rust-backend.md:697-700` — bundling 引数に注記 (`70602b2`)
- `reviews/2026-08-13-rust-backend-bundling-example.md` — `status: applied`,
  `applied_in: 70602b2` (`566dd1a`)
- `src/maou/infra/object_storage/data_source.py:407-412` — `max_cached_bytes`
  の docstring を「並列DLのチャンク幅の元」に訂正 (`10fe752`)
- `src/maou/infra/object_storage/data_source.py:45` — `enable_bundling`
  既定 `True` → `False` (`10fe752`)
- `tests/maou/infra/object_storage/test_bundling_knobs_inert.py` — 新規
  (5 テスト) (`10fe752`)
- `src/maou/app/learning/callbacks.py:1030-1036` — `= loss_detached` →
  `.copy_(loss_detached)` (`bd8b8b9`)
- `tests/maou/app/learning/test_callbacks.py` —
  `test_last_batch_loss_does_not_alias_caller_tensor` 新規 (`bd8b8b9`)
- `scripts/benchmark_file_datasource.py` — 削除 (`901ad55`)

Version: `0.89.9` → `0.89.10` (object_storage) → `0.89.11` (callbacks)．
`docs/` / `reviews/` / `audits/` / `scripts/` の commit は bump 無し
(`scripts/` は `python-source = "src"` により非パッケージ)．

## In flight

**PR は 1 本のみ** — 指定ブランチの制約でクラス毎の分割ができず，
自動帯と判断帯が同じ PR に同居している．したがって**自動帯も未マージ**
であり，このセッションでは `main` に何も落ちていない．

レビュー単位は PR ではなく **commit** が担う:

| commit | クラス | 内容 |
|---|---|---|
| `70602b2` + `566dd1a` | P2 | doc drift + 提案の status 遷移 |
| `10fe752` | P3 | object_storage の docstring + 既定値 |
| `bd8b8b9` | P4 | N8 の `copy_()` |
| `901ad55` | P4 | `.npy` 遺物スクリプトの削除 (ユーザ承認済み) |

未解決の問いは **N8 の `copy_()` を受け入れるか**の 1 点のみ．
それ以外はユーザが既に回答済み (スクリプト削除) か，判断を要さない．

## Re-triaged

上の「判断帯に残した 11 行」の表を参照．すべて行を残し，文言を鋭くした．
今回の再検証で**特に価値があった**のは次の 3 件:

1. **Deferred 3** — 前 run (2026-08-13) が書いた訂正 (i) 自体が誤りだった．
   「`else` 腕が未対応 head 型を弾く検証」ではなく**到達不能な防御コード**．
   2 世代続けて同じ分岐を読み違えており，「`isinstance` 分岐を見たら
   呼び出し元が型を静的に知っているかを先に確認する」が教訓．
2. **D14b** — 記録が挙げる「path 外の障害」が実在しなかった．
   行番号だけを頼りに障害を記録すると，コードが動いた後に**存在しない
   障害**が残り，item を不必要に判断帯へ固定し続ける．
3. **D10+D11** — 「テストも無い」が誤り．テストはあるが問題の条件
   (`cache_mode="memory"`) を踏んでいなかった．「テストが無い」と
   「テストが問題を踏んでいない」は必要な作業が違う．

## Corrections to the source records

- [`2026-08-08-src-maou-app-learning.md`](2026-08-08-src-maou-app-learning.md)
  — Deferred 3 の前 run 訂正 (i) の撤回，および Deferred 2 の
  「4 本目の挙動軸」「28 行 vs 38 行」の訂正．
- [`2026-08-10-src-maou-infra-file-system.md`](2026-08-10-src-maou-infra-file-system.md)
  — D14(b) の「path 外の障害」の撤回 (真の障害はテスト群) と
  `_use_columnar` 分岐の役割分解，D10+D11 の「テストも無い」の訂正．

いずれも**診断が誤っていた**ケースに限った訂正で，worklist state
(`RESOLVED` 等) は書いていない．

**Correction** (2026-08-15, `033d49f`): この run が
`2026-08-08-src-maou-app-learning.md` の Deferred 2 に追記した訂正 (i)
— 「**4 本目の軸**として使う `TrainingLoop` サブクラスが違う
(`Stage1TrainingLoop` vs `RawLogitsTrainingLoop`)．装飾ではなく挙動の
軸なので，『差分は装飾だけ』という前提で統合を設計すると取り落とす」
— は**誤り**である．`training_loop.py:1183` は
`Stage1TrainingLoop = RawLogitsTrainingLoop` という**モジュール level の
別名**であり，`git log -S` によればこの別名は **2026-08-09 の
`568863f`**「全 1 の legal_move_mask を targets タプルから外す」で
導入されている — すなわち**本 run (2026-08-13) の時点で既に別名だった**．
`multi_stage_training.py` の 2 箇所は**同じクラスを構築している**ので，
挙動の軸は存在しない．

見落とした理由は，2 つの構築箇所の**名前が違うことだけを確認し，
その名前の定義を読まなかった**ため．別名は定義側にしか現れないので，
利用箇所の grep では区別がつかない．**「2 つの名前が違う」は
「2 つのクラスが違う」の証拠にならない** — 定義を引くまでは，
それは仮説である．

この誤りには下流の影響がある: 2026-08-14 にユーザが下した設計判断
「`TrainingLoop` サブクラスは戦略として注入する」は，この訂正を
根拠に提示された選択肢なので，**存在しない差異のための設計**に
なっている (`audits/coverage.md` の Deferred 2 行に反映済み)．

## Doc findings

- [`reviews/2026-08-13-rust-backend-bundling-example.md`](../reviews/2026-08-13-rust-backend-bundling-example.md)
  — `status: applied`, `applied_in: 70602b2`．
  **P2 の常設承認 (CLAUDE.md) で適用**．訂正後の本文が一意に決まる根拠は，
  先行提案 `bundling-knobs-are-no-ops` が同じ drift に対する文言を既に
  確定させていること．新しい指針を足していない．

## Out of scope

- **N9 (起票済み)** — `scripts/debug_training.py:30` と
  `scripts/verify_bce_training.py:23` が `*.npy` を glob している．
  削除した `benchmark_file_datasource.py` と同じ `.npy` 時代の遺物だが，
  選択された item の外なので触っていない (**G2**)．`coverage.md` の
  Out-of-scope backlog に行を追加した．

## Environment notes

- **コンテナは再作成されており venv は空だった** (site-packages 2 エントリ)．
  `uv sync --extra cpu` に約 7 分，`uv run maturin develop --release` に
  **7 分 4 秒**．入れ終えるまで `pytest` すら存在しなかった．
  これは N4 行の実害の **6 回連続**の確認である．
- **G1 で回せなかったもの**: GPU (Deferred 5/6/7)，BigQuery (O9)，
  数百ファイルのネットワークストレージ (D10+D11 の測定)．
- **実際に回した QA**: `ruff format` (2 ファイル整形) /
  `ruff check --fix` (All checks passed) / `mypy src/`
  (135 ファイル，issue なし) / `pytest tests/` (**1956 passed,
  54 skipped**)．
- Rust クレートは変更していないので `cargo` 系は未実行．
- リポジトリは PR に対してテストを回さない (`claude-code-review.yml` は
  `workflow_dispatch` のみ) ため，上の実行結果が唯一の証拠である．

## Reconciliation (6d)

```
触れた item 14 + 新規所見 2 = 16
  = resolved 0            (行削除ゼロ — 単一 PR が未マージのため)
  + in flight 3           (O5 / N8 / benchmark_file_datasource.py の削除)
  + re-triaged 12         (Deferred 2,3,5,6,7 / D5 / D10+D11 / D13 / D14b /
                           O9 / N4 / N6-2)
  + new rows 1            (N9 — 残る 2 本の `.npy` 遺物スクリプト)
  + not a finding 0
```

内訳の注記:

- **O5 は "in flight" に 1 回だけ数えている**．(b) と (c) の一部を出荷した
  ので行は動いており，残り ((a)(d) + ノブ削除) は同じ行の中で文言を
  鋭くした — 2 つの disposition には割らない．
- **Deferred 3 の `else` 腕**は独立した item として数えていない．到達不能
  だと判明したが，削除すると将来 head 型を足したとき `UnboundLocalError`
  になるだけで改善にならないため，Deferred 3 の判断に内包させた．

backlog 行数: **14 → 15** (N9 を追加，削除ゼロ)．
