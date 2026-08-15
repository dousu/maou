---
kind: backlog
date: 2026-08-15
path:
  - src/maou/app/learning/multi_stage_training.py
  - src/maou/app/learning/stage_component_factory.py
  - src/maou/app/learning/dataset.py
  - src/maou/app/learning/streaming_dataset.py
  - src/maou/app/learning/gradient_noise_scale.py
  - src/maou/app/learning/adaptive_batch.py
  - src/maou/infra/console/learn_model.py
  - src/maou/infra/console/utility.py
  - docs/commands/learn_model.md
scope: python
level: medium
last_sha: 377e1e3
record_sha: 57f0664
---

# backlog consumption — Stage1/Stage2 の統合と，GNS 計測間隔の既定値

`/audit-backlog` (2026-08-15, `pw4zzw`)．前 run
([2026-08-15 writeable-contract-and-decisions](2026-08-15-backlog-writeable-contract-and-decisions.md))
が「決定済みだが未実装」として残した 2 件を出荷した run である．

**backlog は 5 行で始まり 4 行で終わった** — 15 run ぶりに行が 1 本減った．
減らせた理由は本 run が賢かったからではなく，**過去 4 run の設計判断が
在庫を作っていたから**である: 5 行のうち 3 行 (Deferred 2 / D13 / O9) は
すでに「人間待ちではないただの作業」で，本 run はそのうち最も大きい
Deferred 2 (~585 行) を消化しただけである．

**自動帯は空 (15 run 連続)．** 5 行すべてが P4 以上だった．

## Classification

| ID | backlog 行 | Target | クラス | クラスを決めたテスト | ゲート | 再検証 |
|---|---|---|---|---|---|---|
| **B-1** | Deferred 2 | `src/maou/app/learning` | **P4** | 挙動不変を意図した refactor だが ~585 行．「全出力が同一か」への正直な答えが "probably" なので**フェイルセーフで P4** (ladder の「迷ったら上」)．P3 を主張するには等価性の証明が要り，それは本 run の予算を超える | **なし** (G3 は 2026-08-15 にユーザが retire) | **confirmed** |
| **B-2** | Deferred 7 の緩和策 | `app/learning` + `infra/console` | **P4** | フラグを渡していない既存の実行で GNS 計測頻度が 1/5 になる．データは読める，起動行は有効なので P5/P6 ではない | **なし** | **changed shape** |
| — | Deferred 5 | `src/maou/app/learning` | P4 | — | **G1** | **confirmed** |
| — | D13 (2)(3)(4) | `infra/file_system` → `app/learning` | P4/P6 | — | **G2** | **confirmed** |
| — | O9 | `src/maou/infra/bigquery` | P4 | — | **G1 (縮小済み)** | **confirmed** |

再検証は **stale 0 / changed shape 1 / confirmed 4**．前 run 以降 `src/` への
コミットが無かったため，**5 行すべてで行番号の移動がゼロ**だった (2 run 連続)．

**B-1 のクラスについて**: 本 run は P3 を主張しなかった．ladder の P3 は
「あらゆる入力に対して書き出す成果物と返り値が不変」を要求し，本 run が
実際に固定できたのは既存 suite (2016 passed) とログ文言の一致までである．
~585 行の restructure でそれを「不変の証明」と呼ぶのは過大なので P4 に
置いた．**この判断自体がユーザに問う対象**である (下記 Q1)．

## Consumed

| backlog 行 | Target | 出荷したもの | commit |
|---|---|---|---|
| **Deferred 2** — 全消化，**行を削除** | `app/learning` | Stage1/Stage2 の重複 4 組を共通実装へ | `fdbc990` |
| **Deferred 7** — **緩和策のみ**，行は残す | `app/learning` + `infra/console` | `measurement_interval` の既定 1 → 5 (4 箇所) + doc + 回帰テスト | `57f0664` |

## Applied

**B-1 (`fdbc990`) — Stage1/Stage2 の 4 組を統合**

| 対象 | 変更前 | 変更後 |
|---|---|---|
| run 関数 2 本 | `multi_stage_training.py:376-523` / `:525-672` | `_run_stage_with_training_loop` を head 型 / callback 生成 / metric 取得 / ログ見出し 2 種でパラメータ化．公開関数 2 本は署名・戻り値型ともそのままの薄いラッパ |
| 工場の完全一致 28 行 | `stage_component_factory.py:704-731` ≡ `:794-821` | `_assemble_stage_components` へ切り出し |
| dataset の対 | `dataset.py:242-316` / `:319-391` | `_StageDataset` 基底 + 教師フィールド名と平坦化有無だけを持つサブクラス 2 つ |
| streaming の対 | `streaming_dataset.py:834-891` / `:894-948` | `_yield_stage_batches` へ集約．既存の 2 名はラッパとして残し，呼び出し側 3 箇所は不変 |

**`TrainingLoop` サブクラスの注入機構は作っていない．** 2026-08-14 の設計
判断はこれを含んでいたが，2026-08-15 に前 run が
`training_loop.py:1183` の `Stage1TrainingLoop = RawLogitsTrainingLoop` が
2026-08-09 (`568863f`) 以来の別名だと確定させており，**注入すべき差異が
存在しない**．統合はその分素直になった (差分は head クラス / callback
クラス / metric getter / ログ 2 本のみ)．

**B-2 (`57f0664`) — GNS 計測間隔の既定 1 → 5**

| # | 場所 | 備考 |
|---|---|---|
| (i) | `gradient_noise_scale.py:85` | クラス既定 |
| (ii) | `adaptive_batch.py:68` | dataclass フィールド既定 |
| (iii) | `infra/console/learn_model.py:217` | **本番経路を決めるのはここ** |
| (iv) | `infra/console/utility.py:688` | `benchmark-training` の同名オプション |

## Decisions asked

3d の `AskUserQuestion` は **受理 1 問 + 設計判断 1 問**．G4 の行は 1 つも
無いので (2026-08-14 の 3 run で全て retire 済み)，設計判断の枠は
「ゲートが塞いでいる行を人間待ちから外せる問い」に充てた．

TBD_ANSWERS

## In flight

TBD_INFLIGHT

## Re-triaged

- **Deferred 5** (P4 + G1) — 行番号の移動なしを再確認 (`:1117` の同期，
  `:1122` の 2 つ目の同期，`:514` のハードコード `None`，`:523` の唯一の
  `TrainingContext` 構築，`:1183` の別名)．**この run では出荷していない**．
  休眠の根拠は 2 重 (産出者ゼロ + `RawLogitsTrainingLoop` がマスク処理を
  クラスとして迂回) で，前 run から変わっていない．
- **D13 (2)(3)(4)** (P4/P6 + G2) — 行番号の移動なしを再確認．設計も
  writeability 契約も決定済みで**人間待ちではない**．G2 の作業量が
  B-1 (~585 行) と同居できなかっただけである．**次 run の先頭候補**．
- **O9** (P4 + G1 縮小済み) — 行番号の移動なしを再確認．設計も
  fake client のテスト土台の方針も決定済みで**人間待ちではない**．
  (0) の土台新設まで含めると本 run の枠に入らなかった．

## Corrections to the source records

**元記録への訂正は無し．** B-2 の "changed shape" は元記録
(`2026-08-08-src-maou-app-learning.md` Deferred 7) の診断の誤りではなく，
**前 run が backlog 行に書いた scope 説明の不足**である (行の側で訂正済み)．

前 run は「既定は 3 箇所」と書いたが，実際は **4 箇所**だった —
`infra/console/utility.py:688` の `benchmark-training` の同名オプションが
数えられておらず，`utility.py:1365` が常に明示的に渡すため，これを直さないと
`benchmark-training` 側だけ旧既定のまま残っていた．

## Doc findings

- `reviews/2026-08-15-gns-measurement-interval-default.md` — **status:
  applied** (`applied_in: 57f0664`)．CLAUDE.md の drift correction 恒久承認
  (P2) で適用．`docs/commands/learn_model.md:32` の既定セルは
  `learn_model.py` の `@click.option` の `default=` を写したものなので，
  コードが `5` になった以上セルの値は一意に決まる．
  既定値を**上げるという決定そのもの**は P2 由来ではなく 2026-08-15 に
  ユーザが 3d で選んだ判断帯の回答である — P2 が覆うのは doc 1 セルの
  追随だけ．
- `docs/commands/utility_benchmark_training.md:48` は同オプションを他の
  adaptive batch オプションとまとめた 1 行で扱い**既定値を書いていない**
  ので，訂正対象外 (再確認済み)．

## Out of scope

新規の out-of-scope 所見は**なし**．

本 run が気づいた 1 点は backlog 行に書くほどのものではないので，ここに
記録だけしておく: `tests/maou/app/learning/test_gradient_noise_scale.py::
test_reset_between_cycles` が `measurement_interval` を渡さず**既定が 1 で
あることに暗黙に依存**していた．B-2 で顕在化し同 commit で直した
(検証意図は「サイクル間で内部状態がリセットされること」なので
`measurement_interval=1` を明示する形にした)．**既定変更が実際に挙動を
変えることの独立した証拠**でもある．

## Environment notes

- コンテナが素の状態で起動したため `uv sync --extra cpu` からやり直した．
  torch のダウンロードが agent proxy 経由で遅く，**既定の
  `UV_HTTP_TIMEOUT=30` では `mpmath` の取得に失敗**した
  (`Failed to download distribution due to network timeout`)．
  `UV_HTTP_TIMEOUT=300` で再実行して torch 2.11.0+cpu が入った．
  次 run も同じ足止めを食う可能性が高いので，**最初から
  `UV_HTTP_TIMEOUT=300 uv sync --extra cpu` を打つとよい**．
- **G3 は発生していない．** 全 QA をこの環境で実行できた:
  `ruff format` / `ruff check src+tests` / `mypy src` (135 files) /
  `mypy src+tests` (pre-commit) / `bash scripts/check-cli-docs.sh` /
  `pytest` **2016 passed, 53 skipped** (`gradio` 未導入で
  `tests/maou/infra/visualization/test_indexing_status.py` の 1 モジュールが
  未 collect である旨は conftest が明示している)．
- 無効化テストは 3 通り実施し，いずれも検出されることを確認した:
  (a) `metric_label` を Stage 1 側で `"F1"` に取り違え → `TestStageRunLogText`
  1 本が落ちる，(b) `head_type` を Stage 1 側で `LegalMovesHead` に取り違え
  → `TestStageRunHeadTypeAssertion` 1 本が落ちる，(c) **CLI 既定だけを
  `1` に戻す** (クラス既定は 5 のまま) → `test_gns_measurement_interval_
  default.py` の 2 本が落ちる．(c) が B-2 の罠そのものである．
- GPU は無い．Deferred 5 / Deferred 7 の本丸 / O9 の実クエリ確認は
  この環境では**測れない** (G1)．
