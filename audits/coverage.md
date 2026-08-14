# Audit coverage ledger

One row per path touched by `/audit-and-fix`. Shape, status vocabulary,
and protocol: [README.md](README.md).

**This table lists only what has been audited.** It is not a plan and not
an inventory of remaining *paths* — to see which paths are left, compare
against the tree (`ls src/maou/*/`, `ls rust/`, `find docs -name '*.md'`),
which is always current where a checked-in list would not be.

Remaining *findings* are a different question, and they **are** inventoried
here: see § "Open findings backlog" below. That is the live worklist; the
per-run records are immutable accounts and are never read to decide what
work remains.

| Path | Scope | Status | Level | Last SHA | Record | Open items |
|---|---|---|---|---|---|---|
| `src/maou/domain/game_graph` | python | done | high | `2686689` | [2026-08-08](2026-08-08-src-maou-domain-game-graph.md) | 0 |
| `src/maou/app/learning` | python | done | high | `52d9bd2` | [2026-08-08](2026-08-08-src-maou-app-learning.md) | 3 deferred |
| `src/maou/infra/file_system` | python | done | high | `1c6a442` | [2026-08-10](2026-08-10-src-maou-infra-file-system.md) | 0 |

## Blocked

_(none)_

<!-- Rows move here only while status is `blocked`, with the blocker and
     what would unblock it. Keep the main table for in-progress/done. -->

## Open findings backlog — the single live worklist

The two tables below are **the** authority on what audit work remains.
Both `/audit-and-fix` and `/audit-backlog` gather candidate work from
here and **only** from here.

**Why this file and not the records.** A per-run record is read only when
someone opens that specific path — so a finding left there is visible
exactly to the audit least able to act on it. This file is read at the
start of every run.

**Why the records are not also consulted.** A record is an *immutable
account of one run at one time*: its Deferred section says "as of that
run, this was deferred", and that stays true forever even after the
finding is fixed. Reading records for open work therefore re-surfaces
resolved findings on every run, with no way to remove them — the ledger
would never shrink. Deleting a row here is what makes a finding
*consumed*, and it is the only mechanism that does.

**Protocol (both tables).**
- **Before auditing a path**, check both tables for rows whose target
  falls inside it and fold them into the run.
- **At the end of a run**, append a row for every finding left open —
  deferred (inside the path) and out-of-scope (outside it) alike.
  Writing it only into the run's record buries it.
- **When a finding is resolved, delete its row.** The resolving record is
  the durable account. Do not delete a row that was merely re-triaged —
  sharpen its text instead.

Records of runs that resolved rows deleted from here:

- [2026-08-09 backlog tier-a](2026-08-09-out-of-scope-tier-a.md)
- [2026-08-09 backlog contained-fixes](2026-08-09-backlog-contained-fixes.md)
- [2026-08-09 backlog streaming-len-and-docs](2026-08-09-backlog-streaming-len-and-docs.md)
- [2026-08-09 backlog tier-3-contracts](2026-08-09-backlog-tier-3-contracts.md)
- [2026-08-12 backlog auto-band-and-p4](2026-08-12-backlog-auto-band-and-p4.md)
- [2026-08-12 backlog n5-fill-accepted](2026-08-12-backlog-n5-fill-accepted.md)
- [2026-08-12 backlog auto-band-and-n1](2026-08-12-backlog-auto-band-and-n1.md)
- [2026-08-12 backlog arrow-format-and-clippy](2026-08-12-backlog-arrow-format-and-clippy.md)
- [2026-08-13 backlog columnar-dedup-and-split-seed](2026-08-13-backlog-columnar-dedup-and-split-seed.md)
- [2026-08-13 backlog scan-share-and-abc](2026-08-13-backlog-scan-share-and-abc.md)
  ([PR #492](https://github.com/dousu/maou/pull/492) — D8+D9 / N3 / N-1 /
  N5-1 の 4 行を削除．D14 は **(a) だけ**を消化したので行は残り，(b) の
  記述に縮めてある)
- [2026-08-13 backlog oom-estimate-and-bq-contract](2026-08-13-backlog-oom-estimate-and-bq-contract.md)
  および
  [2026-08-13 backlog truncated-feather-diagnostics](2026-08-13-backlog-truncated-feather-diagnostics.md)
  ([PR #493](https://github.com/dousu/maou/pull/493) — 2 つの run が
  同じ指定ブランチに積まれたため 1 本の PR にまとまっている)．
  **N6-1 / N7 / D15 の 3 行を削除．** ユーザが判断帯の 2 点を承認して
  マージした — P4 (BigQuery `iter_batches`) は「元々壊れていたので修正で
  問題ない」，P3+G1 (行数スキャンの並列化) は「測れていなくて問題ない」．
  O5 / D5 / D10+D11 は**行の一部だけ**の消化なので行は残り，それぞれ
  (c) の doc drift / 見積り部分 / (2) を消化済みと明記して縮めてある．
  D15 は "changed shape" で，記録の「size/footer 検査が要る」が誤り
  だったので [元記録](2026-08-10-src-maou-infra-file-system.md) に訂正を
  追記した．N6-2 行の前 run の記述も訂正済み．
- [2026-08-13 backlog callback-accumulator-table](2026-08-13-backlog-callback-accumulator-table.md)
  ([PR #494](https://github.com/dousu/maou/pull/494) — **Deferred 4 の 1 行を削除**)．
  14 行を再検証して stale 0 / changed shape 2 / confirmed 12．自動帯に
  入ったのは Deferred 4 だけで，残る 13 行はすべてゲート付きか判断帯
  だったため，質問を上げずに文言だけ鋭くして残した．Deferred 3 と O9 は
  "changed shape" — Deferred 3 は[元記録](2026-08-08-src-maou-app-learning.md)
  に訂正を追記した．新規所見 N8 を起票．
- [2026-08-13 backlog bundling-knobs-and-loss-aliasing](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md)
  ([PR #495](https://github.com/dousu/maou/pull/495) — **N8 の 1 行を削除**)．
  指定ブランチ 1 本の制約でクラス毎の PR 分割ができず，自動帯と判断帯が
  同じ PR に同居したため，レビュー単位は commit が担った．ユーザが内容を
  確認して承認・マージ (`95f31e3`)．14 行を再検証して stale 0 /
  changed shape 4 / confirmed 10．自動帯は P2 1 件 + P3 2 件，判断帯は
  N8 と新規所見 1 件．O5 は**行の一部だけ** ((b) と (c) の既定値) の消化な
  ので行は残り，残り (a)(d) + ノブ削除の記述に縮めてある．
  **元記録への訂正 3 件** — Deferred 2 (4 本目の挙動軸)，
  Deferred 3 (前 run の訂正 (i) 自体が誤り: `else` 腕は到達不能)，
  D14b/D10+D11 (path 外の障害とテスト不在の見立てが誤り)．新規所見 N9 を起票．
- [2026-08-14 backlog diagnostics-and-npy-remnants](2026-08-14-backlog-diagnostics-and-npy-remnants.md)
  (**N9 の 1 行を削除**)．指定ブランチ 1 本の制約でクラス毎の PR 分割が
  できず，自動帯 3 件と判断帯 1 件が同じ PR に同居したため，レビュー単位は
  commit が担った．14 行を再検証して stale 0 / changed shape 2 /
  confirmed 12．自動帯は P1 1 件 (テストの黙殺を可視化) + P2 1 件
  (新規に気づいた `.npy` 併存記述の drift) + P3 1 件 (pre-process の
  入力エラー文言)．N9 は「`.feather` 版へ書き直す」側が**修理ではなく
  書き下ろし**だと判明 (`Network` / `LossOptimizerFactory` /
  `KifDataset` の 3 API が全て変わっており，glob 以外に 3 つの
  破綻がある) したため削除の向きで出荷したが，「今は残して後で別物を
  書く」選択は利用者のものなので G4 は retire せず PR に判断を載せた．
  **ユーザが同一セッション内で「削除する」と回答**したため，そのまま
  マージした ([PR #498](https://github.com/dousu/maou/pull/498))．
  N4 と O5 は**行の一部だけ**の消化なので行は残り，それぞれ「黙って
  消える部分」「メッセージが原因を指していない部分」を消化済みと明記
  して縮めてある．元記録への訂正は**なし** (changed shape 2 件は
  いずれもコードが動いた結果で，診断の誤りではない)．
- [2026-08-14 backlog design-decisions](2026-08-14-backlog-design-decisions.md)
  (**N4 の 1 行を削除**)．改訂後の `/audit-backlog` (`88a3425`, PR #499) の
  **最初の run** で，step 3d (設計判断をユーザに問う枠) を初めて行使した．
  13 行を再検証して stale 0 / changed shape 1 / confirmed 12．
  **自動帯は空** — 13 行すべてが P4 以上かつ全行ゲート付きだったため，
  `AskUserQuestion` の 4 問**すべてを設計判断に充てた**．4 件とも回答を
  得た: **Q1 `cache_mode` はノブごと削除** (D5 全体 + O5(d) を governs)，
  **Q2 torch は CPU extra を必須化** (N4)，**Q3 `preprocess.DataSource` は
  HCPE 専用と明示** (N6-2 + D14(b) を governs)，**Q4 local-cache は dir に
  一本化** (O5(a))．実装まで到達したのは Q2 のみ (P1: `tests/conftest.py`
  から `torch` を外し回帰テスト 3 本を追加，`docs/testing-guide.md` に
  前提を明記) で，**残り 3 件は決定を行に書いて G4 を retire した**
  (D5 / O5 / D14(b) / N6-2 の 4 行 — 人間待ちではなくなったが未実装)．
  D14(b) と N6-2 の **G2 は retire していない** (設計の回答は結合の制約を
  動かさない)．元記録への訂正は**なし**．予算に入らなかった設計判断の
  待ち行列は記録の § "Decisions asked" に順序付きで残してある．
- [2026-08-14 backlog cache-knob-removal](2026-08-14-backlog-cache-knob-removal.md)
  ([PR #501](https://github.com/dousu/maou/pull/501) — **D5 と D10+D11 の
  2 行を削除**)．前 run が得た設計判断のうち**未実装だった 3 件**が
  「人間待ちではないただの作業」として残っていたのが入口で，そのうち
  2 件 (Q1 `cache_mode` ノブ削除 / Q4 local-cache は dir に一本化) を
  実装・出荷した．「決定を行に書けば次 run が通常作業として拾える」が
  実際に機能した最初の run である．12 行を再検証して
  **stale 0 / changed shape 0 / confirmed 12**，行番号のずれも 0
  (前 run が `src/` に触れていないため — 8 run ぶり)．**自動帯は空**
  (9 run 連続)．**D10+D11(1) は B1 の副次効果で構造的に消滅した** —
  結合経路が無くなり `iter_batches()` の yield 数が常に `total_pages()`
  と一致するようになったので，「ファイル数と yield 数のどちらを
  意味させるか」の決めが不要になった (不変条件は回帰テストで固定)．
  O5 は (a)(d) を消化して**残りは (c) だけ**に縮めてある．
  `AskUserQuestion` は受理 1 問 + **設計判断 3 問**で，3 件とも回答を得た:
  **Q2 O5(c) の bundling ノブは削除**，**Q3 O9 は決定的ハッシュ条件へ
  置き換え** (推奨した一時テーブル実体化は却下)，**Q4 Deferred 3 は統合し
  `set_epoch` ガードは付ける側で揃える**．**3 件とも決定を行に書いて G4 を
  retire したが実装は次 run 以降** (O5(c) は受理済み PR に scope を足す
  ことになるため見送り，O9 は G1，Deferred 3 は G2)．元記録への訂正は
  **なし**．
- [2026-08-14 backlog decided-work-second-pass](2026-08-14-backlog-decided-work-second-pass.md)
  (**O5 と Deferred 3 の 2 行を削除**)．前 run が残した「決定済みだが
  未実装」の 3 件のうち **2 件を実装・出荷**した — 決定を行に書けば次 run
  が通常作業として拾える仕組みが **2 run 連続で機能した**．10 行を
  再検証して stale 0 / changed shape 1 / confirmed 9．**自動帯は B1
  (台帳の retrieval bug 修復) の 1 件**で，10 run ぶりに空でなくなった．
  **O5 行はこれで全消化** ((a)(b)(d) は前の 3 run，(c) が本 run)．
  Deferred 3 は G2 (テスト 2 本の識別力喪失) を**取り込んで**解消し，
  型アサーションを挙動アサーションへ書き換えた — 無効化テストにより
  「旧アサーションでは検出できない回帰を新アサーションが捕まえる」
  ことを確認済み．**新規所見 B1 は台帳自身のバグ**: Deferred 3 行と
  O9 行が 4 セル目を持ち，GFM が余剰セルを捨てるため**設計判断の記述が
  描画時に丸ごと消えていた** (G4 を retire する仕組みが，適用された
  4 行のうち 2 行で不可視だった)．同 run 内で修復したので backlog 行は
  起票していない．環境: `uv sync --extra cpu` で torch を導入し
  `app/learning` の QA を実行可能にした (G3 は発生せず)．元記録への訂正は
  **なし**．

- [2026-08-14 backlog stream-wait-and-abc-direction](2026-08-14-backlog-stream-wait-and-abc-direction.md)
  (**Deferred 6 の 1 行を削除**)．**backlog に G4 の行が 1 つも無い状態で
  始まった最初の run** — 過去 3 run の設計判断で 8 行すべての G4 が
  retire され，残りは「決定済み，あとは作業」か「G1/G2/G3 が塞いでいる」
  かのどちらかになった．8 行を再検証して **stale 0 / changed shape 1 /
  confirmed 7**．**自動帯は空** (11 run 連続)．出荷したのは Deferred 6
  (P4，G1 は retire 済み) の 1 件で，`training_loop.py:460` の
  `stream.synchronize()` を `compute_stream.wait_stream(stream)` に置換し，
  `torch.cuda` を差し替えて CPU 上で順序保証を固定する回帰テストを 4 本
  足した (無効化テストで 3 本が落ちることを確認済み)．
  **N6-2 は "changed shape" で，記録の処方そのものが誤りだった** —
  「基底 `iter_batches_df` を HCPE 専用と明記して**改名**する (P6)」と
  あるが，`iter_batches_df` の 4 実装のうち **3 つは汎用**なので，契約名を
  HCPE 名へ改めると汎用の 3 実装と `docs/rust-backend.md` の 5 箇所が
  誤った名前になる．[元記録](2026-08-13-backlog-scan-share-and-abc.md) に
  訂正を追記し，行は 2 案の向きを問う形に書き換えた (2026-08-14 の
  「HCPE 専用と明示する」という設計判断自体は有効なまま)．
  D14(b) は G2 の規模を確定 (テスト **3 ファイル 11 箇所**，記録が挙げて
  いた 2 箇所は `FileDataSource` ではなかった)．新規所見は**なし**．

### Deferred backlog

Findings an audit **confirmed inside** its target path but deliberately
did not fix — ambiguous, cross-layer, architecturally significant, or
needing a decision. A deferred finding is a diagnosis with the fix
withheld pending a decision, **not** a decision never to fix it.

| Found by | Target | Item |
|---|---|---|
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 2 | `src/maou/app/learning` | Stage 1 / Stage 2 pipeline cloned across five files (three of four review angles reported it independently). `run_stage1_with_training_loop` / `run_stage2_with_training_loop` (`multi_stage_training.py:422`/`:571`, ~150 lines each) differ only in head class, callback class, metric getter and two log strings — the loop class is already shared. `_build_stage1_model_and_optimizer` / `_build_stage2_model_and_optimizer` (`stage_component_factory.py:646`/`:735`) have byte-identical 38-line tails. Also `dataset.py:202`/`:279` (file untouched since the record, so those still hold) and `_yield_stage1_batches`/`_yield_stage2_batches` (`streaming_dataset.py:851`/`:911`). **~400-line refactor of the multi-stage training path — architecturally significant.** (Line numbers re-verified 2026-08-09 at `ff5bbaa`.) **2026-08-13 の再検証で 2 点を訂正** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md))．(i) 「head/callback/metric getter とログ 2 本しか違わない」は不完全 — **4 本目の軸**として使う `TrainingLoop` サブクラスが違う (`Stage1TrainingLoop` `:451` vs `RawLogitsTrainingLoop` `:600`)．装飾ではなく挙動の軸なので，「差分は装飾だけ」という前提で統合を設計すると取り落とす．(ii) 工場の「38 行の byte-identical な末尾」は過大で，完全一致は **28 行** (`:705-732`/`:795-822`)．38 行の窓には `stage_name="Stage 1"`/`"Stage 2"` (`:702`/`:792`) が入る．現在の行: run 関数は `:422-568`/`:571-717`，`dataset.py` の対は `:242-316`/`:319-391` (記録の `:202`/`:279` は現在 `_numpy_to_tensor`)，`streaming_dataset.py` は `:848-905`/`:908-962`．統合の footprint は合計 ~600 行．**P4 + G3** (この環境で等価性を示せない) **+ G4**． **2026-08-14 の 2 回目の run にユーザが設計判断を回答: 「統合する — TrainingLoop サブクラスの差も設計に含める」** (`/audit-backlog` step 3d, [記録](2026-08-14-backlog-decided-work-second-pass.md))．run 関数の対・工場の 28 行一致・`dataset.py` の対・`streaming_dataset.py` の対をまとめて統合し，**4 本目の軸である `TrainingLoop` サブクラス (`Stage1TrainingLoop` `:451` vs `RawLogitsTrainingLoop` `:600`) は戦略として注入する**．「重複を受け入れて backlog から落とす」案 (推奨として提示) と「工場の 28 行一致だけ切り出す」案は却下された．**G4 は retire** (決めは済んだ)．**G3 は残る** — ~600 行の統合の等価性をこの環境で示す手段が無いので，出荷にはユーザ側での実地確認が要る．**残る作業**: `multi_stage_training.py` の run 関数 2 本，`stage_component_factory.py` の完全一致 28 行，`dataset.py`，`streaming_dataset.py` の 4 組． **2026-08-14 の 3 巡目で行番号を更新** (`aad00d9` のアダプタ統合でファイルが動いたため): run 関数は **`:376-524`/`:525-673`** (記録の `:422-568`/`:571-717`)，`TrainingLoop` サブクラスの分岐は **`:428` `Stage1TrainingLoop` / `:577` `RawLogitsTrainingLoop`** (`:451`/`:600`)，工場の完全一致 28 行は **`:704-731`/`:794-821`** (`:705-732`/`:795-822` — 全体が 1 行だけ上へずれており，`diff` で一致を再確認済み)．`dataset.py` と `streaming_dataset.py` の対は未再測．**P4** は不変．この run では着手していない (受理済み PR #502 に後から scope を足すことになるため見送った)． |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 5 | `src/maou/app/learning` | `training_loop.py:1110` per-batch host-device sync — `if not has_legal.all():` calls `Tensor.__bool__` on a CUDA tensor, a full pipeline stall once per batch, to guard a warning. **Now dormant, not fixed** (2026-08-09, backlog `tier-3-contracts`): the record's premise "Stage 3 always ships a `legal_move_mask`, so the branch is always taken" no longer holds — no data path supplies a mask, so `_compute_policy_loss` never enters the masking arm and the sync does not execute. The code is unchanged and the stall returns the moment a real legal-move mask is wired in; fix it **then**, together with whatever produces the mask, and measure on GPU. **2026-08-13 に dormant の証明を強化** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md)): `TrainingContext` の構築箇所は `training_loop.py:516` の 1 つだけで，その 3 行上 `:507` に `legal_move_mask: torch.Tensor | None = None` と**ハードコードされている**．`src/` 全体に産出者はゼロ (`stage2_data_generation.py:247` の Rust `legal_move_masks()` はデータ生成用で `TrainingContext` に届かない)．`_unpack_batch` の docstring `:497-502` も「将来経路のために残している」と明言．**P4 + G1**． **2026-08-14 の 2 回目の run にユーザが「GPU で測れなくても意味論的に等価な変更は出荷してよい」と回答** (`/audit-backlog` step 3d, [記録](2026-08-14-backlog-decided-work-second-pass.md), Deferred 5/6/7 を 1 問で governs)．**ただし本行の G1 は retire されない** — この修正は意味論的に自明な置換ではなく，産出者がゼロで休眠中という前提の上に立つ性能改善なので，**legal_move_mask を実際に配線する変更と同時に，GPU 上で測って直す**という方針は不変．回答は Deferred 6 だけを人間待ちから外した．**P4 + G1** は不変． |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 7 | `src/maou/app/learning` | `gradient_noise_scale.py:150,189-192,247` — one GPU sync per parameter tensor per micro-batch (`.item()` inside `for param in model.parameters()`): 60-300 syncs per micro-batch on a ResNet/ViT backbone whenever adaptive batch is on. Accumulating into a device scalar changes when the value materializes, and GNS feeds the adaptive batch controller — needs a numerical equivalence check. **2026-08-13 に再確認** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md)): 3 箇所とも健在 (`:150` / `:188-192` / `:246-248`)．加えて `g.clone()` (`:151`/`:193`) が勾配の全複製をパラメータ毎に確保する．緩和策の `should_measure` (`:115-122`) はあるが **`measurement_interval` の既定は `1`** (`:85`) なので，caller が上書きしない限り毎 optimizer step で走る．値は `compute()` → `b_noise` → `training_loop.py:1024-1031` で Python float として消費され `gradient_accumulation_steps` を書き換えるため，controller 側を tensor 値に作り替えないと同期は遅延できない．**P4 + G1**． **2026-08-14 の 2 回目の run にユーザが「GPU で測れなくても意味論的に等価な変更は出荷してよい」と回答** (`/audit-backlog` step 3d, [記録](2026-08-14-backlog-decided-work-second-pass.md), Deferred 5/6/7 を 1 問で governs)．**ただし本行の G1 は retire されない** — device スカラーへの蓄積は値が materialize するタイミングを変え，その値が `gradient_accumulation_steps` を書き換えるので，**数値等価性の確認が必要**であり「意味論的に自明な等価変換」には当たらない．**P4 + G1** は不変． |

### Out-of-scope backlog

Findings an audit surfaced **outside** the path it was auditing, and was
therefore not allowed to fix.

| Found by | Target | Item |
|---|---|---|
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D13 | `src/maou/infra/file_system` | **(a) の contained な部分と (c) は 2026-08-12 に消化済み** (`/audit-backlog`, PR #456)．探索対象を `__init__` へ巻き上げ `bisect.bisect_right` に置き換え，`_STRUCTURED_DTYPES` は `get_dtype` へ畳んだ．**残り**: (a) の**根本**部分 — `__getitem__` は訓練サンプル1件ごとに `np.empty(1, dtype)` (preprocessing で 3,089B) を確保して6フィールドを memcpy する．batch 1024 で約3MB の memcpy + 1024回の小確保/バッチ．`dataset.py:230` の `torch.from_numpy` は「ゼロコピー」と称しているが，コピーは1層下で既に起きている．根本解決 (`KifDataset` が `ColumnarBatch` を直接スライスする) は `app/learning/dataset.py` と ABC を触るので path 外．(b) `get_items` は「バッチで取得」と謳いながら `get_item` を要素ごとに Python ループで呼ぶだけで何もバッチ化していない．`ColumnarBatch.slice` によるベクトル化が可能 — (a) の根本解決と同じ方向なので一緒に扱うべき．**2026-08-13 の再検証**: `FileDataSource.get_items` の呼び出し側は**ゼロ** (`FileManager.get_items` への内部委譲だけ) なので (b) 単独では **dormant**．実害は (a) の根本解決に踏み込んだときに初めて出るため，「(b) を先に直す」選択肢は実質無い．**記録の見立ての誤り**: (c) を消せば `assert self._structured_dtype is not None` が不要になると書かれているが，`_use_columnar` が False (hcpe) の経路では属性は依然 None なので型は Optional のままで，assert は落とせない (PR #456 後も `:613`/`:733` に残っている)． **2026-08-13 の再検証で数値を訂正** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md)): 機構は健在だが記録の数字が古い．1 サンプルは **3,089B ではなく 9,081B** (`get_preprocessing_dtype` に `moveWinRate` (f32×1496 = 5,984B) が入ったため)，コピーは **6 フィールドではなく 8 書き込み** (`:586-587` の `id` ゼロ埋め含む)．`np.empty(n, dtype)` は `:584`，`torch.from_numpy` は `dataset.py:230` ではなく **`:239`** (「ゼロコピー」を謳うコメントは `:95`)．batch 1024 なら約 **9.3MB** の memcpy/バッチ．hcpe 経路は view を返すので影響なし — 割り当ては columnar 型に限る．(b) の caller ゼロも再確認 (`tests/` にもヒット無し)．**P4/P6 + G2 + G4**． **2026-08-14 の再検証で機構の所在を訂正** ([記録](2026-08-14-backlog-diagnostics-and-npy-remnants.md))．`__getitem__` 自身はもう確保をしていない — `:692-696` の薄い委譲 (`return self.__file_manager.get_item(...)`) になり，実体は `_columnar_batch_to_structured_array` (**`:553-634`**) へ移った (`get_item` → `_columnar_to_structured_record` `:471-494` 経由)．`np.empty(n, dtype=self._structured_dtype)` は `:584` で行番号は一致するが，**フィールド書き込みは `:586-632`** に広がっており，前の記述の `:586-587` は狭すぎる (8 書き込みという数は正しい)．`get_items` (外側 `:701-712`) が `FileManager.get_items` (`:496-509`) へ委譲し，そこが `[self.get_item(idx) for idx in indices]` で回しているのも不変．結論は不変． **2026-08-14 の 2 回目の run の再検証で行番号を更新**: 前 run の `bdda7b5` がファイルを縮めたため `_columnar_batch_to_structured_array` は **`:408-`** (記録の `:553-634`)，`np.empty` は **`:439`** (`:584`)，`_columnar_to_structured_record` は **`:338-359`** (`:471-494`)，`FileManager.get_items` は **`:363-`** (`:496-509`)，`FileDataSource.__getitem__`/`get_items` は **`:310-`/`:550-`** (`:692-696`/`:701-712`)．**2026-08-14 の 2 回目の run にユーザが設計判断を回答: 「`ColumnarBatch` を直接スライスする」** (`/audit-backlog` step 3d, [記録](2026-08-14-backlog-decided-work-second-pass.md))．**1 つの回答が (a) の根本と (b) の両方を governs する** — `KifDataset` が `ColumnarBatch` を直接スライスするようにし，サンプル毎の `np.empty(9,081B)` + 8 フィールド memcpy (batch 1024 で約 9.3MB/バッチ) を無くす．「現状維持」「`get_items` だけベクトル化」「先に計測してから決める」は却下された．**G4 は retire** (決めは済んだ)．**G2 は残る** — `app/learning/dataset.py` と ABC に触るので `infra/file_system` の path 外に出る．**残る作業**: `KifDataset` 側のスライス経路の新設，`FileManager.get_item`/`get_items` の縮退，hcpe 経路 (view を返すので影響なし) との分岐整理．`assert self._structured_dtype is not None` は `_use_columnar` が False の経路が残る限り落とせない点も不変．**P4/P6** は不変． **2026-08-14 の 3 巡目で行番号を再確認**: `_columnar_to_structured_record` **`:338`**，`FileManager.get_items` **`:363`**，`_columnar_batch_to_structured_array` **`:408`**，`np.empty(n, dtype=self._structured_dtype)` **`:439`**，`FileDataSource.__getitem__` **`:541`**，`FileDataSource.get_items` **`:550`** — いずれも前 run の記述どおりで移動なし．結論は不変．この run では着手していない (**設計は決定済みで人間待ちではなく**，G2 の作業量が今回の枠に入らなかっただけ — 次 run の先頭候補)． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D14 (残り) | `src/maou/infra/file_system` | **(a) 行数スキャンの共有化は 2026-08-13 に出荷済み** ([PR #492](https://github.com/dousu/maou/pull/492), [記録](2026-08-13-backlog-scan-share-and-abc.md))．`domain/data/arrow_format.scan_row_counts` を 2 実装が引くようになり，`StreamingHcpeDataSource` は per-file カウントを `row_counts` で公開する．**残り**: **(b) `FileDataSource` が2つの ABC を着ている．** `preprocess.DataSource` 側の役割は `StreamingHcpeDataSource` に移った (`console/pre_process.py:489-492`) のに継承が残り，`hcpe` を `FileManager` の columnar 機構に通すための `_use_columnar` 分岐 (`:376`, `:605`, `:729`) を生かし続けている．**2026-08-13 の再検証で「path 外の障害」の記述を撤回** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md))．`benchmark_polars_io.py:419-451` は `print_summary` の docstring と print 行で `FileDataSource` とは無関係．実使用は `:386-390` (構築) と `:394` (`iter_batches_df`) だが，**`iter_batches_df` は `FileDataSource` 自身の具象メソッド** (`file_data_source.py:726`) なので ABC を外しても壊れない．**真の障害はテスト群** — `test_hcpe_transform.py:79`/`:92`/`:305`/`:331`，`test_app_hcpe_transform.py:185-190`/`:259-264`，`test_convert_and_preprocess.py:226-231`/`:328`/`:430`/`:534` が `FileDataSource` を `PreProcess(datasource:)` に渡している．また `_use_columnar` の現在位置は `:165`,`:171`,`:182`,`:236`,`:241`,`:305`,`:456`,`:539` で，**preprocess の役割に属するのは `:539` と `:726` だけ**．`:241`/`:456` は `learn.LearningDataSource` 側を支えるので**外しても退役しない** — 「継承を外せば分岐がまとめて消える」という見積りは過大．**P6 + G2 + G4**． **2026-08-14 にユーザが設計判断を回答: 「HCPE 専用と明示する」** (`/audit-backlog` step 3d, [記録](2026-08-14-backlog-design-decisions.md))．`preprocess.DataSource` ABC と基底 `iter_batches_df` を HCPE 専用と明記し (改名を含む)，`FileDataSource` からは `preprocess.DataSource` の継承を外す．**G4 は retire** (決めは済んだ)．**G2 は残る** — テスト 3 ファイル (`test_hcpe_transform.py`, `test_app_hcpe_transform.py`, `test_convert_and_preprocess.py`) が `FileDataSource` を `PreProcess(datasource:)` に渡しているので，継承を外すにはそれらを `StreamingHcpeDataSource` へ寄せる作業とセットになる．**P6** は不変．この run では着手していない (予算外)． **2026-08-14 の再検証で 2 点更新**: `_use_columnar` の 8 箇所の役割分担がより細かく判った — preprocess の役割に属するのは **`:539` だけ** (`iter_batches`)，`:456` は `get_item` 経由で learn 側，残る 6 箇所 (`:165`,`:171`,`:182`,`:236`,`:241`,`:305`) は `FileManager` 内部の**両者共用**．「継承を外せば分岐がまとめて消える」という見積りが過大である点は，前 run の指摘より更に強く成り立つ．また `benchmark_polars_io.py` の正確な位置は **`src/maou/infra/utility/benchmark_polars_io.py`** (`scripts/` ではなく `src/` 配下なので出荷物であり，触るなら version bump の対象)． **2026-08-14 の 3 巡目の再検証で G2 の規模を確定** ([記録](2026-08-14-backlog-stream-wait-and-abc-direction.md)): `FileDataSource` を `PreProcess(datasource=)` に渡しているのは **3 ファイル 11 箇所** — `test_hcpe_transform.py:92,146,177,210,280,305,331` (7)，`test_app_hcpe_transform.py:190,264` (2)，`test_convert_and_preprocess.py:231,328` (2)．前 run の記述が挙げていた `test_convert_and_preprocess.py:430,534` は **`BigQueryDataSource` / `S3DataSource`** であって `FileDataSource` ではないので数に入らない．また各テストの `local_datasource.iter_batches()` は `FileDataSource` の**具象メソッド** (`:563`) を呼ぶだけなので，継承を外しても**実行時には壊れない** — 破れるのは `datasource=` の型 (mypy) の側だけである．`_use_columnar` の現在位置は `:137,143,154,208,213,323,394` の **7 箇所** (前 run の 8 箇所から移動・減少)． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O9 | `src/maou/infra/bigquery` | `bq_data_source.py:212-234` (記録の `:222-243` から移動) — `sample_ratio` 指定時，`total_rows` は `TABLESAMPLE` 付き `COUNT(*)` から得るが `__fetch_from_bigquery`(**`:395-411`**) はページごとに**別々に引き直す** `TABLESAMPLE` を発行する．ファイル系ソースは常に厳密．`:226` の `num_rows` はテーブルメタデータでストリーミング挿入に遅れる． **2026-08-13 の再検証で記述を強めた** ([記録](2026-08-13-backlog-callback-accumulator-table.md)): 非決定性は**二重**である．(i) `TABLESAMPLE SYSTEM` はクエリごとに独立評価なので数えた行集合と返す行集合が別物，(ii) その上で `LIMIT/OFFSET` を **`ORDER BY` 無しの再サンプル結果**に掛けているので，**同じ `page_num` を 2 回引いても同じ行が返る保証が無い**．「件数がずれる」ではなく**再現性が無い**． **修正の向きが 3 つに割れている** (G4 相当): (a) サンプルを一時テーブルへ 1 度実体化してページングする，(b) `FARM_FINGERPRINT` 等の決定的ハッシュ条件へ置き換える (キー列の決めが要る)，(c) `sample_ratio` とページングの併用を拒否する．**G1**: BigQuery がこの環境に無く，実際の非決定性は fake client では再現できない． **2026-08-13 に再確認** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md)): 行が移動して `__get_total_rows` は `:213-234` (サンプル分岐 `:215-223`)，`total_pages` の導出は `:202-204`，ページ側の再サンプルは `__fetch_from_bigquery` (def `:345`) の `:396-405` (`start_index = page_num * batch_size` `:398`)．クラスタ/パーティション経路も `tablesample_clause` `:380-383` を `:385-389` に差し込むので同じ欠陥を弱い形で持つ．非サンプル経路は `list_rows(start_index=…)` (`:415-419`) なので無傷．**緩和材料**: `get_page` (`:486-520`) がページをキャッシュするため 1 run 内では通常 1 回しか引かれず，非再現性は退避時と run 跨ぎで表面化する． **2026-08-14 にユーザが設計判断を回答: 「決定的ハッシュ条件に置き換える」** (`/audit-backlog` step 3d, [記録](2026-08-14-backlog-cache-knob-removal.md))．3 つの向きのうち **(b)** が選ばれた — `TABLESAMPLE SYSTEM` をやめ，`MOD(FARM_FINGERPRINT(<key>), N) < k` 等の**行に対して決定的な条件**へ置き換える．推奨として提示した (a) 一時テーブルへの実体化は却下された (状態を持たずに済む方を選好)．**G4 は retire** (決めは済んだ)．**G1 は残る** — BigQuery がこの環境に無く，実際の再現性は fake client では確認できない．**残る作業と未解決点**: (i) **キー列の決め**がこの方式の要 — 分布が偏るとサンプルも偏るので，`id` 系の高カーディナリティ列を選ぶ必要がある (現行スキーマに一意キーの保証があるかは未調査)．(ii) `__get_total_rows` (`:213-234`) の COUNT とページ側 (`__fetch_from_bigquery` `:396-405`) が**同じ条件式**を使うようにする — 現在は各々が独立に TABLESAMPLE を発行しているのが二重の非決定性の源．(iii) クラスタ/パーティション経路 (`tablesample_clause` `:380-389`) も同じ差し替えが要る．(iv) `LIMIT/OFFSET` に `ORDER BY` が無い点は条件を決定的にしても残るので，安定な並び順を与える必要がある．**P4** は不変 (同じ呼び出しで返る行が変わる)．この run では着手していない (予算外 + G1)． |
| [2026-08-13 backlog scan-share-and-abc](2026-08-13-backlog-scan-share-and-abc.md) N6-2 | `src/maou/app/pre_process` | **基底の具象 `iter_batches_df` が HCPE スキーマ決め打ち．** `hcpe_transform.py:86-139` の既定実装は `get_hcpe_polars_schema()` を直に引くので，`preprocessing` 型のソースに対しては黙って誤動作する．現状 production の caller はすべて override 側 (`FileDataSource` / `ObjectStorageDataSource`) を通るので **dormant**．「HCPE 専用と明記して名前を変える」か「array_type で分岐させる」かの判断が要る．**2026-08-13 の再検証で dormant の度合いが増した**: N6-1 の修正で `BigQueryDataSource` も `iter_batches_df` を override したため，基底の既定実装を**呼ぶ** production 経路はゼロになった．**ただし同日 3 本目の run で上の記述を訂正**: ゼロなのは*呼び出し*であって*継承*ではない．`StreamingHcpeDataSource` (`console/pre_process.py:494` が構築する production の pre-process ソース) は `preprocess.DataSource` を継承しつつ `iter_batches_df` を override していないので，基底の既定実装を**着ている**．しかもそれは hcpe 専用クラスなので，**HCPE 決め打ちの既定実装はそこでは正しい**．この訂正は判断に効く — 「HCPE 専用と明記する」方向の根拠が強まり，「array_type で分岐させる」方向は分岐を要する継承者が居ない分だけ弱まる．ただし前者は行の文言上**改名 (P6) を含む**ので 2 案はまだ 1 案に潰れていない (G4 継続)． **2026-08-13 に再確認，行範囲を更新** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md)): 既定実装は現在 **`:95-148`** (記録の `:86-139` から移動)，`get_hcpe_polars_schema()` を引くのは `:114-118`，`pl.DataFrame(data, schema=schema)` は `:147`．override 状況も再確認 — `BigQueryDataSource` (`:740`，docstring `:743-756` が本行を明示的に参照)，`ObjectStorageDataSource` (`:475`)，`FileDataSource` (`:726`) は override 済み，`StreamingHcpeDataSource` (`streaming_hcpe_source.py:25`) のみ override せず既定を着ている (hcpe 専用クラスなので正しい)．この配置は `tests/maou/app/pre_process/test_datasource_abc.py:75-100` が固定している (`iter_batches_df` を敢えて非 abstract に保ち `StreamingHcpeDataSource` を構築可能にする)． **2026-08-14 にユーザが設計判断を回答: 「HCPE 専用と明示する」** (`/audit-backlog` step 3d, [記録](2026-08-14-backlog-design-decisions.md))．D14(b) 行と**同じ 1 つの回答が両方を governs** する — 基底 `iter_batches_df` を `array_type` で分岐させる案は却下され，HCPE 専用と明記する (改名を含む) 側に決まった．再検証が積み上げてきた「override せず基底を着ているのは HCPE 専用の `StreamingHcpeDataSource` だけ」という事実がこの向きを支持していた．**G4 は retire** (決めは済んだ)．**2026-08-14 の 3 巡目の再検証で，行が書いていた処方「改名 (P6)」は**誤り**と判明** ([記録](2026-08-14-backlog-stream-wait-and-abc-direction.md))．`iter_batches_df` の 4 実装のうち **3 つは汎用**である — `FileDataSource` (`file_data_source.py:575`) は `array_type` で hcpe/preprocessing/stage1/stage2 を分岐し，`ObjectStorageDataSource` (`data_source.py:480`) は hcpe/preprocessing でスキーマを切り替え，`BigQueryDataSource` (`bq_data_source.py:737`) は DataFrame をそのまま返す．HCPE 決め打ちなのは**基底の既定実装だけ** (`hcpe_transform.py:95-148`，スキーマ取得は `:114-118`)．よって ABC のメソッド名を HCPE 名へ改めると，汎用の 3 実装と `docs/rust-backend.md:680,701,714,725,730` が誤った名前になる．**改名すべきは*契約*ではなく*既定実装*だった．** 2026-08-14 の設計判断「HCPE 専用と明示する」は有効なまま，その実装の向きが 2 案に割れる: **(A)** 基底 `iter_batches_df` を abstract にし，HCPE 決め打ちの本体を唯一の継承者である `StreamingHcpeDataSource` へ移す (**P6** — 外部の継承者が壊れる代わりに，非 HCPE の実装が黙って誤動作する余地が構造的に消える)．**(B)** 基底は concrete のまま，本体を HCPE と名の付くヘルパへ切り出して docstring に「HCPE 専用」と明記する (**P3** — 契約も継承者も無傷で最も安いが，override を忘れた非 HCPE の継承者は依然黙って誤動作する)．**残る作業**: 向きの決定と，`tests/maou/app/pre_process/test_datasource_abc.py:77-100` の更新 ((A) なら不変条件を反転させる)．この run では着手していない． |
