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

- [2026-08-15 backlog columnar-uninitialized-fields](2026-08-15-backlog-columnar-uninitialized-fields.md)
  (**backlog 行の削除は無し — 5 行すべてが残る**)．先頭候補だった D13 の
  再検証中に**新規所見 N10** を発見し，そちらを出荷した run である．
  `_columnar_batch_to_structured_array` は `np.empty` で確保して
  フィールドごとに**条件付き代入**していたため，dtype に有って
  `ColumnarBatch` に無い列 (preprocessing dtype は `moveWinRate` を
  無条件に含むが，`moveWinRate` 列を持たない旧 `.feather` では
  `None`) が**未初期化メモリのまま**返り，`KifDataset` が列の有無を
  dtype 名から判定するためその NaN 混じりのメモリを**訓練ターゲット
  として** torch に渡していた．写しを dtype 駆動のループに一本化し，
  batch が供給しない列はゼロで埋めた (`033d49f`)．
  5 行を再検証して **stale 0 / changed shape 2 / confirmed 3**．
  **自動帯は空** (13 run 連続)．
  **D13 は "changed shape" — 決定済みの設計が記録の処方のままでは
  実装できない**と判明した．`_explode_list_column` は polars 側の
  dtype が目標と一致すると `astype` を挟まず `to_numpy()` の
  **read-only な Arrow ビュー**を返すので，preprocessing の
  `boardIdPositions` / `piecesInHand` / `moveWinRate` は
  `writeable=False` になり，read-only を設計として撥ねる
  `KifDataset._numpy_to_tensor` に直接渡すと落ちる．**合成テストは
  `astype` が挟まって全部通る**ので，この罠は本番データでしか出ない．
  writeability をどこで確立するかの設計判断 (実質的な G4 の再発) を
  Q2 で問い，**ユーザは (a) `domain` の `_explode_list_column` で
  保証する**を選択した (最上流で一度だけ保証すれば全経路が自動的に
  正しくなる)．**決定を行に書いて G4 を retire した**が実装は次 run
  以降 (G2 が残るため) — **次 run の先頭候補**．
  **Deferred 2 も "changed shape"** — `Stage1TrainingLoop` は
  `RawLogitsTrainingLoop` の**別名** (2026-08-09 `568863f` から) で，
  2026-08-13 の記録が書いた訂正 (i)「4 本目の挙動の軸」は**書かれた
  時点で既に誤り**だった．[元記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md)
  に訂正を追記した．2026-08-14 の設計判断「`TrainingLoop` サブクラスは
  戦略として注入する」は**存在しない差異のための設計**になっている
  (統合するという決定自体は有効)．
  **O9 のテスト網羅の実態を確定**: `sample_ratio` / `TABLESAMPLE` に
  触れるテストは `tests/` に**ゼロ**で，"fake" と呼ばれる 2 本は
  BigQuery クライアントを fake しておらず `pm.get_page` を lambda に
  差し替えて**対象経路を丸ごと潰している**．新規の out-of-scope 行は
  **なし**．

- [2026-08-15 backlog writeable-contract-and-decisions](2026-08-15-backlog-writeable-contract-and-decisions.md)
  ([PR #505](https://github.com/dousu/maou/pull/505) — **backlog 行の削除は
  無し．5 行すべてが残る**)．前 run が「次 run の先頭候補」と名指しした
  **D13 の残る作業 (1) を出荷**し，**残る 4 行のうち 3 行の設計判断を得た**
  run である．5 行を再検証して **stale 0 / changed shape 1 / confirmed 4**
  (D13 のみ — `file_data_source.py` の行が前 run の `033d49f` で +15〜+20)．
  **自動帯は空** (14 run 連続)．出荷したのは B-1 (P4，ゲート無し) の 1 件で，
  `_explode_list_column` の fast path が `astype` を挟まなかった場合だけ
  コピーを取るようにし，「C-contiguous かつ writeable」を契約として
  docstring に明記した (`0b3f1ab`)．**回帰テスト 10 本は polars 側 dtype を
  目標 dtype と明示的に一致させて書いてあり** (list から組むと `astype` が
  挟まって空虚になる)，無効化テストで **10 本中 7 本が落ちる**ことを確認済み．
  **backlog は 5 行で始まり 5 行で終わったが，動いたのは行数ではなくゲートの
  数である** — `AskUserQuestion` は受理 1 問 + **設計判断 3 問**で，3 件とも
  回答を得た: **Q2 Deferred 2 の G3 は retire** (挙動不変を意図した純粋な
  refactor なので，2026-08-14 に Deferred 5/6/7 へ与えた原則を適用．**これで
  Deferred 2 のゲートは無くなり次 run の先頭候補になった**)，**Q3 Deferred 7 は
  `measurement_interval` の既定を上げる**，**Q4 O9 は fake BigQuery client の
  テスト土台を新設して同梱** (**G1 が「出荷前の実地確認」へ縮小**した — 決定的
  ハッシュ条件の要点である再現性は fake で CI に載せられる)．**3 件とも決定を
  行に書いたが実装は次 run 以降**．元記録への訂正は**なし**だが，**本 run 自身が
  Q3 でユーザへ述べた scope の説明が誤っていた** — 「1 行で contained」と述べた
  ものの，`measurement_interval` の既定は 3 箇所にあり，本番経路を決めているのは
  クラス既定ではなく **CLI 既定** (`learn_model.py:217`) で，CLI は `:984` で
  常に明示的に渡すため (i)(ii) だけ変えても挙動は変わらない．訂正を行に書いた
  ので次 run は正しい scope から始められる．環境: `uv sync --extra cpu` で
  消費側 suite も回して **862 passed** (G3 は発生せず)．

### Deferred backlog

Findings an audit **confirmed inside** its target path but deliberately
did not fix — ambiguous, cross-layer, architecturally significant, or
needing a decision. A deferred finding is a diagnosis with the fix
withheld pending a decision, **not** a decision never to fix it.

| Found by | Target | Item |
|---|---|---|
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 2 | `src/maou/app/learning` | Stage 1 / Stage 2 pipeline cloned across five files (three of four review angles reported it independently). `run_stage1_with_training_loop` / `run_stage2_with_training_loop` (`multi_stage_training.py:422`/`:571`, ~150 lines each) differ only in head class, callback class, metric getter and two log strings — the loop class is already shared. `_build_stage1_model_and_optimizer` / `_build_stage2_model_and_optimizer` (`stage_component_factory.py:646`/`:735`) have byte-identical 38-line tails. Also `dataset.py:202`/`:279` (file untouched since the record, so those still hold) and `_yield_stage1_batches`/`_yield_stage2_batches` (`streaming_dataset.py:851`/`:911`). **~400-line refactor of the multi-stage training path — architecturally significant.** (Line numbers re-verified 2026-08-09 at `ff5bbaa`.) **2026-08-13 の再検証で 2 点を訂正** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md))．(i) 「head/callback/metric getter とログ 2 本しか違わない」は不完全 — **4 本目の軸**として使う `TrainingLoop` サブクラスが違う (`Stage1TrainingLoop` `:451` vs `RawLogitsTrainingLoop` `:600`)．装飾ではなく挙動の軸なので，「差分は装飾だけ」という前提で統合を設計すると取り落とす．(ii) 工場の「38 行の byte-identical な末尾」は過大で，完全一致は **28 行** (`:705-732`/`:795-822`)．38 行の窓には `stage_name="Stage 1"`/`"Stage 2"` (`:702`/`:792`) が入る．現在の行: run 関数は `:422-568`/`:571-717`，`dataset.py` の対は `:242-316`/`:319-391` (記録の `:202`/`:279` は現在 `_numpy_to_tensor`)，`streaming_dataset.py` は `:848-905`/`:908-962`．統合の footprint は合計 ~600 行．**P4 + G3** (この環境で等価性を示せない) **+ G4**． **2026-08-14 の 2 回目の run にユーザが設計判断を回答: 「統合する — TrainingLoop サブクラスの差も設計に含める」** (`/audit-backlog` step 3d, [記録](2026-08-14-backlog-decided-work-second-pass.md))．run 関数の対・工場の 28 行一致・`dataset.py` の対・`streaming_dataset.py` の対をまとめて統合し，**4 本目の軸である `TrainingLoop` サブクラス (`Stage1TrainingLoop` `:451` vs `RawLogitsTrainingLoop` `:600`) は戦略として注入する**．「重複を受け入れて backlog から落とす」案 (推奨として提示) と「工場の 28 行一致だけ切り出す」案は却下された．**G4 は retire** (決めは済んだ)．**G3 は残る** — ~600 行の統合の等価性をこの環境で示す手段が無いので，出荷にはユーザ側での実地確認が要る．**残る作業**: `multi_stage_training.py` の run 関数 2 本，`stage_component_factory.py` の完全一致 28 行，`dataset.py`，`streaming_dataset.py` の 4 組． **2026-08-14 の 3 巡目で行番号を更新** (`aad00d9` のアダプタ統合でファイルが動いたため): run 関数は **`:376-524`/`:525-673`** (記録の `:422-568`/`:571-717`)，`TrainingLoop` サブクラスの分岐は **`:428` `Stage1TrainingLoop` / `:577` `RawLogitsTrainingLoop`** (`:451`/`:600`)，工場の完全一致 28 行は **`:704-731`/`:794-821`** (`:705-732`/`:795-822` — 全体が 1 行だけ上へずれており，`diff` で一致を再確認済み)．`dataset.py` と `streaming_dataset.py` の対は未再測．**P4** は不変．この run では着手していない (受理済み PR #502 に後から scope を足すことになるため見送った)． **2026-08-15 の再検証で「4 本目の軸」が誤りと判明 — 決定の前提が崩れている** ([記録](2026-08-15-backlog-columnar-uninitialized-fields.md))．`training_loop.py:1183` は `Stage1TrainingLoop = RawLogitsTrainingLoop` の**モジュール level の別名**であり，`git log -S` によれば **2026-08-09 (`568863f` "全 1 の legal_move_mask を targets タプルから外す") から**そうである．したがって `multi_stage_training.py:428` と `:577` は**同じクラスを構築している**．2026-08-13 の記録が書いた訂正 (i)「4 本目の軸として使う `TrainingLoop` サブクラスが違う，装飾ではなく挙動の軸」は**書かれた時点で既に誤り**で ([元記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md) に訂正を追記した)，それを根拠に 2026-08-14 にユーザが下した「**`TrainingLoop` サブクラスは戦略として注入する**」という決定は，**存在しない差異のための設計**になっている．**「統合する」という決定自体は有効**だが，注入機構は不要なので統合はその分素直になる (差分は head クラス / callback クラス / metric getter / ログ 2 本のみ)．現在の行: run 関数は **`:376-523`/`:525-672`**，工場の完全一致は **`:704-731` ≡ `:794-821`** (`diff` で再確認，1 行広げると `stage_name="Stage 1"`/`"Stage 2"` で崩れる)，`dataset.py` の対は **`Stage1Dataset` `:242-316` / `Stage2Dataset` `:319-391`**，`streaming_dataset.py` の対は **`:834-891`/`:894-948`** (記録の `:848-905`/`:908-962` から **−14**)．footprint 合計 **~585 行**．**P4 + G3** は不変 (~600 行の等価性をこの環境で示す手段が無い)． **2026-08-15 の 2 回目の run にユーザが G3 の扱いを回答: 「G3 を retire する」** (`/audit-backlog` step 3d, [記録](2026-08-15-backlog-writeable-contract-and-decisions.md))．2026-08-14 に Deferred 5/6/7 へ与えた「GPU で測れなくても意味論的に等価な変更は出荷してよい」と同じ原則を，挙動不変を意図した純粋な refactor である本行にも適用する．却下されたのは「4 組を 1 PR ずつに分割して出荷」(run 数が 4 倍になる) と「G3 を残す」(実 GPU 環境まで待つ)．**G3 は retire** — 回帰は既存 suite (`tests/maou/app/learning/` + `tests/maou/infra/file_system/` + `tests/maou/domain/data/` で 862 passed) が担い，~585 行ぶんの最終的な実地確認はユーザ側で行う前提．**これで本行に残るゲートは無くなった — 決定済みの通常作業であり，次 run の先頭候補である**．**残る作業** (2026-08-15 に再確認した現在行): `multi_stage_training.py` の run 関数 2 本 **`:376-523`/`:525-672`**，`stage_component_factory.py` の完全一致 28 行 **`:704-731` ≡ `:794-821`** (`diff` で再確認済み)，`dataset.py` の **`Stage1Dataset` `:242-316` / `Stage2Dataset` `:319-391`**，`streaming_dataset.py` の **`:834-891`/`:894-948`** の 4 組．**`TrainingLoop` サブクラスの注入機構は不要** (`training_loop.py:1183` が `Stage1TrainingLoop = RawLogitsTrainingLoop` の別名なので差異が存在しない — 2026-08-15 に確定済み)．差分は head クラス / callback クラス / metric getter / ログ 2 本のみ．**P4** は不変． |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 5 | `src/maou/app/learning` | `training_loop.py:1110` per-batch host-device sync — `if not has_legal.all():` calls `Tensor.__bool__` on a CUDA tensor, a full pipeline stall once per batch, to guard a warning. **Now dormant, not fixed** (2026-08-09, backlog `tier-3-contracts`): the record's premise "Stage 3 always ships a `legal_move_mask`, so the branch is always taken" no longer holds — no data path supplies a mask, so `_compute_policy_loss` never enters the masking arm and the sync does not execute. The code is unchanged and the stall returns the moment a real legal-move mask is wired in; fix it **then**, together with whatever produces the mask, and measure on GPU. **2026-08-13 に dormant の証明を強化** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md)): `TrainingContext` の構築箇所は `training_loop.py:516` の 1 つだけで，その 3 行上 `:507` に `legal_move_mask: torch.Tensor \| None = None` と**ハードコードされている**．`src/` 全体に産出者はゼロ (`stage2_data_generation.py:247` の Rust `legal_move_masks()` はデータ生成用で `TrainingContext` に届かない)．`_unpack_batch` の docstring `:497-502` も「将来経路のために残している」と明言．**P4 + G1**． **2026-08-14 の 2 回目の run にユーザが「GPU で測れなくても意味論的に等価な変更は出荷してよい」と回答** (`/audit-backlog` step 3d, [記録](2026-08-14-backlog-decided-work-second-pass.md), Deferred 5/6/7 を 1 問で governs)．**ただし本行の G1 は retire されない** — この修正は意味論的に自明な置換ではなく，産出者がゼロで休眠中という前提の上に立つ性能改善なので，**legal_move_mask を実際に配線する変更と同時に，GPU 上で測って直す**という方針は不変．回答は Deferred 6 だけを人間待ちから外した．**P4 + G1** は不変． **2026-08-15 の再検証で行番号を更新** (前 run の `wait_stream` 化でファイルが伸びたため，全体 **+7**)．同期は **`:1117`** (`if not has_legal.all():`，記録の `:1110`)，`has_legal` の生成は `:1116`，**2 つ目の同期 `int((~has_legal).sum().item())` が `:1122`** にある (前の記述はこれを数えていなかった)．`_compute_policy_loss` の def は `:1105`，ハードコードの `legal_move_mask: torch.Tensor \| None = None` は **`:514`** (`:507`)，唯一の `TrainingContext(...)` 構築は **`:523`** (`:516`)．**産出者ゼロを再確認** — `src/` の全ヒットは宣言・読み出し・不在の明記のみで，非 None を与えるのは `tests/maou/app/learning/test_callbacks.py` の 3 箇所だけ．**休眠の根拠が 1 つ強くなった**: `RawLogitsTrainingLoop._compute_policy_loss` (`:1170`，クラスは `:1160`) はマスク処理を丸ごと迂回するので，Stage1/Stage2 からはデータ以前に**クラスとして**到達不能である．**P4 + G1** は不変． |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 7 | `src/maou/app/learning` | `gradient_noise_scale.py:150,189-192,247` — one GPU sync per parameter tensor per micro-batch (`.item()` inside `for param in model.parameters()`): 60-300 syncs per micro-batch on a ResNet/ViT backbone whenever adaptive batch is on. Accumulating into a device scalar changes when the value materializes, and GNS feeds the adaptive batch controller — needs a numerical equivalence check. **2026-08-13 に再確認** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md)): 3 箇所とも健在 (`:150` / `:188-192` / `:246-248`)．加えて `g.clone()` (`:151`/`:193`) が勾配の全複製をパラメータ毎に確保する．緩和策の `should_measure` (`:115-122`) はあるが **`measurement_interval` の既定は `1`** (`:85`) なので，caller が上書きしない限り毎 optimizer step で走る．値は `compute()` → `b_noise` → `training_loop.py:1024-1031` で Python float として消費され `gradient_accumulation_steps` を書き換えるため，controller 側を tensor 値に作り替えないと同期は遅延できない．**P4 + G1**． **2026-08-14 の 2 回目の run にユーザが「GPU で測れなくても意味論的に等価な変更は出荷してよい」と回答** (`/audit-backlog` step 3d, [記録](2026-08-14-backlog-decided-work-second-pass.md), Deferred 5/6/7 を 1 問で governs)．**ただし本行の G1 は retire されない** — device スカラーへの蓄積は値が materialize するタイミングを変え，その値が `gradient_accumulation_steps` を書き換えるので，**数値等価性の確認が必要**であり「意味論的に自明な等価変換」には当たらない．**P4 + G1** は不変． **2026-08-15 の再検証: `gradient_noise_scale.py` の行番号は全て記録どおりで移動なし** — `measurement_interval: int = 1` `:85` (clamp は `:88-89`)，`should_measure` `:115-122`，第 1 マイクロバッチ経路の `.item()` `:150` と `g.clone()` `:151`，蓄積経路の `.item()` `:189`/`:192` と `g.clone()` `:193`，`compute()` の `.item()` `:247` (式は `:246-248`)．**消費側だけ +7 ずれている**: `b_noise` の生成は `:304`，`GNSEstimate` として返すのが `:307`，消費は `_maybe_update_adaptive_batch` (**`:1018`**) の **`:1031-1038`** (記録の `:1024-1031`) で，`self.gradient_accumulation_steps = new_steps` は `:1038`．鎖は健在．**P4 + G1** は不変． **2026-08-15 の 2 回目の run にユーザが緩和策の設計判断を回答: 「`measurement_interval` の既定を上げる」** (`/audit-backlog` step 3d, [記録](2026-08-15-backlog-writeable-contract-and-decisions.md))．本丸 (device スカラー化) を GPU 環境まで待つ間，既定値を上げて毎 optimizer step の sync を減らす．却下されたのは「既定は `1` のまま本丸を待つ」と「既定を廃して caller に必須指定させる」(`GradientNoiseScale` の署名が変わる P6)． **ただし実装直前の調査で "changed shape" が判明した — 質問時に述べた「1 行で contained」は誤りである**．`measurement_interval` の既定は **3 箇所**にあり，本番経路を決めているのは**クラス既定ではなく CLI 既定**である: (i) `gradient_noise_scale.py:85` `measurement_interval: int = 1`，(ii) `adaptive_batch.py:68` `measurement_interval: int = 1`，(iii) **`infra/console/learn_model.py:217` の `--adaptive-batch-measurement-interval` `default=1`**．CLI は `:984` で `measurement_interval=adaptive_batch_measurement_interval` と**常に明示的に渡す**ので，(i)(ii) だけ変えても**本番経路の挙動は 1 ミリも変わらない**．したがって決定を実現するには (iii) を変える必要があり，**CLI オプションの既定変更 → CLAUDE.md の MUST により `docs/commands/learn_model.md` の更新が伴う** (この doc 更新はコードから一意に決まるので P2 の drift correction 相当)．この run では**実装していない** — ユーザが「1 行で contained」という説明のもとで回答したため，受理済みの PR #505 に CLI 既定 + doc の変更を後から足すのは受理の範囲を超えると判断した． **値の選択の材料** (次 run 向け): リポジトリ自身の推奨は既に存在する — `gradient_noise_scale.py` の docstring `:76-78` が「大規模モデル(数百M〜数Bパラメータ)では 5〜10 を推奨」と書き，`learn_model.py:222` の help も "Recommended: 5-10 for large models (100M+ params)" と書き，`training_benchmark.py:2310` の `_recommend_measurement_interval` は **1 / 5 / 10** のいずれかを返す．**推奨は 5** (リポジトリ自身の推奨帯の下限で，sync を 5 倍減らしつつ既存のガイダンスと矛盾しない)． **残る作業**: (iii) を主とする既定値の変更 + `docs/commands/learn_model.md` の更新 + 回帰テスト，および本丸の device スカラー化 (**G1 は残る** — 数値等価性を測れない)．**P4** は不変 (既定を渡していない既存の実行は挙動が変わる)． |

### Out-of-scope backlog

Findings an audit surfaced **outside** the path it was auditing, and was
therefore not allowed to fix.

| Found by | Target | Item |
|---|---|---|
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) D13 | `src/maou/infra/file_system` | **(a) の contained な部分と (c) は 2026-08-12 に消化済み** (`/audit-backlog`, PR #456)．探索対象を `__init__` へ巻き上げ `bisect.bisect_right` に置き換え，`_STRUCTURED_DTYPES` は `get_dtype` へ畳んだ．**残り**: (a) の**根本**部分 — `__getitem__` は訓練サンプル1件ごとに `np.empty(1, dtype)` (preprocessing で 3,089B) を確保して6フィールドを memcpy する．batch 1024 で約3MB の memcpy + 1024回の小確保/バッチ．`dataset.py:230` の `torch.from_numpy` は「ゼロコピー」と称しているが，コピーは1層下で既に起きている．根本解決 (`KifDataset` が `ColumnarBatch` を直接スライスする) は `app/learning/dataset.py` と ABC を触るので path 外．(b) `get_items` は「バッチで取得」と謳いながら `get_item` を要素ごとに Python ループで呼ぶだけで何もバッチ化していない．`ColumnarBatch.slice` によるベクトル化が可能 — (a) の根本解決と同じ方向なので一緒に扱うべき．**2026-08-13 の再検証**: `FileDataSource.get_items` の呼び出し側は**ゼロ** (`FileManager.get_items` への内部委譲だけ) なので (b) 単独では **dormant**．実害は (a) の根本解決に踏み込んだときに初めて出るため，「(b) を先に直す」選択肢は実質無い．**記録の見立ての誤り**: (c) を消せば `assert self._structured_dtype is not None` が不要になると書かれているが，`_use_columnar` が False (hcpe) の経路では属性は依然 None なので型は Optional のままで，assert は落とせない (PR #456 後も `:613`/`:733` に残っている)． **2026-08-13 の再検証で数値を訂正** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md)): 機構は健在だが記録の数字が古い．1 サンプルは **3,089B ではなく 9,081B** (`get_preprocessing_dtype` に `moveWinRate` (f32×1496 = 5,984B) が入ったため)，コピーは **6 フィールドではなく 8 書き込み** (`:586-587` の `id` ゼロ埋め含む)．`np.empty(n, dtype)` は `:584`，`torch.from_numpy` は `dataset.py:230` ではなく **`:239`** (「ゼロコピー」を謳うコメントは `:95`)．batch 1024 なら約 **9.3MB** の memcpy/バッチ．hcpe 経路は view を返すので影響なし — 割り当ては columnar 型に限る．(b) の caller ゼロも再確認 (`tests/` にもヒット無し)．**P4/P6 + G2 + G4**． **2026-08-14 の再検証で機構の所在を訂正** ([記録](2026-08-14-backlog-diagnostics-and-npy-remnants.md))．`__getitem__` 自身はもう確保をしていない — `:692-696` の薄い委譲 (`return self.__file_manager.get_item(...)`) になり，実体は `_columnar_batch_to_structured_array` (**`:553-634`**) へ移った (`get_item` → `_columnar_to_structured_record` `:471-494` 経由)．`np.empty(n, dtype=self._structured_dtype)` は `:584` で行番号は一致するが，**フィールド書き込みは `:586-632`** に広がっており，前の記述の `:586-587` は狭すぎる (8 書き込みという数は正しい)．`get_items` (外側 `:701-712`) が `FileManager.get_items` (`:496-509`) へ委譲し，そこが `[self.get_item(idx) for idx in indices]` で回しているのも不変．結論は不変． **2026-08-14 の 2 回目の run の再検証で行番号を更新**: 前 run の `bdda7b5` がファイルを縮めたため `_columnar_batch_to_structured_array` は **`:408-`** (記録の `:553-634`)，`np.empty` は **`:439`** (`:584`)，`_columnar_to_structured_record` は **`:338-359`** (`:471-494`)，`FileManager.get_items` は **`:363-`** (`:496-509`)，`FileDataSource.__getitem__`/`get_items` は **`:310-`/`:550-`** (`:692-696`/`:701-712`)．**2026-08-14 の 2 回目の run にユーザが設計判断を回答: 「`ColumnarBatch` を直接スライスする」** (`/audit-backlog` step 3d, [記録](2026-08-14-backlog-decided-work-second-pass.md))．**1 つの回答が (a) の根本と (b) の両方を governs する** — `KifDataset` が `ColumnarBatch` を直接スライスするようにし，サンプル毎の `np.empty(9,081B)` + 8 フィールド memcpy (batch 1024 で約 9.3MB/バッチ) を無くす．「現状維持」「`get_items` だけベクトル化」「先に計測してから決める」は却下された．**G4 は retire** (決めは済んだ)．**G2 は残る** — `app/learning/dataset.py` と ABC に触るので `infra/file_system` の path 外に出る．**残る作業**: `KifDataset` 側のスライス経路の新設，`FileManager.get_item`/`get_items` の縮退，hcpe 経路 (view を返すので影響なし) との分岐整理．`assert self._structured_dtype is not None` は `_use_columnar` が False の経路が残る限り落とせない点も不変．**P4/P6** は不変． **2026-08-14 の 3 巡目で行番号を再確認**: `_columnar_to_structured_record` **`:338`**，`FileManager.get_items` **`:363`**，`_columnar_batch_to_structured_array` **`:408`**，`np.empty(n, dtype=self._structured_dtype)` **`:439`**，`FileDataSource.__getitem__` **`:541`**，`FileDataSource.get_items` **`:550`** — いずれも前 run の記述どおりで移動なし．結論は不変．この run では着手していない (**設計は決定済みで人間待ちではなく**，G2 の作業量が今回の枠に入らなかっただけ — 次 run の先頭候補)． **2026-08-15 の再検証で "changed shape" — 決定済みの設計が，記録の処方のままでは実装できないことが判明した** ([記録](2026-08-15-backlog-columnar-uninitialized-fields.md))．行番号は前 run から一律 **+14** (`_columnar_to_structured_record` **`:352`**，`FileManager.get_items` **`:377`**，`_columnar_batch_to_structured_array` **`:422`**，`np.empty` **`:453`**，`FileDataSource.__getitem__` **`:555`**，`get_items` **`:564`**，`assert self._structured_dtype is not None` **`:441`**)．`get_items` の caller ゼロも再確認 (`src/`+`tests/` 全体でヒット 3 件，すべて本ファイル内の定義と内部委譲)． **本題**: 「`KifDataset` が `ColumnarBatch` を直接スライスする」は**そのままでは本番データで落ちる**．`_explode_list_column` (`domain/data/schema.py:782`) は polars 側の dtype が目標 dtype と一致するとき `astype` を挟まず `to_numpy()` の結果をそのまま返すが，**polars の `to_numpy()` は Arrow バッファの read-only ビューを返す**．preprocessing の polars スキーマ (`schema.py:457-475`) は `boardIdPositions: List(List(UInt8))` / `piecesInHand: List(UInt8)` / `moveWinRate: List(Float32)` で，いずれも目標 dtype と**一致する**ので `ColumnarBatch` のこれら 3 フィールドは **`writeable=False`** になる (実測済み)．一方 `moveLabel` (Float32→float16) と `resultValue` (Float32→float16) は `astype` が挟まるのでコピーになり writeable．そして `KifDataset._numpy_to_tensor` (`app/learning/dataset.py:232-238`) は read-only を**設計として `ValueError` で撥ねる**．**この罠は合成テストでは絶対に露見しない** — テストが Python の list から組む DataFrame は int64/float64 になり `astype` が挟まって全フィールドがコピーになるため，直接スライス経路は**テストで全部通り，本番データだけで落ちる**． **したがって新しい設計判断が要る (実質的な G4 の再発)**: writeability をどこで確立するか — (a) `domain/data/schema.py` の `_explode_list_column` が常に writeable を返す (最上流で一度，ただし domain 層に触り，現在ゼロコピーの箇所にコピーが入る)，(b) `infra` の `FileManager.__init__` がロード後に read-only フィールドだけコピーする (層は infra に閉じるがピークメモリが一時的に増える)，(c) `app` の `KifDataset` 側が read-only を受け入れる (`_numpy_to_tensor` の writeable チェックを緩める — チェックが存在する理由を壊す)．**G2 は不変** (`app/learning/dataset.py` と ABC に触る)． **2026-08-15 にユーザが writeability の設計判断を回答: 「(a) `domain` の `_explode_list_column` が常に writeable を返す」** (`/audit-backlog` step 3d, [記録](2026-08-15-backlog-columnar-uninitialized-fields.md))．最上流で一度だけ保証するので**全経路が自動的に正しくなり，罠が構造的に消える**．却下されたのは (b) `FileManager` がロード後にコピー (層は infra に閉じるが，`streaming_file_source.py` など `FileManager` を通らない経路には効かず同じ罠が別の場所で再発しうる) と (c) `KifDataset` が read-only を受け入れる (`_numpy_to_tensor` の writeable チェックが存在する理由 — テンソルが共有ストレージへ書き戻すのを防ぐ — を壊す) と (d) D13 を落とす．**これで実装可能性の問題は解消し，実質的に再発していた G4 は retire された** — D13 は再び「決定済み，あとは作業」である． **残る作業** (順に): (1) `_explode_list_column` (`domain/data/schema.py:782`) の fast path で `astype` が挟まらなかった場合に writeable を保証する (`np.array(..., copy=True)` 相当)．`ColumnarBatch` の docstring が謳う「C-contiguous」に「writeable」を足す形で契約を明示すること．**回帰テストは polars 側 dtype が目標 dtype と一致する列** (`List(UInt8)` → `uint8` 等) **で書かないと空虚になる** — Python の list から組んだ DataFrame は int64/float64 になり `astype` が挟まって必ず writeable になるため．(2) `KifDataset` 側のスライス経路の新設，(3) `FileManager.get_item`/`get_items` の縮退，(4) hcpe 経路 (view を返すので影響なし) との分岐整理．**G2 は残る** — (2)(3)(4) が `app/learning/dataset.py` と ABC に触るため．**P4/P6** は不変． **2026-08-15 の 2 回目の run で残る作業 (1) を出荷した** ([記録](2026-08-15-backlog-writeable-contract-and-decisions.md), [PR #505](https://github.com/dousu/maou/pull/505), `0b3f1ab`)．`_explode_list_column` の fast path に `elif not result.flags.writeable: result = np.array(result, copy=True)` を入れ，`astype` が挟まらなかった場合だけコピーを取るようにした (`astype` 経路は既にコピーなので追加コストはゼロ)．「C-contiguous かつ writeable」を `_explode_list_column` と `ColumnarBatch` の docstring に契約として明記．**回帰テスト 10 本**は polars 側 dtype を目標 dtype と明示的に一致させて書いてあり (list から組むと `astype` が挟まって空虚になるため)，無効化テストで **10 本中 7 本が落ちる**ことを確認済み．**これで本番データでのみ落ちる罠は構造的に消えた**． **残る作業は (2)(3)(4) のみ**: (2) `KifDataset` 側のスライス経路の新設，(3) `FileManager.get_item`/`get_items` の縮退，(4) hcpe 経路 (view を返すので影響なし) との分岐整理．**G2 は残る** — (2)(3)(4) が `app/learning/dataset.py` と ABC に触るため．`assert self._structured_dtype is not None` は `_use_columnar` が False の経路が残る限り落とせない点も不変． **2026-08-15 の 2 回目の run で行番号を更新** (前 run の `033d49f` がファイルを伸ばしたため `file_data_source.py` は **+15〜+20**): `_columnar_to_structured_record` **`:367`** (前 `:352`)，`FileManager.get_items` **`:392`** (`:377`)，`_columnar_batch_to_structured_array` **`:437`** (`:422`)，`assert self._structured_dtype is not None` **`:456`** (`:441`)，`np.empty(n, dtype=self._structured_dtype)` **`:468`** (`:453`)，`FileDataSource.__getitem__` **`:575`** (`:555`)，`FileDataSource.get_items` **`:584`** (`:564`)．`schema.py` 側の `_explode_list_column` は **`:782`** で記録どおり．**P4/P6** は不変． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O9 | `src/maou/infra/bigquery` | `bq_data_source.py:212-234` (記録の `:222-243` から移動) — `sample_ratio` 指定時，`total_rows` は `TABLESAMPLE` 付き `COUNT(*)` から得るが `__fetch_from_bigquery`(**`:395-411`**) はページごとに**別々に引き直す** `TABLESAMPLE` を発行する．ファイル系ソースは常に厳密．`:226` の `num_rows` はテーブルメタデータでストリーミング挿入に遅れる． **2026-08-13 の再検証で記述を強めた** ([記録](2026-08-13-backlog-callback-accumulator-table.md)): 非決定性は**二重**である．(i) `TABLESAMPLE SYSTEM` はクエリごとに独立評価なので数えた行集合と返す行集合が別物，(ii) その上で `LIMIT/OFFSET` を **`ORDER BY` 無しの再サンプル結果**に掛けているので，**同じ `page_num` を 2 回引いても同じ行が返る保証が無い**．「件数がずれる」ではなく**再現性が無い**． **修正の向きが 3 つに割れている** (G4 相当): (a) サンプルを一時テーブルへ 1 度実体化してページングする，(b) `FARM_FINGERPRINT` 等の決定的ハッシュ条件へ置き換える (キー列の決めが要る)，(c) `sample_ratio` とページングの併用を拒否する．**G1**: BigQuery がこの環境に無く，実際の非決定性は fake client では再現できない． **2026-08-13 に再確認** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md)): 行が移動して `__get_total_rows` は `:213-234` (サンプル分岐 `:215-223`)，`total_pages` の導出は `:202-204`，ページ側の再サンプルは `__fetch_from_bigquery` (def `:345`) の `:396-405` (`start_index = page_num * batch_size` `:398`)．クラスタ/パーティション経路も `tablesample_clause` `:380-383` を `:385-389` に差し込むので同じ欠陥を弱い形で持つ．非サンプル経路は `list_rows(start_index=…)` (`:415-419`) なので無傷．**緩和材料**: `get_page` (`:486-520`) がページをキャッシュするため 1 run 内では通常 1 回しか引かれず，非再現性は退避時と run 跨ぎで表面化する． **2026-08-14 にユーザが設計判断を回答: 「決定的ハッシュ条件に置き換える」** (`/audit-backlog` step 3d, [記録](2026-08-14-backlog-cache-knob-removal.md))．3 つの向きのうち **(b)** が選ばれた — `TABLESAMPLE SYSTEM` をやめ，`MOD(FARM_FINGERPRINT(<key>), N) < k` 等の**行に対して決定的な条件**へ置き換える．推奨として提示した (a) 一時テーブルへの実体化は却下された (状態を持たずに済む方を選好)．**G4 は retire** (決めは済んだ)．**G1 は残る** — BigQuery がこの環境に無く，実際の再現性は fake client では確認できない．**残る作業と未解決点**: (i) **キー列の決め**がこの方式の要 — 分布が偏るとサンプルも偏るので，`id` 系の高カーディナリティ列を選ぶ必要がある (現行スキーマに一意キーの保証があるかは未調査)．(ii) `__get_total_rows` (`:213-234`) の COUNT とページ側 (`__fetch_from_bigquery` `:396-405`) が**同じ条件式**を使うようにする — 現在は各々が独立に TABLESAMPLE を発行しているのが二重の非決定性の源．(iii) クラスタ/パーティション経路 (`tablesample_clause` `:380-389`) も同じ差し替えが要る．(iv) `LIMIT/OFFSET` に `ORDER BY` が無い点は条件を決定的にしても残るので，安定な並び順を与える必要がある．**P4** は不変 (同じ呼び出しで返る行が変わる)． **2026-08-14 の 3 巡目にユーザが (i) の未決点に回答: 「行全体のフィンガープリントを使う」** (`/audit-backlog` step 3d, [記録](2026-08-14-backlog-stream-wait-and-abc-direction.md))．`MOD(ABS(FARM_FINGERPRINT(TO_JSON_STRING(t))), N) < k` の形にし，**キー列は選ばない**．却下されたのは (b) `id` 列 (preprocessing の `id` は盤面ハッシュなので同一局面が必ず同じバケットへ落ちてサンプルが偏る) と (c) clustering/partitioning キー優先 (経路ごとに挙動が変わる) と (d) `sample_ratio` とページングの併用拒否 (既存の呼び出しが動かなくなる)．**これで (i) は解決**で，残る作業は (ii) COUNT 側 (`:213-234`) とページ側 (`:396-405`) が**同じ条件式**を使うようにすること，(iii) クラスタ/パーティション経路 (`tablesample_clause` `:380-389`) の同じ差し替え，(iv) `LIMIT/OFFSET` に安定な `ORDER BY` を与えること の 3 点．**G1 は残る** — BigQuery がこの環境に無く，実際の再現性は fake client では確認できない．設計は全て決まったので，**BigQuery に触れる環境があれば通常作業として着手できる**． **2026-08-15 の再検証で行番号を更新** (一律 −1〜−4)．`__get_total_rows` **`:212-233`** (サンプル分岐 `:214-222`)，`total_pages` の導出 **`:201-203`**，`__fetch_from_bigquery` の def **`:344`**，ページ側の再サンプル **`:395-404`** (`start_index = page_num * self.batch_size` は `:397`)，`tablesample_clause` **`:380-382`** とその差し込み **`:384-388`**，非サンプル経路の `list_rows(start_index=…)` **`:412-418`**，`get_page` のキャッシュ **`:485-532`**．欠陥の構造は不変 (COUNT 側とページ側が**別々の** `TABLESAMPLE SYSTEM` を発行し，その上に `ORDER BY` 無しの `LIMIT/OFFSET` を掛けている)． **テスト網羅の実態を確定** (これまで未記載): **`sample_ratio` / `TABLESAMPLE` に触れるテストは `tests/` に 1 件も無い**．実クライアントを叩く `test_bq_data_source.py` は `@pytest.mark.skipif(os.getenv("TEST_GCP") != "true")` で全体が閉じており，"fake" と呼ばれる 2 本 (`test_bq_get_item_contract.py:36` / `test_bq_iter_batches_contract.py:44`) は **BigQuery クライアントを fake していない** — `object.__new__(PageManager)` で組んでから `pm.get_page` を lambda に差し替えており，**まさにこの所見の対象であるサンプリング経路を丸ごと潰している**．`__fetch_from_bigquery` / `__get_total_rows` / `get_page` のキャッシュは**完全に未テスト**である．G1 (BigQuery がこの環境に無い) に加えて，**修正時には回帰テストの土台自体を作る必要がある**ことがこれで確定した．**P4 + G1** は不変． **2026-08-15 の 2 回目の run にユーザがテスト土台の設計判断を回答: 「fake BigQuery client のテスト土台を新設して修正に同梱する」** (`/audit-backlog` step 3d, [記録](2026-08-15-backlog-writeable-contract-and-decisions.md))．却下されたのは「既存の 2 本の "fake" テストを先に独立して作り直してから本題に入る」(run が 2 つに分かれる) と「実 BigQuery 環境が取れるまで着手しない」．**この回答は G1 の意味を実質的に変える** — 決定的ハッシュ条件 (`MOD(ABS(FARM_FINGERPRINT(TO_JSON_STRING(t))), N) < k`) の要点は「**同じ `page_num` を 2 回引くと同じ行が返る**」ことであり，これはクライアントを fake すれば実 BigQuery なしで CI 上で検証できる．**したがって「実 BigQuery が無いと着手できない」は成り立たなくなり，本行は設計・テスト方針とも決定済みの通常作業になった**．ただし**実クエリが BigQuery 上で意図どおり動くことの最終確認は依然として実環境が要る**ので，G1 は「出荷前の実地確認」に縮小した形で残る． **残る作業** (順に): (0) **fake BigQuery client のテスト土台の新設** — 既存の 2 本 (`test_bq_get_item_contract.py:36` / `test_bq_iter_batches_contract.py:44`) は `object.__new__(PageManager)` で組んで `pm.get_page` を lambda に差し替えており対象経路を丸ごと潰しているので，土台はこれとは別に作る，(ii) `__get_total_rows` (`:212-233`) の COUNT 側とページ側 (`__fetch_from_bigquery` `:395-404`) が**同じ条件式**を使うようにする，(iii) クラスタ/パーティション経路 (`tablesample_clause` `:380-382` とその差し込み `:384-388`) の同じ差し替え，(iv) `LIMIT/OFFSET` に安定な `ORDER BY` を与える．**P4** は不変． |
