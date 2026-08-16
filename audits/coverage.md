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
| `src/maou/app/learning` | python | done | high | `52d9bd2` | [2026-08-08](2026-08-08-src-maou-app-learning.md) | 0 |
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

- [2026-08-15 backlog stage-unification-and-gns-interval](2026-08-15-backlog-stage-unification-and-gns-interval.md)
  (**Deferred 2 と Deferred 5 の 2 行を削除**)．**15 run ぶりに backlog の
  行が減り，しかも 2 本消えた** run である．内訳は 2 通り: **Deferred 2** は
  過去 4 run の設計判断が作った「人間待ちではないただの作業」の在庫を
  消化しただけ (5 行のうち 3 行が既にその状態にあった)．**Deferred 5** は
  **本 run の再検証が G1 の前提そのものを崩した**結果で，3 run 連続で
  「GPU で測れないから」と塞がれていたが，経路がクラスとして到達不能で
  ある以上**測る対象が無い**．これを 3d で問い，ユーザが
  **「到達不能な今のうちに同期を除去する」**を選んだ (却下されたのは
  「休眠経路ごと削除」と「現状維持」)．**ゲートは再検証で外れることが
  あり，外れたゲートは同じ run 内で消化まで行ける**というのが本 run の
  実質である．
  5 行を再検証して **stale 0 / changed shape 1 / confirmed 4**，
  **行番号の移動は 5 行すべてでゼロ** (前 run 以降 `src/` へのコミットが
  無かったため — 2 run 連続)．**自動帯は空** (15 run 連続)．
  出荷は 2 件とも P4 でゲート無し: **B-1** (`fdbc990`) が Stage1/Stage2 の
  4 組 (run 関数 2 本 / 工場の完全一致 28 行 / `dataset.py` の対 /
  `streaming_dataset.py` の対) を共通実装へ統合し，**`TrainingLoop`
  サブクラスの注入機構は作らなかった** (`training_loop.py:1183` が別名で
  差異が存在しないと前 run が確定させたため — 2026-08-14 の設計判断のうち
  この部分は不要になっていた)．**B-2** (`57f0664`) が
  `measurement_interval` の既定を 4 箇所すべてで 1 → 5 にした．
  **B-3** (`a57ad2c`) が `_compute_policy_loss` のマスキング腕から
  per-batch host sync 2 つ (`if not has_legal.all():` の
  `Tensor.__bool__` と `int((~has_legal).sum().item())`) を除去した —
  `safe_mask = mask_bool | ~has_legal.unsqueeze(1)` は `has_legal` が
  全 True のとき `mask_bool` と厳密に一致するので**分岐自体が冗長**で
  あり，数値結果は変わらない (全ゼロ行の警告ログは廃止．診断を戻す場合は
  per-batch で同期しない形にすべき旨をコメントに残した)．
  回帰テストは `Tensor.item` / `Tensor.__bool__` を例外に差し替えて
  経路を通すことで**同期の不在を CPU 上で直接検証**しており，
  旧分岐形に戻すと 2 本が落ちることを確認済み．
  **Deferred 7 は "changed shape"** — 前 run が「既定は 3 箇所」と書いたが
  実際は **4 箇所**で，`infra/console/utility.py:688`
  (`benchmark-training` の同名オプション) が数えられていなかった．
  元記録への訂正は**なし** (診断ではなく前 run の backlog 行の記述の
  不足だったので，行の側で訂正した)．
  回帰テストは**クラス既定と CLI 既定を別々に**検証しており，
  「クラス既定だけ直して CLI 既定を 1 のまま残す」罠を無効化テストで
  固定してある．副産物として
  `test_gradient_noise_scale.py::test_reset_between_cycles` が
  **既定が 1 であることに暗黙に依存**していたことが判明し，同 commit で
  意図どおり `measurement_interval=1` を明示する形に直した．
  残る 3 行 (Deferred 7 の本丸 / D13 / O9) は**いずれも設計が決定済みで
  人間待ちではなく**，枠に入らなかっただけである — **G4 も未回答の設計判断も
  現時点でゼロ**なので，次 run は D13 (2)(3)(4) から通常作業として始められる．環境: 素のコンテナから
  `uv sync --extra cpu` をやり直した際，既定の `UV_HTTP_TIMEOUT=30` では
  torch の依存取得がタイムアウトするので `UV_HTTP_TIMEOUT=300` が要る．
  QA は全て実行できた (**2016 passed**，G3 は発生せず)．

- [2026-08-15 backlog gns-sync-and-batch-api](2026-08-15-backlog-gns-sync-and-batch-api.md)
  ([PR #507](https://github.com/dousu/maou/pull/507) — **Deferred 7 の 1 行を削除．
  deferred backlog はこれで空になった**)．**G4 も未回答の設計判断もゼロという
  状態から始まった最初の run** で，3 行を再検証して **stale 0 / changed shape 0 /
  confirmed 3**．**自動帯は空** (16 run 連続)．指定ブランチ 1 本の制約でクラス毎の
  PR 分割ができず，判断帯 2 件が同じ PR に同居したため，レビュー単位は commit が担った．
  **本 run の実質は「ゲートの本文を読み直すと外れることがある」の 2 例目である** —
  前 run が Deferred 5 の G1 を「経路が到達不能なら測る対象が無い」と崩したのに続き，
  本 run は **Deferred 7 の G1 を崩した**．ゲートは「数値等価性の確認が必要」と
  書いてあり，3 run にわたり「GPU が無いので確認できない」と読まれていたが，
  旧実装の `acc += x.item()` は **Python float すなわち float64 の逐次加算**なので，
  累算器を **float64 の device スカラー**にして同じ順序で加算すれば結果は
  **bit 単位で同一**になり，その一致は旧算術を写した参照実装との突き合わせで
  **CPU 上で厳密に検証できる**．GPU が要るのは「速くなった」ことの計測だけである．
  出荷は 2 件とも判断帯: **B-1** (`b5b4457`，P4) が
  `gradient_noise_scale.py` の勾配統計を device 上の float64 スカラー累算に変え，
  host への materialize を `compute()` の 1 回 (S と G を `torch.stack` して
  1 回の `tolist()`) に集約した — 同期は計測サイクルあたり
  *(パラメータ数 × micro-batch 数 + パラメータ数)* 回から **1 回**になり，
  パラメータ数にも micro-batch 数にも依存しなくなった．回帰テスト 4 本のうち
  2 本は参照実装との**厳密一致** (float32 / float16)，2 本は
  `Tensor.item` / `Tensor.tolist` を数えて**同期の不在**を CPU 上で直接観測する．
  **累算器を float32 に落とすと厳密一致テストが落ちる**ことを無効化テストで
  確認済み (float64 であることが classification そのものを支えている)．
  **B-2** (`380866c`，P6) が `get_items` を `FileDataSource` と `FileManager` の
  両方から削除した — これは修理ではなく **revert の完了**で，`docs/adr-003` §5 が
  "❌ FAILED - REVERTED" と記録している実験の残骸 (`__getitems__` だけが消えて
  包み紙が残っていた) である．ADR は過去形の記録なので**編集していない**．
  `AskUserQuestion` は**受理 1 問 + 設計判断 2 問**で 3 件とも回答を得た:
  **Q1 両方マージ**，**Q2 D13 の口は「ABC に列アクセサを追加」**
  (**(2)(3)(4) を governs**．過去 3 run が「G2 の作業量」とだけ書いていた先送りの
  真因が，実は **ABC が `__getitem__`/`__len__` の 2 つしか持たず列に届く口が
  未定だったこと**だと本 run が特定した)，**Q3 O9 の並び順は
  `ORDER BY fingerprint` + `LIMIT/OFFSET`** (**これで O9 の未定点はゼロ**)．
  **Q2/Q3 とも決定を行に書いたが実装は次 run 以降** (D13 は G2，O9 は (0) の
  fake client 土台の新設が枠に入らなかった)．**予算に入らなかった設計判断は無い**
  (開始時点で G4 ゼロ，新たに見つかった穴が 2 件だけだったので 4 問枠に対し 3 問で
  足りた)．元記録への訂正は**なし** (行番号のずれのみ — B-1 が一律 +1)．
  付随修復として **`uv.lock` の `maou` version が `main` 時点で `0.92.2` のまま
  `pyproject.toml` (`0.93.1`) からずれていた**のを追随させた
  (`pre-commit` の `uv-lock` hook は通るので検出されない種類のずれ)．
  環境: git の pre-commit hook がコンテナに入っていなかったので
  `uv run pre-commit run --from-ref/--to-ref` で 2 commit 分を明示実行し全 hook Passed．
  QA は全て実行できた (**2024 passed**，G3 は発生せず)．

- [2026-08-16 backlog columnar-accessor](2026-08-16-backlog-columnar-accessor.md)
  ([PR #508](https://github.com/dousu/maou/pull/508) — **D13 の 1 行を削除**)．
  **4 run にわたり「次 run の先頭候補」と名指しされ続けた D13 を全消化した
  run** である．2 行を再検証して **stale 0 / changed shape 0 / confirmed 2**，
  **行番号の移動は 2 行ともゼロ** (前 run 以降 `src/` へのコミットが無かった
  ため — 3 run 連続)．**自動帯は空** (17 run 連続)．
  **本 run の実質は「設計判断を行に書いておけば後の run が通常作業として
  拾える」仕組みが最後まで走り切った初めての例**である — D13 は
  2026-08-14 に「`ColumnarBatch` を直接スライスする」，2026-08-15 に
  「writeability は `_explode_list_column` で保証する」，同 8-15 に
  「口は ABC に列アクセサを追加する」と **3 段階の設計判断**を積んでおり，
  本 run は **1 問も設計を問うことなく実装だけを行っている**
  (`AskUserQuestion` は受理 1 問のみ．開始時点で G4 ゼロ，実装中にも
  新しい未定点が出なかったため設計判断の枠は使っていない)．
  指定ブランチ 1 本の制約でクラス毎の PR 分割ができないため，レビュー単位は
  commit が担い，台帳の行削除は**同じ PR の中**に入れてある (6a の
  separability test)．
  出荷は **B-1** (`7c0157b`，P4 + G2) の 1 件で，学習側 `DataSource` ABC に
  `columnar_record(idx) -> dict[str, np.ndarray] | None` を足し (既定 `None`
  なので BigQuery / ObjectStorage / テストの fake 7 本は無変更で従来経路に
  落ちる)，`FileDataSource` が `ColumnarBatch` のビューを辞書で返し，
  `KifDataset` / `_StageDataset` がそれを使う．訓練サンプル 1 件ごとの
  `np.empty(9,081B)` + 8 フィールド memcpy (batch 1024 で約 9.3MB/バッチ) が
  **production の経路から消えた**．
  **G2 は step 4a の「隣を取り込む」側で解消した** — 4 run にわたり
  「作業量が枠に入らない」と読まれてきたが，結合は「ABC に口が無い限り
  infra 側だけでは何もできない」という構造的なもので，見積りではなく設計から
  決まる．拡大そのものはユーザに問うている．
  **行の記述に 1 点の不正確さがあった** — 口を生やす先を `LearningDataSource`
  と書いていたが，`__getitem__`/`__len__` を宣言し消費側が型付けされているのは
  `DataSource` (`dataset.py:45`) の方で，`LearningDataSource` (`dl.py:71`) は
  `DataSourceSpliter` を足すだけの中間抽象である．前者に置かないと消費側から
  型安全に呼べない (継承関係があるので決定の実質は満たされている)．
  **2 つの罠を回帰テストで固定した**: (i) キー集合を「batch が実際に持つ列」で
  組むと，`moveWinRate` 列を持たない旧 `.feather` で structured 経路は 3 要素
  (ゼロ埋め)，速い口は 2 要素になり**同じデータで教師の形が経路によって変わる**
  ので，dtype 駆動に固定した．(ii) `source[row]` は 1 次元列でスカラーを返すため
  `resultValue` が 0 次元配列でなくなるが `.item()` はどちらでも通るので
  **値の比較では検出できない**．
  回帰テスト 16 本 (infra 10 / app 6)．等価性は同一データを 2 経路で読む
  characterization test で，経路そのものは `np.shares_memory` で観測する
  (値の一致だけでは経路を判別できない — 両経路が同じ値を返すのが正しいため)．
  **無効化テスト 9 通り**すべてで意図した本数だけ落ちることを確認．うち 1 つ
  (`_StageDataset` を遅い経路のまま) は**最初は空虚で**，`np.shares_memory` に
  よる経路の直接観測を足して初めて落ちるようになった．
  **O9 は re-triage** — 行番号の移動ゼロ，設計・テスト土台・並び順とも決定済みで
  **未定点はゼロ**．(0) の fake BigQuery client 土台の新設が D13 の枠と同居
  できなかっただけで，**次 run の唯一の候補**である．
  元記録への訂正は**なし**．`reviews/` の起票も**なし** (学習 `DataSource` の
  抽象メソッド集合を列挙している durable doc は存在しない)．
  **`AskUserQuestion` は受理 1 問 + 設計判断 1 問で，2 件とも回答を得た** —
  **Q1 両方の commit をマージ**，**Q2 `PolarsDataFrameSource` は削除**．
  Q2 は**開始時点には存在しなかった設計判断**で，D13 の実装中に表面化した:
  このクラスは `_PolarsRow` / `_PolarsField` / `_FakeDtype` の 3 つの shim で
  structured array のフィールドアクセスを**模して**おり ABC を継承しないので，
  列アクセサを足した結果 `KifDataset` に `isinstance` ガードを要求する
  **唯一の理由**になっていた．却下されたのは「`DataSource` を実装させる」
  (`__getitem__` が `_PolarsRow` を返すので契約に合わせるには戻り値の作り直しか
  `type: ignore` が要り shim 3 つは残る) と「現状のまま残す」．
  **Q2 は同 run 内で実装した** (`654986c`，P6 + `feat!` + minor bump —
  1.0 以前なので破壊的変更の置き場は minor．`380866c` の `get_items` 削除に
  ならった)．`polars_datasource.py` (365 行) とそのテスト (232 行) を削除し，
  不在を固定する回帰テストを 1 本足した．doc drift 1 件
  ([reviews/2026-08-16-rust-backend-polars-dataset-section.md](../reviews/2026-08-16-rust-backend-polars-dataset-section.md)，
  `status: applied` / `applied_in: 654986c`) は **P2 の standing approval で適用** —
  節の構成が丸ごと削除対象クラスに従属しており「節を残す書き方」が存在しない
  ので削除は一意に決まる．**置き換えの案内文は書いていない** (一意に決まらず
  P2 の 2 つ目のテストに落ちるため)．
  **新規所見 N11 を起票**: `domain/data/polars_tensor.py` の
  `polars_row_to_hcpe_arrays` (40 行) が Q2 の削除で到達不能になった．
  ユーザが判断したのは `PolarsDataFrameSource` の扱いであって当該モジュール
  ではないので，**黙って広げず行に落とした** (step 4a の scope 境界)．
  **backlog は 2 行 → 2 行**だが，消えたのは 4 run にわたり先頭候補だった行で，
  入ったのは 40 行の contained な残骸である．
  環境: `uv sync --extra cpu` (`UV_HTTP_TIMEOUT=300` が要る)．QA は全て実行
  できた (**2025 passed**，G3 は発生せず)．**無効化テストの手順自体に 1 度
  失敗している** — 各無効化の後始末に `git checkout <file>` を使い，まだ
  commit していない修正ごと HEAD へ戻していたため，2 周目以降は「無効化を
  当てた状態」ではなく「修正が無い状態」を測っていた (落ちる本数が多いので
  気付きにくい)．修正済みファイルを退避してから復元する形に改め，9 通りを
  測り直している．

### Deferred backlog

Findings an audit **confirmed inside** its target path but deliberately
did not fix — ambiguous, cross-layer, architecturally significant, or
needing a decision. A deferred finding is a diagnosis with the fix
withheld pending a decision, **not** a decision never to fix it.

| Found by | Target | Item |
|---|---|---|

_(none — 2026-08-15 の `23drjy` run で Deferred 7 を消化し，deferred backlog は空になった)_

### Out-of-scope backlog

Findings an audit surfaced **outside** the path it was auditing, and was
therefore not allowed to fix.

| Found by | Target | Item |
|---|---|---|
| [2026-08-16 backlog columnar-accessor](2026-08-16-backlog-columnar-accessor.md) N11 | `src/maou/domain/data/polars_tensor.py` | **`polars_row_to_hcpe_arrays` (`polars_tensor.py`，モジュール全体で 40 行) が到達不能になった．** 2026-08-16 の run が `PolarsDataFrameSource` を削除した (ユーザの設計判断) 結果，この関数の呼び出し側は `src/` `tests/` ともに**ゼロ**になった (削除前も唯一の呼び出し側が `polars_datasource.py` だった)．`src/maou/app/common/data_io_service.py:35` に "Direct integration with PyTorch via polars_tensor module" という**docstring の言及だけ**が残っており，これも指す先が無い記述になっている． **この行を起票した理由**: ユーザが 3d で判断したのは `PolarsDataFrameSource` の扱いであって当該モジュールではないので，同じ run で黙って広げなかった (step 4a の scope 境界)． **向きは 2 つ**: (a) モジュールごと削除し `data_io_service.py:35` の docstring も直す (残骸を残さない．`get_items` / `PolarsDataFrameSource` と同じ判断の系列)，(b) 残す (Polars 行 → HCPE 配列の変換は将来使うかもしれない — ただし現在それを使う経路は設計されていない)． **P6** (公開名の削除) / **G4** (2 案のどちらかを人間が選ぶ)．**削除する場合は 40 行 + docstring 1 行で contained**． |
| [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) O9 | `src/maou/infra/bigquery` | `bq_data_source.py:212-234` (記録の `:222-243` から移動) — `sample_ratio` 指定時，`total_rows` は `TABLESAMPLE` 付き `COUNT(*)` から得るが `__fetch_from_bigquery`(**`:395-411`**) はページごとに**別々に引き直す** `TABLESAMPLE` を発行する．ファイル系ソースは常に厳密．`:226` の `num_rows` はテーブルメタデータでストリーミング挿入に遅れる． **2026-08-13 の再検証で記述を強めた** ([記録](2026-08-13-backlog-callback-accumulator-table.md)): 非決定性は**二重**である．(i) `TABLESAMPLE SYSTEM` はクエリごとに独立評価なので数えた行集合と返す行集合が別物，(ii) その上で `LIMIT/OFFSET` を **`ORDER BY` 無しの再サンプル結果**に掛けているので，**同じ `page_num` を 2 回引いても同じ行が返る保証が無い**．「件数がずれる」ではなく**再現性が無い**． **修正の向きが 3 つに割れている** (G4 相当): (a) サンプルを一時テーブルへ 1 度実体化してページングする，(b) `FARM_FINGERPRINT` 等の決定的ハッシュ条件へ置き換える (キー列の決めが要る)，(c) `sample_ratio` とページングの併用を拒否する．**G1**: BigQuery がこの環境に無く，実際の非決定性は fake client では再現できない． **2026-08-13 に再確認** ([記録](2026-08-13-backlog-bundling-knobs-and-loss-aliasing.md)): 行が移動して `__get_total_rows` は `:213-234` (サンプル分岐 `:215-223`)，`total_pages` の導出は `:202-204`，ページ側の再サンプルは `__fetch_from_bigquery` (def `:345`) の `:396-405` (`start_index = page_num * batch_size` `:398`)．クラスタ/パーティション経路も `tablesample_clause` `:380-383` を `:385-389` に差し込むので同じ欠陥を弱い形で持つ．非サンプル経路は `list_rows(start_index=…)` (`:415-419`) なので無傷．**緩和材料**: `get_page` (`:486-520`) がページをキャッシュするため 1 run 内では通常 1 回しか引かれず，非再現性は退避時と run 跨ぎで表面化する． **2026-08-14 にユーザが設計判断を回答: 「決定的ハッシュ条件に置き換える」** (`/audit-backlog` step 3d, [記録](2026-08-14-backlog-cache-knob-removal.md))．3 つの向きのうち **(b)** が選ばれた — `TABLESAMPLE SYSTEM` をやめ，`MOD(FARM_FINGERPRINT(<key>), N) < k` 等の**行に対して決定的な条件**へ置き換える．推奨として提示した (a) 一時テーブルへの実体化は却下された (状態を持たずに済む方を選好)．**G4 は retire** (決めは済んだ)．**G1 は残る** — BigQuery がこの環境に無く，実際の再現性は fake client では確認できない．**残る作業と未解決点**: (i) **キー列の決め**がこの方式の要 — 分布が偏るとサンプルも偏るので，`id` 系の高カーディナリティ列を選ぶ必要がある (現行スキーマに一意キーの保証があるかは未調査)．(ii) `__get_total_rows` (`:213-234`) の COUNT とページ側 (`__fetch_from_bigquery` `:396-405`) が**同じ条件式**を使うようにする — 現在は各々が独立に TABLESAMPLE を発行しているのが二重の非決定性の源．(iii) クラスタ/パーティション経路 (`tablesample_clause` `:380-389`) も同じ差し替えが要る．(iv) `LIMIT/OFFSET` に `ORDER BY` が無い点は条件を決定的にしても残るので，安定な並び順を与える必要がある．**P4** は不変 (同じ呼び出しで返る行が変わる)． **2026-08-14 の 3 巡目にユーザが (i) の未決点に回答: 「行全体のフィンガープリントを使う」** (`/audit-backlog` step 3d, [記録](2026-08-14-backlog-stream-wait-and-abc-direction.md))．`MOD(ABS(FARM_FINGERPRINT(TO_JSON_STRING(t))), N) < k` の形にし，**キー列は選ばない**．却下されたのは (b) `id` 列 (preprocessing の `id` は盤面ハッシュなので同一局面が必ず同じバケットへ落ちてサンプルが偏る) と (c) clustering/partitioning キー優先 (経路ごとに挙動が変わる) と (d) `sample_ratio` とページングの併用拒否 (既存の呼び出しが動かなくなる)．**これで (i) は解決**で，残る作業は (ii) COUNT 側 (`:213-234`) とページ側 (`:396-405`) が**同じ条件式**を使うようにすること，(iii) クラスタ/パーティション経路 (`tablesample_clause` `:380-389`) の同じ差し替え，(iv) `LIMIT/OFFSET` に安定な `ORDER BY` を与えること の 3 点．**G1 は残る** — BigQuery がこの環境に無く，実際の再現性は fake client では確認できない．設計は全て決まったので，**BigQuery に触れる環境があれば通常作業として着手できる**． **2026-08-15 の再検証で行番号を更新** (一律 −1〜−4)．`__get_total_rows` **`:212-233`** (サンプル分岐 `:214-222`)，`total_pages` の導出 **`:201-203`**，`__fetch_from_bigquery` の def **`:344`**，ページ側の再サンプル **`:395-404`** (`start_index = page_num * self.batch_size` は `:397`)，`tablesample_clause` **`:380-382`** とその差し込み **`:384-388`**，非サンプル経路の `list_rows(start_index=…)` **`:412-418`**，`get_page` のキャッシュ **`:485-532`**．欠陥の構造は不変 (COUNT 側とページ側が**別々の** `TABLESAMPLE SYSTEM` を発行し，その上に `ORDER BY` 無しの `LIMIT/OFFSET` を掛けている)． **テスト網羅の実態を確定** (これまで未記載): **`sample_ratio` / `TABLESAMPLE` に触れるテストは `tests/` に 1 件も無い**．実クライアントを叩く `test_bq_data_source.py` は `@pytest.mark.skipif(os.getenv("TEST_GCP") != "true")` で全体が閉じており，"fake" と呼ばれる 2 本 (`test_bq_get_item_contract.py:36` / `test_bq_iter_batches_contract.py:44`) は **BigQuery クライアントを fake していない** — `object.__new__(PageManager)` で組んでから `pm.get_page` を lambda に差し替えており，**まさにこの所見の対象であるサンプリング経路を丸ごと潰している**．`__fetch_from_bigquery` / `__get_total_rows` / `get_page` のキャッシュは**完全に未テスト**である．G1 (BigQuery がこの環境に無い) に加えて，**修正時には回帰テストの土台自体を作る必要がある**ことがこれで確定した．**P4 + G1** は不変． **2026-08-15 の 2 回目の run にユーザがテスト土台の設計判断を回答: 「fake BigQuery client のテスト土台を新設して修正に同梱する」** (`/audit-backlog` step 3d, [記録](2026-08-15-backlog-writeable-contract-and-decisions.md))．却下されたのは「既存の 2 本の "fake" テストを先に独立して作り直してから本題に入る」(run が 2 つに分かれる) と「実 BigQuery 環境が取れるまで着手しない」．**この回答は G1 の意味を実質的に変える** — 決定的ハッシュ条件 (`MOD(ABS(FARM_FINGERPRINT(TO_JSON_STRING(t))), N) < k`) の要点は「**同じ `page_num` を 2 回引くと同じ行が返る**」ことであり，これはクライアントを fake すれば実 BigQuery なしで CI 上で検証できる．**したがって「実 BigQuery が無いと着手できない」は成り立たなくなり，本行は設計・テスト方針とも決定済みの通常作業になった**．ただし**実クエリが BigQuery 上で意図どおり動くことの最終確認は依然として実環境が要る**ので，G1 は「出荷前の実地確認」に縮小した形で残る． **残る作業** (順に): (0) **fake BigQuery client のテスト土台の新設** — 既存の 2 本 (`test_bq_get_item_contract.py:36` / `test_bq_iter_batches_contract.py:44`) は `object.__new__(PageManager)` で組んで `pm.get_page` を lambda に差し替えており対象経路を丸ごと潰しているので，土台はこれとは別に作る，(ii) `__get_total_rows` (`:212-233`) の COUNT 側とページ側 (`__fetch_from_bigquery` `:395-404`) が**同じ条件式**を使うようにする，(iii) クラスタ/パーティション経路 (`tablesample_clause` `:380-382` とその差し込み `:384-388`) の同じ差し替え，(iv) `LIMIT/OFFSET` に安定な `ORDER BY` を与える．**P4** は不変． **2026-08-15 の 3 回目の run で再検証: 行番号の移動なし** (`__get_total_rows` `:212`，`__fetch_from_bigquery` の def `:344`，ページ側の再サンプル `:395-404`，`tablesample_clause` `:380-382` と差し込み `:384-388`，`get_page` `:485` — すべて記録どおり)．**この run では着手していない** — 設計もテスト土台の方針も決定済みで**人間待ちではない**が，(0) の fake client 土台の新設まで含めると今回の枠に入らなかった．**P4 + G1 (縮小済み)** は不変． **2026-08-15 の 4 回目の run にユーザが最後の未定点 (iv) を回答: 「`ORDER BY fingerprint` + `LIMIT/OFFSET`」** (`/audit-backlog` step 3d, [記録](2026-08-15-backlog-gns-sync-and-batch-api.md))．決定的ハッシュ条件で行を絞ったうえで，同じ fingerprint を並び順にも使って安定な全順序を与える．既存のページング形と **`batch_size` が固定という契約** (`total_pages` と `get_page` のキャッシュが前提にしている) がそのまま保たれるので差分が最小で済む．却下されたのは (b) fingerprint のバケット化 (`MOD(fp, total_pages) = page_num`；`ORDER BY` も `OFFSET` も不要で 1 ページのコストが一定になるが，**1 ページの行数が不均一**になり `batch_size` が上限でしかなくなるため `total_pages` とキャッシュの契約を書き直すことになる) と (c) `sample_ratio` とページングの併用拒否 (既存の呼び出しが動かなくなる)． **これで O9 の未定点はゼロになった** — キー (行全体の `FARM_FINGERPRINT(TO_JSON_STRING(t))`)，テスト土台 (fake BigQuery client を新設して同梱)，並び順 (fingerprint) がすべて決定済みである． **残る作業** (順に): (0) fake BigQuery client のテスト土台の新設 — 既存の 2 本 (`test_bq_get_item_contract.py:36` / `test_bq_iter_batches_contract.py:44`) は `object.__new__(PageManager)` で組んで `pm.get_page` を lambda に差し替えており対象経路を丸ごと潰しているので，土台はこれとは別に作る，(ii) `__get_total_rows` (`:212-233`) の COUNT 側とページ側 (`__fetch_from_bigquery` `:395-404`) が**同じ条件式**を使うようにする，(iii) クラスタ/パーティション経路 (`tablesample_clause` `:380-382` とその差し込み `:384-388`) の同じ差し替え，(iv) `LIMIT/OFFSET` に `ORDER BY fingerprint` を与える．**G1 は縮小形のまま残る** — 決定的条件の要点である再現性は fake client で CI に載せられるが，実クエリが BigQuery 上で意図どおり動く最終確認には実環境が要る．**P4** は不変． **2026-08-16 の run で再検証: 行番号の移動なし** (`__get_total_rows` `:212`，`__fetch_from_bigquery` の def `:344`，ページ側の再サンプル `:395-404`，`tablesample_clause` `:380-382` と差し込み `:384-388`，`get_page` `:485` — すべて記録どおり)．欠陥の構造も不変． **この行は現在 backlog に残る唯一の行であり，未定点はゼロである** — キー (行全体の `FARM_FINGERPRINT(TO_JSON_STRING(t))`)，テスト土台 (fake BigQuery client を新設して同梱)，並び順 (`ORDER BY fingerprint`) がすべて決定済みで，**人間待ちではない**．この run で着手しなかったのは (0) の fake client 土台の新設が D13 の枠と同居できなかったからで，それだけの理由である ([記録](2026-08-16-backlog-columnar-accessor.md))．**次 run の唯一の候補**．**P4 + G1 (縮小形)** は不変． |
