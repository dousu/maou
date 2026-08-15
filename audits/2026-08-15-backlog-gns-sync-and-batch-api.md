---
kind: backlog
date: 2026-08-15
path:
  - src/maou/app/learning/gradient_noise_scale.py
  - src/maou/infra/file_system/file_data_source.py
scope: python
level: medium
last_sha: 4cd0731
record_sha: 380866c
---

# backlog consumption — GNS の同期除去と，棄却済みバッチ取得 API の残骸

`/audit-backlog` (2026-08-15, `23drjy`)．前 run
([2026-08-15 stage-unification-and-gns-interval](2026-08-15-backlog-stage-unification-and-gns-interval.md))
が 2 行を消して残した **3 行**で始まった run である．

3 行はいずれも過去の設計判断で「人間待ちではないただの作業」に
なっており，**G4 も未回答の設計判断もゼロ**という状態からの開始
だった．本 run はそのうち **1 行を全消化，1 行を部分消化**し，
残る 2 行それぞれについて**先送りの真因になっていた未定点を
1 問ずつ潰した**．

**本 run の実質は「ゲートの本文を読み直すと外れることがある」の
2 例目**である．前 run が Deferred 5 の G1 を「経路が到達不能なら
測る対象が無い」と崩したのに続き，本 run は Deferred 7 の G1 を
「数値等価性は GPU でなく **CPU 上の厳密一致テスト**で示せる」と
崩した．3 run 連続で「GPU が無いから」と塞がれていた行が，
ゲートの前提を読み直しただけで同 run 内の出荷まで到達している．

指定ブランチ 1 本の制約でクラス毎の PR 分割ができず，自動帯・
判断帯が同じ PR ([PR #507](https://github.com/dousu/maou/pull/507))
に同居したため，**レビュー単位は commit が担った**．

## Classification

3 行を再検証して **stale 0 / changed shape 0 / confirmed 3**．
**自動帯は空** (16 run 連続)．

| ID | 由来の行 | クラス | クラスを決めたテスト | ゲート | 転帰 |
|---|---|---|---|---|---|
| **B-1** | Deferred 7 | **P4** | 返す値は bit 一致するので素直には P3 だが，「あらゆる device / dtype で一致する」は**論拠であって測定ではない**．訓練クリティカルな値 (`gradient_accumulation_steps` を書き換える) なので fail-safe の向きに倒した | **G1 を retire** | 出荷 (`b5b4457`) |
| **B-2** | D13 の (3) の一部 | **P6** | `FileDataSource` の公開名が 2 つ消える．呼び出し側はゼロだが公開 API の削除は P6 | なし | 出荷 (`380866c`) |
| **B-3** | O9 | P4 | 同じ呼び出しで返る行が変わる | **G1** (縮小済み) | 設計判断のみ (Q3) |
| **B-4** | D13 の (2)(4) | P4/P6 | `app/learning/dataset.py` と ABC に触る | **G2** + 設計の穴 | 設計判断のみ (Q2) |

### B-1 の G1 retire の根拠 (本 run で最も重要な判断)

ゲートの本文は「device スカラーへの蓄積は値が materialize する
タイミングを変え，その値が `gradient_accumulation_steps` を書き換える
ので，**数値等価性の確認が必要**」だった．3 run にわたりこれが
「GPU が無いので確認できない」と読まれ，行は塞がれ続けた．

この読みは誤りである．旧実装の `acc += x.item()` は
**Python の float すなわち float64 の逐次加算**なので，
累算器を **float64 の device スカラー**にして同じ順序で加算すれば，
加算順序も dtype 変換も一致し結果は **bit 単位で同一**になる．
そしてその一致は，旧算術をそのまま写した参照実装との突き合わせで
**CPU 上で厳密に検証できる**．

GPU が要るのは「同期が減って**速くなった**」ことの計測だけである．
**同期が起きないこと自体**は前 run の B-3 と同じ手 (`Tensor.item` /
`Tensor.tolist` を数える) で CPU 上から直接観測できる．

したがって「correctness cannot be established in this environment」
という G1 の定義に本行は当てはまらない．**retire**．

なお **float64 でなければならない**ことは無効化テストで確認した —
累算器を float32 に落とすと厳密一致テストが落ちる (実測値
`35.7270622253418` vs 参照 float64 値)．この 1 行が classification
そのものを支えているので，`_accumulate_device_scalar` の docstring に
理由を明記した．

## Consumed

| 行 | 由来の記録 | Target | 出荷したもの | PR |
|---|---|---|---|---|
| **Deferred 7** (全消化 → **行を削除**) | [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) | `src/maou/app/learning/gradient_noise_scale.py` | 勾配統計を device 上の float64 スカラーへ蓄積し，host への materialize を `compute()` の 1 回に集約 | [#507](https://github.com/dousu/maou/pull/507) |
| **D13 の (3) の一部** (行は残る) | [2026-08-10 infra/file_system](2026-08-10-src-maou-infra-file-system.md) | `src/maou/infra/file_system/file_data_source.py` | 棄却済みバッチ取得 API の残骸 `get_items` を 2 箇所から削除 | [#507](https://github.com/dousu/maou/pull/507) |

## Applied

### `b5b4457` — GNS の per-parameter host-device 同期の除去

- `gradient_noise_scale.py:52-79` — `_accumulate_device_scalar` を新設．
  device 上の float64 0 次元テンソルへ加算する．**float64 が要**で
  ある理由を docstring に書いた．
- `gradient_noise_scale.py:134` — `_sum_micro_norm_sq` を
  `float` から `torch.Tensor | None` へ．
- `gradient_noise_scale.py:184-201` / `:215-249` —
  `on_backward_end` の 2 つの累算ループ (第 1 マイクロバッチ経路と
  差分経路) から `.item()` を除去．
- `gradient_noise_scale.py:276-312` — `compute()` の
  `mean_grad_norm_sq` も device 蓄積にし，**S と G を
  `torch.stack` して 1 回の `tolist()` で materialize** する．
  これが cycle 中で唯一の host-device 同期になる．
- モジュール docstring に "host-device 同期" 節を追加．
  クラス docstring の `measurement_interval` の説明から
  「パラメータテンソルごとに `.item()` による host-device 同期が
  走る」の一文を落とした (前 run が既定を 5 に上げた理由の半分が
  本 commit で消えたため — **既定 5 を戻してはいない**．残る理由
  である「モデルパラメータ 1 コピー分の追加メモリ」は健在)．

**同期回数**: 計測サイクルあたり
*(パラメータ数 × micro-batch 数 + パラメータ数)* 回 → **1 回**．
ResNet/ViT backbone (60-300 パラメータテンソル) なら micro-batch
あたり 60-300 回の同期が消える．

**回帰テスト 4 本** (`tests/maou/app/learning/test_gradient_noise_scale.py`):

1. `test_bit_identical_to_python_float_reference` — 旧算術を写した
   `_PythonFloatReferenceEstimator` との **厳密一致** (`==`，許容誤差
   なし) を 3 cycle × 4 micro-batch で確認．
2. `test_bit_identical_with_half_precision_gradients` — float16 勾配で
   同じ確認 (`.to(float64)` と `.item()` の等価性)．torch は param と
   grad の dtype 一致を強制するのでモデルごと `.half()` にしている．
3. `test_no_host_sync_during_backward_hook` — `on_backward_end` が
   `Tensor.item` / `Tensor.tolist` を**一度も呼ばない**．
4. `test_compute_syncs_exactly_once` — `compute` の同期が **1 回**．

参照実装を本体と共有していないのが要点で，共有すると「同じコードは
同じ答えを出す」ことしか言えなくなる．その旨をクラス docstring に
書いた．

**無効化テスト**: 累算器を float32 にすると (1) が落ちる．
パラメータ毎の `.item()` を戻すと (3)(4) が落ちる．

### `380866c` — 棄却済みバッチ取得 API の残骸 `get_items` の削除

- `file_data_source.py` — `FileManager.get_items` (旧 `:392-405`) と
  `FileDataSource.get_items` (旧 `:584-595`) を削除．

**残骸である根拠**:
`docs/adr-003-training-performance-optimization-attempts.md` §5 が
このバッチ取得 API を **"❌ FAILED - REVERTED"** として記録している
(バッチ時間 **+115%** / スループット **-38%**)．ところが revert が
不完全で，PyTorch が実際に呼ぶ `__getitems__` だけが消え，
何の効果も無い包み紙の `get_items` が残った — 「バッチで取得する」と
称しながら中身は `get_item` を要素ごとに呼ぶ Python ループである．
呼び出し側は `src/` `tests/` `rust/` のいずれにもゼロ．

**ADR-003 は編集していない**．過去形で「試して棄却した」ことを
記録した歴史的文書であり，残骸を消すことは記述を偽にしない
(むしろ記録どおりの状態に近づける)．doc drift は発生していない．

**回帰テスト 1 本**
(`tests/maou/infra/file_system/test_file_data_source.py`):
`test_batch_retrieval_api_stays_removed` — `get_items` と
`__getitems__` の**どちらも生えていないこと**，および要素ごとの
取得が従来どおり動くことを固定する．**不在そのものを固定**して
いるのは，同じ形が名前だけ変えて戻るのを防ぐためである．
無効化テスト: 削除を戻すと落ちる．

### バージョン

`0.93.1` → **`0.93.2`** (`perf:` = patch，B-1) → **`0.94.0`**
(`feat!:` = 破壊的変更，B-2)．

破壊的変更を **major ではなく minor** にしたのはこのリポジトリの
確立した慣例に従ったもので，`fc6e968` (`feat!: file_level_split を
削除する` — 公開 API 削除という点で本 run の B-2 と同型) /
`d0c4984` / `bdda7b5` / `232358e` / `1c91fea` がいずれも 0.x で
minor bump している．

**`uv.lock` の付随修復**: `main` の時点で `uv.lock` の `maou` の
version が **`0.92.2`** のままで `pyproject.toml` (`0.93.1`) から
ずれていた．`uv lock` で追随させてある．前 run が `uv.lock` を
更新せずにバージョンを上げたためと思われるが，`pre-commit` の
`uv-lock` hook は通っていたので**検出されない種類のずれ**である．

## Decisions asked

`AskUserQuestion` は **受理 1 問 + 設計判断 2 問**．3 件とも回答を得た．

### Q1 (受理，multiSelect) — PR #507 のどの修正をマージするか

| 選択肢 | 回答 |
|---|---|
| `b5b4457` GNS の同期除去 (推奨) | **選択** |
| `380866c` `get_items` の削除 (推奨) | **選択** |

両方採られたので PR は所定のままマージした．片方だけの場合に
落とす側の commit を除いて force-push する旨を明示したうえでの
回答である．

### Q2 (設計判断) — D13 (2): `KifDataset` はどの口から `ColumnarBatch` に届くか

**これが D13 が 3 run 連続で先送りされた真因**である．「`KifDataset`
が `ColumnarBatch` を直接スライスする」という設計自体は 2026-08-14 に
決まっていたが，学習側の `DataSource` ABC (`dataset.py:45-66`) は
`__getitem__` と `__len__` の**2 メソッドしか無く**，列に届く口が
決まっていなかった．過去 3 run はこの穴に気付かないまま
「G2 の作業量が枠に入らない」とだけ書いていた．

| 選択肢 | 回答 |
|---|---|
| **(a) ABC に列アクセサを追加** — `LearningDataSource` に列単位アクセサを足し，`FileDataSource` が実装．BigQuery / ObjectStorage は既定実装で structured 経路へフォールバック | **採用** |
| (b) `KifDataset` が `hasattr` で duck-type 検出 | 却下 |
| (c) streaming 経路へ一本化 | 却下 |
| (d) D13 (2)(4) を落とす | 却下 |

(a) は契約が型に出るので実装漏れを構築時に捕まえられ，
`b652d5e` (`fix: DataSource基底をabc.ABCにして未実装を構築時に
捕まえる`) の方針と揃う．(b) は差分が最小だが契約が型に出ないので
将来の実装が黙って遅い経路に落ちても気付けない．

**settles**: D13 の (2)(3)(4) すべて — 口が決まれば
`FileManager.get_item` の縮退範囲も hcpe 経路との分岐も従属する．
**fix は本 run では書いていない** (G2 の作業量が B-1 と同居できない)．
**G2 は retire していない** — 設計の回答は結合の制約を動かさない．

### Q3 (設計判断) — O9 (iv): 決定的ハッシュ化後のページングの安定化

| 選択肢 | 回答 |
|---|---|
| **(a) `ORDER BY fingerprint` + `LIMIT/OFFSET`** — 既存のページング形と `batch_size` 固定の契約を保つ | **採用** |
| (b) fingerprint のバケット化 (`MOD(fp, total_pages) = page_num`) | 却下 |
| (c) `sample_ratio` とページングの併用を拒否 | 却下 |

(b) は `ORDER BY` も `OFFSET` も要らず 1 ページのコストが一定になる
利点があるが，**1 ページの行数が不均一**になり `batch_size` が上限
でしかなくなるため，`total_pages` と `get_page` のキャッシュが前提に
している契約を書き直すことになる．(a) は毎ページのソートというコストを
払う代わりに差分が最小で済む．

**settles**: O9 の残る未定点はこれで**ゼロ**．キー (行全体の
`FARM_FINGERPRINT(TO_JSON_STRING(t))`)，テスト土台 (fake BigQuery
client を新設して同梱)，並び順 (fingerprint) がすべて決まった．
**fix は本 run では書いていない** (fake client 土台の新設まで含めると
枠に入らない)．**G1 は縮小した形のまま残る** — 実クエリが BigQuery 上で
意図どおり動く最終確認には実環境が要る．

### 予算に入らなかった設計判断

**なし**．本 run の開始時点で G4 の行はゼロであり，設計の穴として
新たに見つかったのが Q2 と Q3 の 2 件だけだったので，4 問の枠に
対して 3 問で足りた．次 run が引き継ぐ設計判断の待ち行列は空である．

## In flight

**なし**．3d の質問は同一セッション内で全問回答を得ており，
PR #507 はその場でマージした．

## Re-triaged

| 行 | 研ぎ直した理由 |
|---|---|
| **D13** | (3) の一部を消化して行を縮めたうえで，**Q2 の決定 (ABC に列アクセサを追加) を書き込んだ**．過去 3 run が「G2 の作業量が入らない」とだけ書いていたのに対し，本 run は**先送りの真因が ABC の口の未定にあった**ことを特定して塞いだ．残るのは (2)(4) と `get_item` の縮退で，**G2 のみが障害**である |
| **O9** | **Q3 の決定 (`ORDER BY fingerprint`) を書き込んだ**．これで仕様の未定点はゼロになり，残るのは (0) fake client 土台の新設と (ii)(iii)(iv) の実装だけになった．**G1 は縮小形のまま** |

## Corrections to the source records

**なし**．再検証は 3 行とも confirmed で，記録の診断・処方に誤りは
見つからなかった (行番号のずれのみ — B-1 が `.item()` `:151`/`:190`/
`:193`/`:248`，`g.clone()` `:152`/`:194` で記録から一律 +1，これは
前 run の `57f0664` が `measurement_interval` の既定行を複数行に
整形したため)．

**ただし本 run 自身が発見した，記録ではなく台帳の性質の問題**:
Deferred 7 は「G1 が塞いでいる」として 3 run 生き延びたが，
G1 の本文を読み直すだけで外れた．**ゲートは書かれた時点の環境と
実装案に対する判定であって，恒久的な属性ではない** — 再検証が
行番号だけでなく**ゲートの前提そのもの**に及ぶべきことを，
前 run の Deferred 5 に続いて 2 例目として確認した．

## Doc findings

**なし**．`reviews/` 提案は起票していない．

- B-1 は `src/` の docstring のみを触っており，durable doc ではない
  (CLAUDE.md の分類どおり，docstring 編集は P3 以上でバージョン
  bump を伴う扱い)．
- B-2 が触れる `docs/adr-003-training-performance-optimization-attempts.md`
  は**過去形の歴史的記録**であり，棄却された実験の残骸を消すことで
  偽にはならない (上記 Applied 参照)．
- `docs/commands/` に影響する CLI オプションの変更は無く，
  `check-cli-docs` hook も Passed．

## Out of scope

**新規の out-of-scope 行は無し**．

再検証中に気付いたが行を起票しなかったもの:

- **`uv.lock` の version が `pyproject.toml` から静かにずれる** —
  `main` 時点で `0.92.2` vs `0.93.1` だった．`pre-commit` の
  `uv-lock` hook は通るので検出されない．本 run で追随させたので
  現時点の実害は無く，恒久対策 (hook の強化) は
  `.pre-commit-config.yaml` の話で本 backlog の対象外．
  再発するようなら起票する．

## Environment notes

- `uv sync --extra cpu` に **`UV_HTTP_TIMEOUT=300` が必要** (既定の
  30 秒では torch の依存取得がタイムアウトする)．前 run の記述どおり．
- **git の pre-commit hook がこのコンテナには入っていなかった**
  (`.git/hooks/pre-commit` が無い)．`uv run pre-commit run
  --from-ref/--to-ref` で 2 commit 分を明示的に実行し，全 hook が
  Passed であることを確認した (`test` / `mypy` / `check-cli-docs` /
  `uv-lock` / ruff を含む)．初回は hook 環境の構築で 2 分以上かかる．
- **G3 は発生せず** — QA は全て実行できた．
  `pytest` 全体で **2024 passed, 53 skipped**．
  `mypy src/` は 135 source files で Success．
- 収集されなかったモジュールが 1 つある
  (`tests/maou/infra/visualization/test_indexing_status.py` —
  `gradio` 未導入)．本 run の変更とは無関係．
- **GPU は無い**ので，B-1 の同期除去による実際の速度向上は
  測っていない．正しさの確認には GPU は不要 (上記 G1 retire の根拠)．
