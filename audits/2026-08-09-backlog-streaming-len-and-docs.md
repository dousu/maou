---
path: src/maou/app/learning, src/maou/interface, src/maou/domain/game_graph, docs/architecture.md, docs/commands/visualize.md
kind: backlog
scope: python, docs
level: medium
status: done
started: 2026-08-09
last_sha: 12b02d4
---

# Audit — backlog 消化，Stage2 `__len__` と game_graph 文書

**これは path 監査ではない．** `audits/coverage.md` の Deferred /
Out-of-scope backlog から個別の finding 5 行 (作業単位で 4 件) だけを
消化した記録である．対象パスはどれもまとめて監査したわけではないので，
ledger のメインテーブルに `done` 行は書かない．

## Resume point

_(complete — no resume point)_

選択された 4 件はすべて resolved．QA は全て実行済みで pass．

## Consumed

| # | Source record | Target | 消化 |
|---|---|---|---|
| T1-1 | [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 1 | `src/maou/app/learning` | `51eadfa` |
| T3-1 | 同 Out-of-scope 2 (= Deferred 10) | `src/maou/interface`, `src/maou/app/learning` | `d41cafd` |
| T3-2 | [2026-08-08 game_graph](2026-08-08-src-maou-domain-game-graph.md) Deferred 2 | `src/maou/domain/game_graph` | `e5428df` |
| T6-1 | 同 Out-of-scope 4 | `docs/commands/visualize.md` | `843344b` |
| T6-2 | 同 Out-of-scope 5 | `docs/architecture.md` | `843344b` |

backlog 13 行 (deferred 9 + out-of-scope 4) 全件を HEAD (`2e54fd4`) に
対して再検証し，6 段階に並べた．**stale は 0 件，changed shape が 2 件**
(T1-1 と T3-1)．ユーザは T1-1 + T3-1 + T3-2 + T6 の doc 2 件を選択．
T4 (大規模リファクタ 3 件) と T5 (GPU 実機必須 4 件) は今回触らない．

**取った判断.**

- T1-1: `__len__` を `__iter__` に合わせて shard-aware にする
  (グループのみ / scheduler 側で直す / 文書化のみ の 3 案は却下)．
- T3-1: ストリーミングの 2 シグネチャから `test_ratio` を削除する
  (分割を実装する / 受け取るが未使用と文書化する は却下)．
- T3-2: ドメイン API は変えず won't-do で閉じる (ただし前提条件は
  docstring に明記する)．
- T6: 提案 `reviews/2026-08-09-game-graph-architecture-and-openings-docs.md`
  を両方承認．

## Applied

### `e5428df` — T3-2 (`pyproject.toml` 0.83.2 → 0.83.3, patch)

`domain/game_graph/openings.py:115` `find_opening` の docstring に
`Important:` 節を追加し，(a) `moves` は平手初期局面からの手順である
という前提条件，(b) 本メソッドはそれを検査しない (手順だけからは
起点が分からない)，(c) 担保は interface 層の `_root_is_startpos()` の
責務であること，を明記した．

**API シグネチャは変えていない．** 症状は `cc10790` で interface 側に
修正済みで回帰テストも文書も揃っているため，ドメイン側に初期 SFEN
引数やガードを足すと出荷済みガードの二重化になる．

`tests/maou/domain/game_graph/test_openings.py` に
`test_origin_precondition_is_not_enforced_here` を追加．
`find_opening(["5g5f"])` が起点に関わらず「先手中飛車」を返すことを
固定し，「ここでは検査しない」という設計をロックする．

### `51eadfa` — T1-1 (`pyproject.toml` 0.83.3 → 0.83.4, patch)

`app/learning/streaming_dataset.py`:

- `_compute_concat_total_batches()` を新設．`__iter__` と同じ
  ラウンドロビン worker 分割 (`row_counts[shard::n_shards]`) と
  `_FILES_PER_CONCAT` 単位の結合グループ構造を再現して数える．
- `StreamingStage2Dataset.set_num_workers()` を追加し，`__len__` を
  新ヘルパへ切り替えた (`:602` → 現 `:670`)．

`app/learning/stage_component_factory.py`:

- `create_stage2_streaming_data_pipeline` が **DataLoader 構築後に**
  `raw_dataset.set_num_workers(train_dataloader.num_workers)` を呼ぶ．

3 データセット共用の `_compute_total_batches()` は**変更していない**．
kif (`StreamingKifDataset.__len__`) と Stage1
(`StreamingStage1Dataset.__len__`) は結合しないので現状が正しい．

### `d41cafd` — T3-1 (`pyproject.toml` 0.83.4 → 0.83.5, patch)

`create_stage2_streaming_data_pipeline` と
`create_stage2_streaming_components` から `test_ratio` 引数 (と
docstring の記載，転送) を削除し，`interface/learn.py:848` の転送も
やめた．`run_stage2_streaming` の `stage2_test_ratio` 引数と
`:829` の「ストリーミングでは無視する」警告は**残した** — CLI から値が
来る事実は変わらず，警告は運用者への唯一の通知だからである．

### `843344b` — T6-1 / T6-2 (doc のみ，版上げなし)

`reviews/2026-08-09-game-graph-architecture-and-openings-docs.md`
(承認済み，`ff5bbaa` で `status: applied` + `applied_in: 843344b`) を
そのまま適用．

- `docs/architecture.md`: `## Game Graph サブシステム` を `## domain` の
  直後に新設．2 レイヤ (`domain/game_graph` / `app/game_graph`) の
  責務表，依存方向，`find_opening` の平手前提とその担保が interface 層
  にあること，2 つの CLI doc へのリンク．
- `docs/commands/visualize.md`: 既存の定跡説明の下に
  `#### サポートしている定跡` を追加．9 エントリを 8 行の表で列挙
  (矢倉は 2 変化を 1 行に)，唯一の定義元が `_DEFAULT_OPENINGS` である
  ことを明記，表に無い戦型は「定跡でない」ではなく「未収録」と説明．

## 修正が record の示唆と違ったところ

**T1-1 — 「`num_workers` が要る」で渡すべき値は要求値ではない．**
record は「正確な件数には `num_workers` が必要で，どうモデル化するかは
設計判断」として保留していた．素直に読むと呼び出し側の
`dataloader_workers` を dataset へ渡す形になるが，それは誤りである．
`DataLoaderFactory.create_streaming_dataloaders()` は
`_clamp_workers()` で `min(要求, ファイル数, メモリ由来上限)` に
切り下げる (`setup.py:474-476`) ので，切り下げが起きた実行で
バッチ数を読み違える．正しい入力は DataLoader 構築後の
`train_dataloader.num_workers` である．

この罠を `test_stage2_streaming_len_uses_effective_worker_count` で
固定した．24 ファイル / 100 行 / batch 64，要求 20・メモリ上限 2 の
条件で 3 通りの値が**すべて別**になるよう数を選んである:

| 実装 | `len(loader)` |
|---|---|
| 未配線 (`_num_workers` が 0 のまま) | 39 |
| **正しい実装 (実効 2)** | **40** |
| 要求値 20 を使う | 48 |

**T1-1 — 「seed 依存だから判断が必要」は保留の理由にならなかった．**
シャッフルで変わるのは各結合グループの**行数合計**だけで，shard あたりの
ファイル数とグループの区切り方は一定である．したがって `__iter__` の
構造を再現する修正自体は判断待ちにする必要がなく，残る近似
(グループ行数の seed 依存) は `_compute_concat_total_batches` の
docstring に Note として明記した．ファイル間で行数が等しい場合
(生成されたシャードでは通常そう) は厳密である．

**T3-1 — 削除に寄せる決定的な根拠は record に無かった．**
`create_stage1_streaming_data_pipeline` には `test_ratio` 引数が
**そもそも存在しない**．つまり Stage2 側のこれは「設計されたが未実装」
ではなく写経の残骸であり，これが分かって初めて「削除」が
「実装」より明らかに正しい選択になった．record にはこの対称性への
言及がない．

**T3-1 — 罠のテストは既に存在していたので追加しなかった．**
非ストリーミング Stage 2 が `test_ratio` を使い続けることは
`test_no_validation_when_ratio_zero` /
`test_has_validation_when_ratio_positive`
(`test_stage_component_factory.py:240`/`:250`) が既に固定している．
同じ主張のテストを重ねるのはノイズなので書いていない．

## Corrections to the source records

`audits/2026-08-08-src-maou-app-learning.md` の `## Corrections` に
1 件追記した (`12b02d4`)．**Deferred 1 が示唆した修正の入力が誤り**で
あることと，保留理由 (seed 依存) が実際には判断を要さなかったこと．

前 run の教訓 (「record は『どこを見るか』には信頼できるが『何をするか』
には信頼できない」) が今回も成立した．今回さらに分かったのは，
**record が「保留の理由」として挙げた設計判断そのものを疑う**必要が
あることである．T1-1 は 2 回の run で「設計判断が必要」として見送られて
いたが，判断が必要なのは残差の近似だけで，構造の再現には判断が
要らなかった．「判断待ち」というラベル自体が検証対象である．

## Re-triaged

選択した 5 行はすべて resolved (T3-2 は won't-do 側での決着だが，
判断が済んだので open ではない — 行は削除した)．

選択しなかった 8 行は削除せず，再検証で分かったことを `coverage.md`
側に反映して鋭くした:

- **app/learning Deferred 2 / 3 / 4** — 行番号が `da8eabe` /
  `886eea2` でずれていたので HEAD で再検証して更新した
  (`multi_stage_training.py:436`→`:422`, `:585`→`:571`;
  `stage_component_factory.py:636`→`:646`, `:724`→`:735`,
  `:866-872`→`:876-882`; `callbacks.py` の `_ensure_device`
  `:1007`→`:1044`, `:1396`→`:1433`, `:1521`→`:1558`, `:1668`→`:1705`;
  streaming adapters `:645`→`:721`, `:610`→`:686`)．
  `callbacks.py` の accumulator 3 箇所は行番号ではなくクラス名
  (`Stage2F1Callback` / `Stage1AccuracyCallback` / `Stage3LossCallback`)
  で書き直した — 行番号は毎 run ずれるがクラス名はずれない．
- **app/learning Deferred 5 / 7 / 8 / 9** — 同様に行番号更新
  (`training_loop.py:1093`→`:1100`;
  `gradient_noise_scale.py:188-192,246`→`:189-192,247`;
  `dataset.py:91`→`:124`, `streaming_dataset.py:754`→`:830`,
  `callbacks.py:493`→`:509`;
  `polars_datasource.py:204-266`→`:205-268`)．内容は不変．
- **game_graph Out-of-scope (`query.py`)** — 3 回目の再検証でも到達
  可能な入力は構成できず **unconfirmed のまま**．ただし堂々巡りを
  避けるため，必要な判断を明文化した: docstring が
  「ルートから対象ノードまでの」を約束しているのに壊れた親鎖では
  それを返せないので契約上のギャップは実在する．よって
  「防御的に直す (深さ 0 に届かず打ち切ったら `[]` か例外)」か
  「docstring を弱めて部分パスがあり得ると書く」かの二択である．
  また call site は 7 ではなく **5** だった (行番号 5 個の列挙に
  「Seven」と書かれていた)．

## Doc findings

`reviews/2026-08-09-game-graph-architecture-and-openings-docs.md` —
2 ファイル，**承認され `843344b` で適用済み**，frontmatter は
`ff5bbaa` で `status: applied` に更新．

ソース修正による durable doc の無効化は**なかった**．確認方法:
`stage2_test_ratio` / `stage2-test-ratio` / `test_ratio` /
`find_opening` / `OpeningDatabase` / `__len__` / `steps_per_epoch` を
`docs/` `CLAUDE.md` `AGENTS.md` `README.md` に対して検索．
一致したのは以下で，いずれも修正後も正しい:

- `docs/commands/learn_model.md:92` — `--stage2-test-ratio` を
  「ストリーミングモードでは未対応」と既に正しく書いている
  (T3-1 は挙動を変えていないので追随不要)．
- `docs/loss-functions.md:87-91`，`docs/learning-rate-tuning.md:140`，
  `docs/commands/utility_benchmark_training.md:52` — いずれも
  非ストリーミング経路の話で，そちらは無変更．

## Out of scope

この run が新たに見つけた backlog 対象の finding はない．
上記 Re-triaged はいずれも既存行の文面・行番号更新であって新規項目では
ない．

なお `uv.lock` の `maou` バージョンが `0.82.6` のまま `pyproject.toml`
(`0.83.2`) と乖離していた．`uv lock` を各版上げで実行して同期させ，
今回の 3 コミットに含めた (backlog 項目ではなく作業中の付随修正)．

## Environment notes

**QA 状況** (すべて `.venv` のツールを直接実行):

| チェック | 結果 |
|---|---|
| `ruff format src/ tests/` | **pass** (286 files) |
| `ruff check src/ tests/` | **pass** |
| `mypy src/` | **pass** (134 files) |
| `pytest` 対象 3 ファイル | **91 passed** |
| `pytest tests/maou/{app/learning,interface,domain/game_graph}` | **727 passed, 1 skipped** |
| `pytest` 全体 (最終コミット状態) | **1755 passed, 54 skipped** (113s) |

**回帰テストの非空虚性を実証済み** (修正を無効化して失敗を確認し復元):

- `__len__` をファイル単位の `_compute_total_batches` に戻すと
  `test_len` / `test_len_matches_actual_iteration` /
  `test_set_num_workers_changes_len` /
  `test_stage2_streaming_len_uses_effective_worker_count` の
  **4 件が失敗**する．
- `set_num_workers` の呼び出しを消すと
  `test_stage2_streaming_len_uses_effective_worker_count` が
  `39 == 40` で失敗する．
- 同じ箇所で `dataloader_workers` (要求値) を渡すと
  `48 == 40` で失敗する．**未配線・要求値・正解の 3 状態が
  すべて区別される**ことをこれで確認した．
- `find_opening` に `if len(moves) < 2: return None` という
  もっともらしいガードを入れると
  `test_origin_precondition_is_not_enforced_here` だけが失敗し，
  他の openings テストは通り続ける．

**注意**: 上の 4 番目を確認した後 `git checkout` でファイルを戻した
ため，同じコミット前に docstring 修正まで巻き戻していた．
再適用して commit したので最終状態は正しいが，非空虚性の実証で
`git checkout <file>` を使うと未コミットの正規の修正も消える．
以後は個別 revert かバックアップ経由で戻すこと．

**既存テストが欠陥を固定していた．** `test_len` は修正前の
過大評価値 (3 ファイル / 10 行 / batch 7 で 6) を assert しており，
実際の反復は 5 バッチだった．つまり欠陥はテスト付きで出荷されていた．
`test_len_matches_actual_iteration` を `len(dataset) == len(list(dataset))`
の形で追加し，同種の乖離が二度と通らないようにした．

**環境.** `uv run` はこのコンテナで使用不可．
`uv sync --extra cpu --no-install-project` の後
`maturin develop --release` で `maou._rust` をビルドしてから QA を
実施 (約 25 分)．1 回目の `uv sync` は site-packages が空のまま
成功扱いで終わり `.venv/bin/maturin` が無かったので，再実行が必要
だった (同じ症状が出たら sync をもう一度回すこと)．
`patchelf` 未インストールの rpath 警告は無害．
`.git/hooks/pre-commit` はこのコンテナに未インストールなので commit
時にフックは走っていない．同等のチェック (`ruff-format`,
`ruff-check`, `mypy`, `test`, `uv-lock`) は上表のとおり手動で全て
実行済み．
