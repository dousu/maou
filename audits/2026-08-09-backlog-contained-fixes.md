---
path: src/maou/app/learning, src/maou/domain/model, src/maou/domain/data, src/maou/domain/game_graph
kind: backlog
scope: python
level: medium
status: done
started: 2026-08-09
last_sha: cdc4031
---

# Audit — backlog 消化，contained fixes

**これは path 監査ではない．** `audits/coverage.md` の Deferred /
Out-of-scope backlog から個別の finding 5 行 (作業単位で 4 件) だけを
消化した記録である．対象パスはどれも今回まとめて監査したわけでは
ないので，ledger のメインテーブルに `done` 行は書かない．

## Resume point

_(complete — no resume point)_

選択された 4 件はすべて resolved．QA は全て実行済みで pass．

## Consumed

| # | Source record | Target | 消化 |
|---|---|---|---|
| T1-1 | [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) "Smaller confirmed items" | `src/maou/app/learning` | `886eea2` |
| T3-1 | 同 Out-of-scope 3 + Deferred 11 (2 行) | `src/maou/domain/model`, `src/maou/app/learning` | `da8eabe` |
| T3-2 | 同 Out-of-scope 4 | `src/maou/domain/data` | `1c714db` |
| T3-3 | [2026-08-08 game_graph](2026-08-08-src-maou-domain-game-graph.md) Deferred 1 | `src/maou/domain/game_graph` | `1cff5be` |

backlog 19 行 (deferred 13 + out-of-scope 6) 全件を HEAD (`cdc4031`)
に対して再検証したうえで 6 段階に並べ，ユーザが T1 全部と
T3-1/T3-2/T3-3 を選択した．T3-1〜T3-3 は判断待ち項目だったので，
コード変更の前に個別に判断を取った (下記)．T4 (大規模リファクタ) と
T5 (GPU 実機必須) は今回触らない選択．

**取った判断.**

- T3-1: `forward_features` も含めて全部埋める (protocol 宣言のみに
  留めない)．
- T3-2: `polars_tensor.py` の 4 関数は削除する．
- T3-3: `schema.py` の空 DataFrame ヘルパは削除する (テストごと)．

## Applied

### `886eea2` — T1-1 (`pyproject.toml` 0.82.6 → 0.82.7, patch)

1 行に束ねられていた 5 件をすべて修正．

- `dl.py:1178` — `_resolve_trainable_layers()` をメモ化．`learn()`
  (`:296`) と学習設定ログ生成 (`:1034`) の 2 箇所から呼ばれるため，
  `:1189` の警告が 1 回の実行で 2 回出ていた．解決結果が `None` でも
  「未解決」と区別できるよう 1 要素タプルのセル
  (`_trainable_layers_cell`) で保持し，素の計算は
  `_compute_trainable_layers()` へ分離．
- `dl.py:609` — 直後 (`:627`) で上書きされる `epoch_number = 0` を削除．
- `callbacks.py` — 検証バッチごとに同じ logits へ `log_softmax` が
  3 回・`topk` が 2 回かかっていたのを各 1 回に．
  `_policy_cross_entropy_gpu` は `log_probs` を受け取る
  `_cross_entropy_from_log_probs` へ置き換え，
  `_compute_policy_top5_accuracy_stats_gpu` /
  `_compute_policy_f1_components_gpu` は `prediction_top_indices` を
  省略可能引数で受け取る (省略時は従来どおり内部計算するので既存の
  テスト呼び出しはそのまま動く)．
- `setup.py:216` — `_estimate_per_worker_mb` が全データファイルを
  `stat()` していたのを，等間隔サンプル `_SIZE_SAMPLE_LIMIT = 64` 件
  までに制限 (`_sample_for_size_estimate`)．ログの件数表記も
  「サンプル数 / 全件」に直した．
- `setup.py:734` — `ModelFactory` が `HeadlessNetwork` / `Network` の
  既定値 (`board_vocab_size`, `embedding_dim`, `board_size`, `block`,
  `layers`, `strides`, `out_channels`, `num_policy_classes`) を再掲
  していたのをやめ，ネットワーク側を唯一の出所にした．未使用に
  なった import 4 件も削除．

### `da8eabe` — T3-1 (`pyproject.toml` 0.82.7 → 0.83.0, minor)

- `domain/model/protocol.py` — `FreezableBackbone` に
  `preprocess_for_blocks` と `forward_features` を宣言．
- `domain/model/resnet.py` — `ResNet.forward_features` を追加．
  ViT / MLP-Mixer と意味を揃えてプーリング済み特徴ベクトルを返す．
  プーリングは `pooling` 引数で受け取る．
- `domain/model/mlp_mixer.py` — `forward_features` に `@overload` を
  付け，`return_tokens=False` なら `Tensor` を返すことを型で示した．
- `app/learning/network.py` — `forward_features` のアーキテクチャ
  分岐と `getattr` プローブ + `RuntimeError` を削除．前処理を公開
  メソッド `embed_inputs()` として切り出し，`backbone_input_channels`
  と `board_size` プロパティを追加．
- `app/learning/multi_stage_training.py:399` — 写経していた前処理と
  private 参照 6 種を公開 API へ置き換え．

### `1c714db` — T3-2 (`pyproject.toml` 0.83.0 → 0.83.1, patch)

`domain/data/polars_tensor.py` から `polars_row_to_preprocessing_tensors`
/ `polars_row_to_stage1_tensors` / `polars_row_to_stage2_tensors` /
`dataframe_to_tensor_batch` を削除．生きている
`polars_row_to_hcpe_arrays` (`polars_datasource.py:15-17` が import)
のみ残る．未使用になった import とモジュール docstring も整理．

### `1cff5be` — T3-3 (`pyproject.toml` 0.83.1 → 0.83.2, patch)

`domain/game_graph/schema.py` から `_create_empty_df` /
`create_empty_nodes_df` / `create_empty_edges_df` を削除．
`tests/maou/domain/game_graph/test_schema.py` の該当 4 メソッドと
import も同時に削除した．スキーマ定義とその形・型のテストは残る．

## 修正が record の示唆と違ったところ

**T3-1 の `forward_features`.** record は「`DomainResNet` に
`forward_features` がない」とだけ書いていたが，素直に追加すると
`HeadlessNetwork` の `pooling` 拡張ポイントが黙って死ぬ．resnet 経路
だけがプーリングを `self.pool` で行っており，プーリングの持ち主を
`ResNet` へ移すと `pooling` 引数が無視されるようになるからである．
そこで `ResNet.__init__` が `pooling` を受け取り，`HeadlessNetwork` は
**同じモジュールオブジェクト**を `self.pool` と backbone の両方から
参照させる形にした (`AdaptiveAvgPool2d` はパラメータを持たないので
`state_dict` のキーは増えず，チェックポイント互換性も保たれる)．
この落とし穴自体を
`test_resnet_forward_features_uses_injected_pooling` と
`test_custom_pooling_is_still_applied_for_resnet` で固定した．

**T3-1 の protocol 宣言.** `forward_features` をそのまま protocol へ
書くと MLP-Mixer が適合しない — ViT が `(x) -> Tensor` なのに対し
Mixer は `(x, token_mask=None, *, return_tokens=False) -> Tensor |
tuple[Tensor, Tensor]` で，戻り値型が `Tensor` の subtype でない．
`@overload` で `return_tokens: Literal[False]` のとき `Tensor` を
返すことを示して解決した．protocol の戻り値型を広げると
呼び出し側の narrowing が壊れるので，そちらは採らなかった．

## Corrections to the source records

`audits/2026-08-08-src-maou-app-learning.md` に `## Corrections` 節を
追記した．4 件とも「診断か示唆した修正が誤り」であって worklist の
状態ではない:

1. **Deferred 1 の因果が誤り** — Stage2 `__len__` の過大評価が
   `dl.py:318,498` の scheduler へ届くとしているが，`dl.py` の loader は
   kif 経路でその `__len__` は正しい．実際の経路は
   `stage_component_factory.py:713,803`．
2. **Deferred 9 / Out-of-scope 4 の根拠が偽** —
   「`docs/rust-backend.md:704` が公開 API として文書化している」は
   同ファイルを検索して 0 ヒット．削除を止めていた根拠が事実無根
   だった．今回の削除判断はこの再検証に基づく．
3. **Deferred 10 / Out-of-scope 2 の callee 名が 1 ホップずれ** —
   実際の呼び先は `create_stage2_streaming_components`．さらに
   `stage2_test_ratio` は非 streaming 経路では現に使われているので，
   素直に消すと検証分割が壊れる．
4. **Out-of-scope 3 のパスが存在しない** —
   `src/maou/domain/model/network.py` はなく，実体は
   `src/maou/app/learning/network.py` (行番号は偶然一致)．

教訓としては前 run と同じで，**record は「どこを見るか」には信頼
できるが「何をするか」には信頼できない**．今回は加えて，
**record が「やらない理由」として挙げた根拠自体を検証する**必要が
あることが分かった (2 番は，検証しなければこの項目は永久に
「消せない」ままだった)．

## Re-triaged

選択した 4 件はすべて resolved．一方，**選択しなかった行のうち 4 行は
再検証で内容が変わった**ので `coverage.md` 側の文面を鋭くした
(行は削除していない):

- **app/learning Deferred 1** — 上記 correction 1 を反映．加えて
  `_resolve_worker_files` が shuffle するため正確な件数は seed 依存
  であること，`_compute_total_batches` は 3 データセット共用なので
  その場で書き換えてはいけないことを追記．
- **app/learning Deferred 9** — `1c714db` で
  `polars_tensor.py` の tensor ヘルパを削除したため，**record が
  示唆していた修正 (「既存ヘルパへ切り替える」) は選べなくなった**．
  代替案 (スキーマから直接組む / `dataset.py` に非 numpy 経路を作る)
  を書き添えた．
- **app/learning Out-of-scope 2 (`stage2_test_ratio`)** — 上記
  correction 3 を反映し，触ってよい範囲を streaming の 3 シグネチャに
  限定する旨と，`or 0.2` による `0.0` の握り潰しを追記．
- **game_graph Deferred 2 (`find_opening` の前提条件)** — interface 側の
  ガードが回帰テスト済みかつ文書化済みなので，ドメイン API を変えると
  既存ガードの重複になる．won't-do で閉じる選択肢を明示した．
- **game_graph Out-of-scope (`openings.py` 無文書)** — 利用者向け挙動は
  `3600b32` で `docs/commands/visualize.md:183-186` に着地済み．
  残作業はモジュール / API doc と定跡一覧に縮小した．
- **game_graph Out-of-scope (`query.py`)** — 行番号を `174-200` に更新．
  再検証でも到達可能な入力は構成できず，**unconfirmed のまま**．

## Doc findings

**なし．** 今回の 4 件はいずれも durable doc を無効化しなかった．
確認方法: `forward_features` / `FreezableBackbone` /
`preprocess_for_blocks` / `create_empty_*_df` / `polars_tensor` /
`polars_row_to` / `dataframe_to_tensor_batch` /
`_resolve_trainable_layers` / `ModelFactory` を `docs/` `CLAUDE.md`
`AGENTS.md` `README.md` に対して検索．唯一の一致
`docs/stage2-speed-investigation.md:126`
(`HeadlessNetwork.forward_features()`) は変更後も正しいままなので
`reviews/` 提案は出していない．

## Out of scope

この run が新たに見つけた backlog 対象の finding はない．
上記 Re-triaged はいずれも既存行の文面更新であって新規項目ではない．

## Environment notes

**QA 状況** (すべて `.venv` のツールを直接実行):

| チェック | 結果 |
|---|---|
| `ruff format src/ tests/` | **pass** (286 files) |
| `ruff check src/ tests/` | **pass** |
| `mypy src/` | **pass** (134 files) |
| `pytest` 全体 | **1744 passed, 54 skipped** (110.7s) |

**回帰テストの非空虚性を実証済み** (修正を無効化して失敗を確認し復元):

- メモ化を外すと `test_resolve_trainable_layers_warns_only_once` が
  警告 2 回で失敗する．
- `log_probs` / `topk` の使い回しを 1 箇所でも戻すと
  `test_log_softmax_and_topk_run_once_per_batch` が `2 == 1` で失敗する．
- サンプリングを外すと `test_estimate_per_worker_mb_bounds_stat_calls`
  が `256 <= 64` で失敗する．
- `ResNet.forward_features` が注入された `pooling` を使わないように
  すると `test_resnet_forward_features_uses_injected_pooling` と
  `test_custom_pooling_is_still_applied_for_resnet` が失敗する．

**回帰テストを書いていない項目とその理由.**

- `dl.py:609` の死んだ代入削除 — 挙動が変わらないため固定すべき
  失敗モードがない．
- T3-2 / T3-3 の削除 — 「消した関数が戻ってこないこと」を主張する
  テストは，正当な新規追加でも落ちるので有害．削除の妥当性は
  「全テストが通る = どこからも import されていなかった」で担保
  されている．
- `callbacks.py` の等価性テスト
  (`test_expected_win_rate_matches_independent_softmax` など) は
  **非空虚性の実証対象ではない** — 共有化は値を変えない変更なので，
  戻しても通り続ける．欠陥そのものを固定しているのは上記の
  呼び出し回数テストのほう．

**環境.** `uv run` はこのコンテナで使用不可．
`uv sync --extra cpu --no-install-project` の後
`maturin develop --release` で `maou._rust` をビルドしてから QA を
実施した (前 run の 31 分に対し今回は短縮; ビルドキャッシュの差)．
`.git/hooks/pre-commit` はこのコンテナに未インストールなので commit
時にフックは走っていない．同等のチェック (`ruff-format`,
`ruff-check`, `mypy`, `test`) は上表のとおり手動で全て実行済み．
