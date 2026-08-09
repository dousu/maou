---
path: src/maou/app/game_graph, src/maou/app/learning, src/maou/app/utility
kind: backlog
scope: python, docs
level: medium
status: done
started: 2026-08-09
last_sha: a1daeb8
---

# Audit — backlog 消化，T3 (契約が絡む 3 件)

**これは path 監査ではない．** `audits/coverage.md` の Deferred /
Out-of-scope backlog から個別の finding 3 件だけを消化した記録である．
対象パスはどれもまとめて監査したわけではないので，ledger のメイン
テーブルに `done` 行は書かない．

backlog 9 行 (deferred 8 + out-of-scope 1) 全件を HEAD (`b2cf8e8`) に
対して再検証した．`ff5bbaa` 以降 `src/` に変更が無かったため
**stale 0 件 / confirmed 9 件**，ただし **T3-3 は record の診断自体が
誤っていた** (下の Corrections)．T1・T2 は空だった — 直近 3 回の
backlog run が利用者から見える欠陥と agent 向け文書のドリフトを
消化し終えたため．ユーザは T3 全部を選択．T4 (大規模リファクタ 3 件)
と T5 (GPU 実機必須 3 件) は今回触らない．

## Resume point

_(complete — no resume point)_

選択された 3 件はすべて resolved．

## Consumed

| # | Source record | Target | 消化 |
|---|---|---|---|
| T3-1 | [2026-08-08 game_graph](2026-08-08-src-maou-domain-game-graph.md) Out-of-scope 2 | `src/maou/app/game_graph` | `bdcb9f1` (`pyproject.toml` 0.83.5 → 0.83.6, patch) |
| T3-2 | [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 8 | `src/maou/app/learning`, `src/maou/app/utility` | `568863f` (`pyproject.toml` 0.83.6 → 0.83.7, patch) |
| T3-3 | 同 Deferred 9 | `src/maou/app/learning` | `156aa96` (`pyproject.toml` 0.83.7 → 0.83.8, patch) |

**取った判断.** T3 はいずれもコード変更前に個別に判断を取った．

- T3-1: 壊れた親チェーンでは `ValueError` を投げる (部分パスを
  返さない)．`[]` を返す案・docstring を弱める案・won't-do 案は却下．
- T3-2: マスクを targets タプルから完全に外す
  (「device 上で 1 回作る」案 = record の示唆した修正は却下)．
- T3-3: スキーマ由来の dtype に切り替え + 死んだ numpy 模倣を削除
  (`dataset.py` に非 numpy 経路を作る案は T4 相当として却下)．

## Applied

### `bdcb9f1` — T3-1

- `src/maou/app/game_graph/query.py:187-223` — `get_path_to_root` の
  3 つの `break` のうち，**正常終了である `node_depth == 0` 以外の
  3 条件**を `ValueError` に変えた (親が nodes_df に無い /
  depth>0 なのに入力エッジが無い / depth-1 に親が居ない)．
  docstring に `Raises:` と「先頭は必ず depth 0 のルート」を明記．
- `tests/maou/app/game_graph/test_query.py` — 回帰テスト 6 件と
  `_build_broken_chain_tree()` (孤児ノードと「親が同じ depth」の
  2 種類の壊れ方を持つグラフ)．部分パスを返さないこと，例外時に
  キャッシュを汚さないこと，健全な枝が従来どおりであることを固定．

**到達可能性について.** ビルダの出力からこの入力は作れない (3 回の
試行で構築できず)．したがってこれは実害の出ているバグではなく
型/契約のギャップであり，**将来ビルダが壊れたときに黙って誤った
定跡名や SFEN を出す代わりに落ちる**ようにする変更である．
interface 層の 5 箇所 (`game_graph_visualization.py:443,648,697,944,1011`)
はこの例外を捕捉しない — ユーザはその代償を承知の上で選択した．

### `568863f` — T3-2

`legal_move_mask` を targets タプルから外した．この経路が作れる
マスクは 2 箇所とも無条件の `torch.ones_like(moveLabel)` で，
消費側の 5 つのカーネルすべてで no-op でありながら，バッチ毎に
moveLabel と同じサイズ (B=1024 で約 9MB) を PCIe 上に流していた．

- `dataset.py:124` / `streaming_dataset.py:830` — `torch.ones_like`
  を削除．targets は `(move_label, result_value)` ，moveWinRate が
  あれば 3 要素目に入る．
- `training_loop.py:493-511` — **`_unpack_batch` を同時に直した．**
  これがこの変更の罠で，データ側だけ直すと `targets[2]` は依然
  mask として読まれ，moveWinRate の float 値がそのまま
  `masked_fill` に渡って損失が静かに壊れる．index 2 は
  `move_win_rate`，`legal_move_mask` は常に `None`．
- `callbacks.py:506-519` — `policy_move_label_ce` の
  `legal_move_mask is not None` ゲートを外した．外さないと，全経路が
  マスクの供給をやめた瞬間にこの指標が黙って消える．
  `normalize_policy_targets` はマスク無しでも `policy_log_probs` と
  同じ確率空間に正規化するので，値は全 1 マスクの場合と一致する
  (テストで固定)．
- `dataloader_benchmark.py:147-158` — 3 要素固定の分解をやめ，
  3 要素目を `move_win_rate` として転送する．
- `multi_stage_training.py` / `streaming_dataset.py` の Stage1/Stage2
  アダプタ docstring — 3 要素目の名前を更新．**コードは変えていない**:
  これらは `(targets, dummy_value, None)` を返しており，新しい契約でも
  index 2 = `move_win_rate` = `None` で正しい．

**マスク機構は残した．** `TrainingContext.legal_move_mask` と
`_compute_policy_loss` のマスク分岐は，本物の合法手マスクを流す経路が
将来できたときのために残してある．消したのは「常に全 1 のデータを
毎バッチ運ぶ」ことだけである．

**副次的に Deferred 5 が休眠した.** `training_loop.py:1110` の
`if not has_legal.all():` (バッチ毎の host-device sync) は
マスク分岐の中にあるため，もう実行されない．コードは変えていない
ので backlog 行は**削除せず**「dormant, not fixed」に書き換えた．

### `156aa96` — T3-3

- `polars_datasource.py` — `_leaf_numpy_dtype()` を追加し，
  `PolarsDataFrameSource` が `dataframe.schema` から列ごとの numpy
  dtype を導出して `_PolarsRow` → `_PolarsField` へ渡す．変換表は
  書かず，polars 自身に空 Series を `to_numpy()` させて問い合わせる
  ので polars の型が増えても追従不要．`pl.List` / `pl.Array` は
  入れ子を剥がして最内を見る．
- 同 — `_PolarsField.flags` を削除 (下の Corrections 参照)．
  スカラの `dtype` もスキーマ由来の値を返すようになった
  (Python float からの推測では float64 になっていたが，スキーマ上は
  Float32)．
- `tests/maou/app/learning/test_polars_datasource.py` — **新規**．
  この経路にはテストが 1 件も無かった．罠のテスト
  (`List(UInt16)` の列で値 300 が 44 に折り返さないこと) と，
  `flags` を持たないこと，ガードが本物の numpy の flags を見ることを
  固定．`KifDataset` との結合テストは T3-2 の新しい targets 契約も
  同時に押さえる．

**現時点では潜在的な欠陥である.** 全列の推測がたまたまスキーマと
一致していた (`List(List(UInt8))`→uint8，`List(Float32)`→float32 等)．
実害は将来 `List(UInt16)` 等の列が入ったときに出る — そのとき値は
256 で折り返り，`KifDataset` の dtype ガードは期待どおりの uint8 を
受け取るので**何も言わない**．

## Re-triaged

なし (選択された 3 件はすべて resolved)．

ただし **Deferred 5 の行は書き換えた**．削除していない: コードは
変えておらず，本物の合法手マスクを配線した時点で同じ stall が戻る．
行には「dormant, not fixed」と，record の前提
(「Stage 3 always ships a `legal_move_mask`」) がもう成り立たないことを
記した．

## Corrections to the source records

`audits/2026-08-08-src-maou-app-learning.md` の `## Corrections` に
2 件追記した．

1. **Deferred 9 —「FakeFlags は zero-copy ガードを通すためにある」が
   誤り．** そのガードは `_PolarsField.flags` を一度も読まない．
   `_numpy_to_tensor` は先頭で `np.asarray(array)` を呼び
   (`dataset.py:197`)，`__array__` 経由で本物の numpy 配列を作ってから
   その配列の `.flags` を見る (`:216` / `:222`)．リポジトリ全体で
   `_PolarsField.flags` の読み手は 0 件だった．
   **学習点**: 「X は Y を通すために存在する」という記述は，X の
   読み手を実際に検索して確かめない限り信用できない．この record は
   同じ誤りを Deferred 9 で 2 度している (前回 run が訂正した
   「`docs/rust-backend.md` が文書化しているから消せない」も同型)．
2. **Deferred 8 の示唆した修正 (device 上で 1 回作る) は次善．**
   マスクを供給する経路が 1 つも無い以上，外す方が転送もカーネルも
   消える．加えて record は `_unpack_batch` の**位置ベース**の読み出し
   に触れておらず，示唆どおりデータ側だけ直すと損失が静かに壊れる．

なお `PolarsDataFrameSource` 自体は production から呼ばれておらず
テストも無かったが，`docs/rust-backend.md:740` が使用例つきで文書化
している公開 API なので削除は提案していない (`polars_tensor.py` の
4 関数で文書化の主張が偽だったのとは対照的に，こちらは実在する)．

## Doc findings

`reviews/2026-08-09-legal-move-mask-removed-from-targets.md` —
T3-2 が 3 つの文書を無効化した．いずれも**修正前の動作を正確に
説明していた**文書である．

- `docs/loss-functions.md` — 「Stage 3 に供給される `legal_move_mask`
  は全要素 1 のダミー」がまさに今回消した `torch.ones_like` を指す
- `docs/rust-backend.md:771` — `PolarsDataFrameSource` の唯一の
  使用例で，targets を 4 要素に分解している (読者がそのまま
  コピーすると `move_win_rate` をマスクとして受け取る)
- `docs/adr-001-...md:127` — ADR なので本文は書き換えず注記のみ提案

status: **applied** (`03b61ad`；提案 `be363d0`，status 更新 `a1daeb8`)

## Out of scope

この run で新たに気づいた事項．`coverage.md` の backlog にも
追加していないもの (追加が必要なら下記のとおり)．

- なし．`PolarsDataFrameSource` のテスト欠如は T3-3 の一部として
  この run で解消した．

## QA

すべて実行し pass．

| 対象 | 結果 |
|---|---|
| `ruff format src/ tests/` | 3 files reformatted |
| `ruff check src/ tests/ --fix` | All checks passed |
| `mypy src/` | Success: no issues found in 134 source files |
| `pytest tests/maou/app/ tests/maou/interface/` | 1107 passed, 2 skipped |
| `pytest tests/` (残り) | 671 passed, 52 skipped |

**非空虚性を確認済み**．3 件それぞれについて修正を無効化して
テストが落ちることを確認し，元に戻した．

- T3-1: 3 つの `raise` を `break` に戻す → 4 件 fail
- T3-2: `_unpack_batch` を位置ベースの読み出しに戻す +
  `callbacks.py` のゲートを復活 → 3 件 fail
- T3-3: `np_dtype` 分岐を無効化 → 1 件 fail
  (numpy が `the cast overflows` を警告し，罠が実在することを確認)

## Environment notes

- **`uv sync --extra cpu` に長時間を要した**．base venv に torch も
  polars も入っておらず，`maturin` が Rust workspace 全体を
  ビルドし直す必要があった (`maou_rust` の `py-ext` プロファイル)．
  QA はこのビルド完了後に実行した．
- **GPU は無い** (`nvidia-smi` 無し，`torch.cuda` 未検証)．T3-2 の
  動機である PCIe 転送削減そのものは**この環境では測定できていない**．
  変更が挙動を変えないこと (全 1 マスクと `None` で
  `policy_move_label_ce` が一致すること等) はテストで固定したが，
  速度改善は未測定である．
- 同じ理由で T5 の 3 件 (Deferred 5/6/7) はここでは消化できない．
