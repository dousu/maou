---
kind: backlog
date: 2026-08-13
path:
  - docs/commands/pre_process.md
  - docs/commands/utility_benchmark_training.md
  - docs/commands/utility_benchmark_dataloader.md
  - src/maou/domain/data/columnar_batch.py
  - src/maou/domain/data/arrow_format.py
  - src/maou/infra/file_system/file_data_source.py
  - src/maou/infra/bigquery/bq_data_source.py
level: medium
last_sha: 5bd2754
---

# `/audit-backlog` — bundling ノブの doc drift (O5c)，OOM 見積りの数え落とし (FS-D5)，BigQuery `iter_batches` の契約 (N6-1)，行数スキャンの並列化 (FS-D11(2))

`coverage.md` の backlog 2 表から **16 行** (deferred 8 + out-of-scope 8)
を拾い，HEAD `5bd2754` に対して全件を再検証した．

- **stale: 0** — 16 件すべてが今も成立する
- **changed shape: 0**
- **confirmed: 16**

うち **4 件**を消化して 1 本の PR に載せた．**セッションに指定ブランチ
(`claude/audit-backlog-25b9ao`) の縛りがある**ため，5a の規定により
クラスごとの PR を 1 本に畳んでいる — レビュー単位はコミットが担う．
その結果，自動帯 (P2/P3) も判断帯 (P4，および G1 付き P3) と同じ PR に
乗るので，自動帯だけを先にマージすることはできない．

**この run はユーザに何も聞いていない．** 5c の分割テスト (枝が実質
別の diff を生み，外すと作業が無駄になる) に該当する項目が無かった —
消化した 4 件はいずれも修正の形が 1 通りに定まる．

## Classification

判断コスト P1-P6 + ゲートで分類した．`P6 → P1` の順に評価し，最初に
引っ掛かったクラスを採る．

### 消化した 4 件

| ID | backlog 行 | 対象 | クラス | そのクラスに決めたテスト | ゲート |
|---|---|---|---|---|---|
| P2-1 | O5 **(c)** | `docs/commands/` × 3 | **P2** | 変更は `.md` のみ．訂正文は「受け取るが効果なし」以外に書きようがない | なし |
| P3-1 | FS-D5 (見積り部分) | `domain/data` + `infra/file_system` | **P3** | 受理する全入力に対し結合結果・返り値・成果物が不変．差は警告の発火有無だけ | run 中に **G2 が発生** (下記) |
| P4-1 | N6-1 | `infra/bigquery` | **P4** | 落ちていた経路が動くようになる = 観測可能な挙動変化．既存データは読めるまま，CLI 契約も不変なので P5/P6 ではない | **G1 を撤回** (下記) |
| P3-2 | FS-D10+D11 **(2)** | `domain/data` | **P3** | 同じ件数・同じ順・部分結果なしを保つ．差は所要時間とログ順だけ | **G1 が残る** → 判断帯 |

**P4-1 の G1 撤回の理由．** 行は「BigQuery 実環境が無いと確認できない」
と書いていたが，同じ backlog から出た O1 の修正
(`tests/maou/infra/bigquery/test_bq_get_item_contract.py`) が
`PageManager` を `object.__new__` で作り `get_page` だけ差し替える
やり方を既に確立していた．同じ手で `iter_batches` も検証できるので，
G1 は撤回できる．**行が挙げるゲートは，その行が書かれた時点の手段の
限界であって，恒久的な性質ではない** — 前例が増えると撤回できる．

**P3-2 の G1 が残る理由．** 正しさ (件数・順序・例外・メモリ profile)
はこの環境で固定できるが，**便益**はネットワーク越しに数百ファイルを
置いた環境でしか測れない．G1 の「production data が要る」に当たる．
正しさが示せても効果が示せない変更は自動帯に入れない．

### 残した 12 件のクラスとゲート

| ID | backlog 行 | クラス | ゲート | 残す理由 |
|---|---|---|---|---|
| D-L2 | app/learning Deferred 2 | P4 | **G3** | `multi_stage_training.py:422`/`:571`，`stage_component_factory.py:646`/`:735` は健在．~400 行のリファクタで，この環境で等価性を示せない |
| D-L3 | app/learning Deferred 3 | P6 | **G3** | 6 adapter クラス健在．公開名 6 本が消えるうえテストが参照している |
| D-L4 | app/learning Deferred 4 | P4 | **G3** | `callbacks.py` の `_ensure_device` 参照 14 箇所．基底抽出 ~250→~120 行 |
| D-L5 | app/learning Deferred 5 | P4 | **G1** | `training_loop.py:1110` 健在．dormant のまま (mask を供給する経路が無い)．GPU で測らないと直せない |
| D-L6 | app/learning Deferred 6 | P4 | **G1** | `training_loop.py:460` 健在．GPU semantics の変更で D-L5 と一緒に実機検証が要る |
| D-L7 | app/learning Deferred 7 | P4 | **G1** | `gradient_noise_scale.py:150/189/192/247` の `.item()` 健在．GNS は adaptive batch を駆動するので数値等価性の確認が要る |
| FS-D5 (本体) | infra/file_system D5 | P6 | **G4** | ノブ廃止は O5 と一体という行自身の 見送り理由 が生きている．見積り部分だけ切り出して消化した |
| FS-D10/D11 (1) | infra/file_system | P4 | **G4** | `FileDataSource.total_pages()` は production caller ゼロで dormant．「ファイル数」と「yield 数」のどちらを意味させるかの決めが要る |
| FS-D13 | infra/file_system D13 | P4 | **G2** | 根本解決は `app/learning/dataset.py` と ABC を触る．(b) 単独は caller ゼロで dormant |
| FS-D14 (b) | infra/file_system D14 | P6 | **G2** | `file_data_source.py:49` が 2 つの ABC を着ている．外すと `infra/utility/benchmark_polars_io.py` の対応が要る |
| FS-D15 | infra/file_system D15 | P4 | **G4** | `path_utils.py:28` は末尾拡張子判定のまま．size/footer 検査を全ファイルに掛けるかは運用リスクの実在の判断 |
| O5 (本体) | infra/console + object_storage | P6 | **G4** | (a)(b)(d) は健在．bool flag と dir のどちらがキャッシュを有効にするのかの決めが要る |
| O9 | infra/bigquery | P4 | **G1** | `TABLESAMPLE` の不整合はコード上健在．同じ行集合が返らないことの確認に BigQuery 実環境が要る |
| N4 | tests/infra/file_system | P1 | **G4** | 行が名指しする 2 案 (薄いテストへ切り出す / CPU extra 必須) が未決のまま．**この run で 4 回目の実害を確認** (下記 Environment notes) |
| N6-2 | app/pre_process | P4 | **G4** | `hcpe_transform.py:118` の HCPE 決め打ちは健在．override 側は全て正しく分岐しているので production は dormant．改名か分岐かの決めが要る |

(15 行に見えるのは FS-D5 と FS-D10+D11 が「一部消化・一部残し」で
両方の表に現れるため．backlog 行としては 12 行が残る．)

## Consumed

| ID | 由来 | 対象 | 出荷したもの | コミット |
|---|---|---|---|---|
| P2-1 | O5 (c) | `docs/commands/` × 3 | bundling ノブが現状 no-op である旨を明記 | `a1ce41c` |
| P3-1 | FS-D5 (見積り部分) | `domain/data/columnar_batch.py`, `infra/file_system/file_data_source.py` | `ColumnarBatch.nbytes` を `dataclasses.fields` から導出．`_warn_if_oom_risk` に警告を一本化，閾値を `OOM_WARNING_THRESHOLD_GB` に定数化 | `62f39a9` |
| P4-1 | N6-1 | `infra/bigquery/bq_data_source.py` | `iter_batches` を structured array に，`iter_batches_df` を override．変換を `PageManager.to_structured_array` に一本化 | `11834e4` |
| P3-2 | FS-D10+D11 (2) | `domain/data/arrow_format.py` | File 形式のメタデータ読みだけを `ThreadPoolExecutor` に．Stream 形式は逐次のまま (ピークメモリ保護) | 下記 PR の最終コミット |

いずれも **backlog 行は残る** — 4 件とも行の一部だけを消化したか
(P2-1/P3-1/P3-2)，PR が未マージ (全件) だからである．6a の規定により
行の削除はマージ後．

## Applied

- `docs/commands/pre_process.md:23`,
  `docs/commands/utility_benchmark_training.md:34`,
  `docs/commands/utility_benchmark_dataloader.md:21` — `a1ce41c`
- `src/maou/domain/data/columnar_batch.py:62` (`nbytes` プロパティ新設) — `62f39a9`
- `src/maou/infra/file_system/file_data_source.py:44` (`OOM_WARNING_THRESHOLD_GB`),
  `:311` (`_warn_if_oom_risk`), `:328`/`:363` (2 つの結合経路) — `62f39a9`
- `src/maou/infra/bigquery/bq_data_source.py:573` (`to_structured_array`),
  `:691` (`iter_batches`), `:713` (`iter_batches_df`) — `11834e4`
- `src/maou/domain/data/arrow_format.py:67`/`:85`/`:104` (2 相分割),
  `:129` (`scan_row_counts` の並列化) — P3-2 コミット

バージョン: `0.89.4` → `0.89.5` (P3-1) → `0.89.6` (P4-1) → `0.89.7` (P3-2)．
P2-1 は `src/` を触らないので bump なし．

## In flight

**PR は 1 本のみ** (指定ブランチの縛り)．base は `main`．

| 項目 | 内容 |
|---|---|
| ブランチ | `claude/audit-backlog-25b9ao` |
| base | `main` |
| 未決の判断 | **P4-1**: BigQuery 入力の pre-process が動くようになる (今まで `AttributeError` で落ちていた)．**P3-2**: 便益が未測定の並列化を入れるかどうか |
| レビュー単位 | コミット 4 本 (docs / P3-1 / P4-1 / P3-2) |

自動帯 (P2-1, P3-1) はこの PR の下 2 コミットに入っているが，
1 本にまとまっている以上，単独では `main` に入れられない．
判断帯を落としたい場合はコミット単位で外す必要がある — これが
クラスごとに PR を分ける通常運用の利点であり，指定ブランチの
縛りで失ったものである．

## Re-triaged

上表「残した 12 件」がそれ．今回の再検証で**行の記述を鋭くできたもの**
だけを挙げる:

- **FS-D5** — 行は「ノブ廃止は O5 と一体」とだけ書いていたが，
  見積りの数え落とし部分は **O5 と独立に直せる**ことが分かった．
  消化済みなので，残りは「ノブ (cache_mode) を廃止するか」だけになる．
- **FS-D10+D11** — (2) を消化したので，残るのは (1) の意味の決めだけ．
  行を (1) の記述に縮めた．
- **N6-1 / N6-2 の関係** — N6-1 を直すにあたり `iter_batches_df` を
  override したので，**BigQuery は基底の HCPE 決め打ち既定実装を
  もう通らない**．N6-2 の実害範囲がさらに狭まった (production の
  caller は全て override 側を通るので完全に dormant)．N6-2 を直す
  動機は「将来の実装者が既定実装を踏む」ことだけになった．
- **N4** — 「実害を 3 run 連続で確認」から **4 run 連続**になった．
  この run も `infra/file_system` に触れており，`--extra cpu` を
  入れずに回すと変更が無検証のまま緑に見える状態だった．
- **FS-D15** — この run で分かった関連: 行数スキャン
  (`scan_row_count`) は全ファイルの Arrow footer を読むので，
  途中書きの `.feather` は**そこで既に落ちる** (行が引く
  `OSError: failed to fill whole buffer` がまさにそれ)．つまり
  「捕捉できていない」のではなく「分かりにくい形で落ちている」．
  D15 の判断は「検査を足すか」ではなく「この失敗をどこで
  分かりやすくするか」に寄せられる．

## Corrections to the source records

なし．今回再検証した 16 行について，記録の診断・提案する修正が
**誤っていた**ものは見つからなかった (16 件とも confirmed)．

## Doc findings

- `reviews/2026-08-13-bundling-knobs-are-no-ops.md` — **applied**．
  CLAUDE.md § "Standing approval — drift corrections only" の恒久承認で
  適用した (P2)．訂正文は `object_storage/data_source.py:199` の
  docstring と `:288-320` の実装から一意に決まる．
  ノブ自体の削除と既定値の不一致 (`:45` `True` / `:102`,`:394` `False`)
  は O5 の判断と一体なので提案から除いてある．

## Out of scope

**新規 1 件 — この run 中に修正済み (行は作らない)．**

`tests/maou/infra/utility/test_benchmark_polars_io.py:30` の `capsys`
に型注釈が無く，`mypy` が `no-untyped-def` で落ちていた．HEAD
(`5bd2754`) から存在する既存エラーで，`.pre-commit-config.yaml` の
mypy hook は `pass_filenames: false` + `args: ["src/", "tests/"]`
なので，**Python を 1 行でも触るコミットが全て pre-commit で落ちる**
状態だった．注釈は pytest の API から一意に決まる
(`pytest.CaptureFixture[str]`) ので G2 として取り込み，`62f39a9` で
修正した．

**行は作った (N7)．** 「解決済みだから不要」と一度は考えたが，解決して
いるのは**この PR の中**だけで，PR が閉じられれば `main` には残る．
これは in-flight の 4 行とまったく同じ状態なので，同じ扱いにする —
行を置き，PR をリンクし，マージされたら消す．

なぜ HEAD に入り得たのかは追っていない — `e7c5d3e` /
`d0c4984` のどちらかが mypy の差分キャッシュをすり抜けた可能性がある．
再発するようなら hook 側の問題として別途見る．

## Environment notes

- **base の venv が空だった．** `uv sync --extra cpu` に約 7 分．
  N4 の実害がそのまま出ており，これを入れずに回すと
  `tests/maou/infra/file_system/` が丸ごと skip される．
  この run は 4 件中 2 件が `infra/file_system` に触れているので，
  入れずに QA を回していたら**変更が無検証のまま緑に見えていた**．
  N4 の 4 回目の実害確認．
- **`maou_rust` が未ビルドだった．** `uv run maturin develop --release`
  に 12 分．`domain/data/rust_io` 経由で `.feather` を書くテストに要る．
- **pre-commit が毎コミットで全テストを回す．** 1905 passed /
  54 skipped，約 5 分 43 秒．Python を触るコミット 1 本あたり ~6 分．
  今回は 4 コミットなので QA だけで ~25 分．
- **`mypy` hook は `src/` と `tests/` の 296 ファイルを見る．**
  `mypy src/` (135 ファイル) だけを回して緑を確認するのは不十分で，
  `tests/` 側のエラーでコミットが落ちる (上記 Out of scope)．
- **撤回できなかった G1**: GPU 無し (D-L5/D-L6/D-L7)，BigQuery 実環境
  無し (O9)，数百ファイルのネットワークストレージ無し (P3-2 の便益)．
- **撤回できなかった G3**: `app/learning` の ~250-400 行リファクタ
  (D-L2/D-L3/D-L4) は，等価性を示す手立てがこの環境に無い．
