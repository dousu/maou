---
status: pending
applied_in:
date: 2026-07-27
target:
  - docs/design/usi-engine/verification.md
  - docs/commands/search.md
  - docs/commands/selfplay.md
  - docs/commands/usi.md
risk: low
reversibility: trivial
---

# 提案: TensorRT teardown の heap 破壊を「TRT 固有」で確定し，終了時回避を docs へ反映する

## Trigger

`corrupted double-linked list` の切り分けが完了した．

| 構成 | 再現 |
|---|---|
| `--tensorrt --cuda` | **3/3 (決定的)** |
| `--cuda` のみ | **0/3** |

⇒ **TensorRT EP 固有**で確定．CUDA EP も同じ provider 共有ライブラリ機構で
動的ロードされるのに壊れないので，「静的 ORT コア + 動的 provider の二重 ORT」
説は**否定**された．全出力の後 (プロセス終了時) に起きるため数値・棋譜は有効
だが，`abort()` = SIGABRT なので **GUI/対局サーバからクラッシュと見なされ得る**．

TRT を捨てる選択も検討したが，同一 session 内の連続測定で判断した:

| regime | TRT+CUDA | CUDA のみ | 判定 |
|---|---|---|---|
| 単発 30 秒 (バッチが埋まる) | **4,595 p/s** | 2,901 p/s | TRT が **1.58 倍**速い |
| 自己対局 800 playouts/手 (埋まらない) | 2,153.9 p/s | **2,292.9 p/s** | CUDA が 6.5% 速い |

TRT は固定 shape のため毎バッチを `batch_size` へ padding する実装
(`pad_to = batch_size`)．短い探索では padding が無駄になるが，**バッチが埋まる
regime では TRT が明確に速い**ので，実配置 (1 手数千 playouts) と aggregator の
将来像では **TRT 維持**が正しい．よって**症状側を回避する**方針を採った．

併せて，**Colab の session 変動が 2.2 倍ある**ことが判った (同一コマンドの単発
30 秒が 1 時間のうちに 10,095 → 4,595 p/s)．throughput 比を出す計測では天井を
同じ session 内で測り直す必要がある．

## コード変更 (既に実装済み — 本レビューはその docs 反映が対象)

`maou 0.60.1`: `common.exit_skipping_teardown(tensorrt=)` を新設し，
`maou search` / `maou selfplay` / `maou-usi` の**全出力後**に呼ぶ．
**TensorRT 有効時のみ** stdout/stderr と logging を flush して `os._exit(0)`
する (destructor・atexit を経由しない)．TRT 未使用時は何もしないので，通常の
終了パスと将来の teardown バグ検知能力は保つ．
テスト: `tests/maou/infra/console/test_common.py` — no-op 分岐と，別プロセスで
「flush 済み出力が残る / 呼び出し後のコードは実行されない / 終了コード 0」を検証．

## ドキュメント変更内容 (本レビューの承認対象)

### (a) `verification.md` §8.5 の該当項目を「切り分け完了 + 回避実装済み」へ

3/3 vs 0/3 の結果，二重 ORT 説の否定，回避の実装 (対象 3 コマンド / TRT 有効時
のみ)，**根治はしていない** (`ort` / onnxruntime-gpu 更新時に再確認) を明記．

### (b) `verification.md` §1 に TRT 要否の実測表と session 変動を追記

上記 2 表 (regime 別の TRT vs CUDA / 実測幅 4,595-10,909) と，
**絶対値を session をまたいで比較しない・天井は同一 session で測り直す**という
運用規則を追記．ゲート値 (約 11,000) は上限側なので据え置き．

### (c) `verification.md` §5.1 に同一 session 天井での比を併記

aggregator の余地は同一 session の天井 (10,095) で割ると **1.82×**
(別 session の 10,909 では 1.97×)．**比は必ず同一 session の天井で出す**．

### (d) `docs/commands/{search,selfplay,usi}.md` の CLI 表へ終了時の挙動を 1 行

TensorRT 有効時は出力後に destructor を経由せず終了する (終了コードは 0，
結果は書き出し済み)，TRT 未使用時は通常の終了パス — を `--trt-cache-dir` の
直後に記載．

## 代替案と棄却理由

- **TRT を既定から外す (CUDA のみにする)**: 棄却．バッチが埋まる regime で
  TRT が 1.58 倍速く，実配置と aggregator の将来像はその regime に当たる．
  自己対局 800 playouts での CUDA +6.5% は，EP 由来の数値差で局が分岐した
  2 run の比較なので参考値にとどまる．
- **回避を入れず記録だけ残す**: 棄却．USI エンジンが `quit` 後に SIGABRT で
  落ちるのは対局サーバ運用で許容しにくく，回避は数行で副作用が閉じている．
- **回避を常時有効にする (TRT 判定なし)**: 棄却．CPU 経路まで destructor を
  飛ばすと，将来こちら側の teardown バグを隠す．TRT に限定すれば「外部要因が
  確定している経路だけ」を回避できる．
- **`ort` / onnxruntime-gpu を上げて直す**: 現時点では棄却 (版数は静的リンクの
  ORT と pip provider の一致要件で pin されており，上げるなら両方同時 +
  IR ver ≤ 10 の再確認が要る)．§8.5 に「更新時に再確認」と書いて残す．

## リスクと理由

- **risk: low** — 回避は TRT 有効時のみで，出力完了後に走る．TRT 未使用の
  既定経路・テストは不変 (mock 自己対局で終了コード 0 を確認済み)．
- **reversibility: trivial** — `exit_skipping_teardown` の呼び出し 3 箇所を
  削るだけ．

## ロールバック

3 コマンドの `exit_skipping_teardown` 呼び出しと `common.py` のヘルパ，
`tests/maou/infra/console/test_common.py`，および docs の該当記述を削除する．
