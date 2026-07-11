---
status: applied
applied_in: 449834d
title: maou search の warmup_ms 別掲と Colab 事前ビルド wheel 手順の精査
target: docs/commands/search.md, docs/commands/hcpe_convert.md, docs/design/position-search/benchmarking.md
---

# maou search の warmup_ms 別掲と Colab 事前ビルド wheel 手順の精査

## Validation & 適用結果 (2026-07-09, Colab A100-80GB)

実機検証で計画どおり動作し，docs を適用した (@ 449834d)．

- **warmup 分離が必須だった証拠**: `warmup_ms=32431` (TensorRT エンジンビルド
  32.4 秒) が計測窓 `elapsed_ms=30012` (30 秒) より長い．修正前なら deadline
  超過で `nps≈0` になっていた．修正後は `nps=40230` (onnx_bench 41K と ~2%)．
  `stop=time_limit` で全窓使用．実 NN 出力も健全 (▲2六歩/▲7六歩, Eval~50)．
- **pip-provider 方式で tarball 不要を実証**: pyke ビルドの静的コア ×
  Microsoft ビルドの pip provider (共に onnxruntime 1.22) の ABI が一致し，
  provider ロードエラーなく完走．→ **provider tarball を CI から撤去**
  (`build-gpu-wheel.yml`)．provider は `maou[tensorrt-infer]` の
  onnxruntime-gpu 1.22 (`capi/`) から供給する．
- **numpy 衝突を根治**: cshogi (py3.12 で numpy<1.27 を強制) が唯一の原因で，
  production 探索は cshogi 不使用 → **cshogi を base から `hcpe` extra へ分離**．
  `maou[tensorrt-infer]` は numpy 2.x のまま解決 (検証: numpy==2.5.1)．
  BREAKING: hcpe-convert は `maou[hcpe]` が必要 (パーサに案内ガードを追加)．
- **root-dfpn 併走**: startpos では dfpn が王手手段なく 1 ノードで即終了し
  NPS 影響は無視できた (nodes=1)．戦術局面でのみ競合が顕在化する．

## Trigger

事前ビルド GPU wheel + `maou search` CLI での North-star NPS 計測を Colab
(Rust ビルドなし) で行う準備として手順を精査したところ，計測を無効化する
コード上の欠陥を 1 件発見した (下記)．コード側は本セッションで修正済み
(maou_search 0.9.0 / maou_rust 0.11.0 / maou 0.24.0)．本 review は，その修正に
伴う durable-doc (docs/commands/, docs/design/) の同期と，Colab 手順の
精査結果 (自動 wheel 選択・GPU プリフライト・比較の妥当性) を追認するために
起票する．

## 発見した欠陥 (コード側は修正済み)

`maou search` は onnx_bench と違い **timer 外 warmup をしていなかった**．

- `search.rs` で timer 開始 (`let start = Instant::now()`) の**後**に最初の
  推論 (ルート評価) を同期実行していた．
- `--tensorrt` 時は `pad_to = batch_size` なので，この初回推論が
  shape=batch_size の **TensorRT エンジンをビルド** (数十秒〜数分) する．
- そのビルドが `deadline = start + time_ms` の**計測区間内**に入るため，
  新規 VM で `--tensorrt --time-ms 30000` を素で叩くと 30 秒予算が
  エンジンビルドに食われ `playouts≈0`, **`nps≈0`** になる．
- onnx_bench は out-of-timer warmup (`onnx_bench.rs`) で回避していた．
- TRIPWIRE (NPS 計測の健全性) に直結する．

### 修正 (適用済み — 追認を求める commit で確定させる)

- `search.rs`: 計測 (`start`/`deadline`) をルート評価/展開の**後**に開始する
  ように並べ替え，ルート評価の所要を `SearchStats.warmup_ms` として別掲した．
  `nps`/`elapsed_ms` は探索本体のみを反映する (onnx_bench の warmup と同義)．
- `maou_rust`: `SearchResult.warmup_ms` を pyclass に追加．
- `run.py`: Stats 行に `warmup_ms=` を追加．
- テスト: `test_run.py` に `warmup_ms=` の存在チェックを追加．mock 実測で
  `elapsed_ms=3 warmup_ms=80` と分離を確認 (45 Rust + 6 Python テスト green)．
- `examples/onnx_bench.rs` / `examples/nps_bench.rs`: `--root-dfpn` フラグ追加
  (dfpn 併走の CPU/メモリ競合の NPS 影響を A/B 計測可能に; user 指示)．
- `pyproject.toml`: `onnx-gpu-infer` / `tensorrt-infer` extra を
  `onnxruntime-gpu==1.22.*` (ort 静的コアと同版) + `tensorrt-cu12` 系 `==10.*`
  に固定．provider を pip 供給し tarball wget を廃止するため (uv.lock 追随)．

## Proposed change A — docs/commands/search.md (Outputs)

Stats フィールド一覧に `warmup_ms` を追記し，例出力を更新する:

- フィールド説明に追加:
  `warmup_ms` (ルート評価 = 初回推論のエンジンビルド/ロード所要．**計測区間
  `elapsed_ms`/`nps` には含まない**別掲値．TensorRT では初回のみ数十秒〜数分)．
- 例出力の Stats 行を
  `Stats: playouts=38 nps=435 elapsed_ms=87 warmup_ms=0 max_depth=4 ...`
  の形へ更新 (mock は warmup≈0)．

## Proposed change B — docs/design/position-search/benchmarking.md §4

「事前ビルド GPU wheel を使う」節を以下の点で精査する (手順の実体は
本 review 承認後に反映する):

1. **warmup 別掲の明記**: `maou search` は初回推論 (エンジンビルド) を
   `warmup_ms` に分離し計測区間から除外するため，**単発の
   `--time-ms 30000` で正しい NPS が出る** (2 回実行の warmup ダンス不要)．
2. **GPU プリフライト**: 先頭に `!nvidia-smi -L` を置き，GPU ランタイム
   未選択で早期に気づけるようにする．
3. **インストールの簡素化 (provider tarball の wget を廃止)** (user 指示
   2026-07-09): GPU 推論 extra を版固定し，provider .so を pip の
   onnxruntime-gpu パッケージ (`capi/`) から供給する．
   - pyproject の `onnx-gpu-infer` / `tensorrt-infer` を
     **`onnxruntime-gpu==1.22.*`** (ort 静的コアと同版) + **tensorrt-cu12 系
     `==10.*`** に固定した (本 commit)．`maou[tensorrt-infer] @ <wheel>` だけで
     provider 3 .so と libnvinfer.so.10 が揃い，**別途 tarball の wget が不要**
     になる．
   - wheel の自動選択: `<version>`/cp3XX の手動置換をやめ，Release API から
     実行中の Python 版 (`sys.version_info`) に一致する wheel を選ぶ．
   - **残存リスク (要 Colab 実機確認)**: pip provider は Microsoft 公式ビルド，
     ort 静的コアは pyke ビルド．同一 onnxruntime 1.22.0 なら provider bridge
     ABI は版キーで一致し，通信は渡された関数ポインタ経由 (静的コアの
     シンボル解決に依存しない) ため**動く見込み**だが未検証．初回 Colab 実行で
     確認する．失敗時は provider tarball 経路 (下記フォールバック) に戻す —
     CI の tarball 生成は当面残す．
4. **model 準備を prerequisite として明記**: `/content/model_fp16.onnx` の
   入手 (Drive mount / gdown / ドラッグ&ドロップ) を注記．
5. **永続 `--trt-cache-dir /content/trt_cache`**: 同 VM でのパラメータ掃引時，
   同一 batch-size のエンジン再ビルドを避ける (batch-size 変更時は shape が
   変わり再ビルドは走る)．
6. **root_dfpn を計測に含める** (user 指示 2026-07-09): 実配備は
   ルート並行 dfpn を併走させるため，dfpn スレッドの CPU/メモリ競合が
   MCTS の NPS に与える影響込みで計測する．
   - NPS 計測: startpos + `--tensorrt --cuda --threads 2 --batch-size 256
     --root-dfpn`．詰みのない startpos では dfpn は詰みを見つけず
     `stop=time_limit` で全窓を使い切る (途中停止しない)．
   - **Rust ベンチ (onnx_bench / nps_bench) にも `--root-dfpn` フラグを追加した**
     (本 commit)．41K 系列 (root_dfpn なし) との A/B で dfpn 併走の NPS 影響を
     測れる．
   - **予算/窓の注意**: dfpn のノード予算は default `root_dfpn_nodes` (1<<20 ≈
     冒頭 ~1 秒の詰みスキャン)．詰みのない局面では予算消化後に dfpn
     スレッドが終了するため，競合は計測窓の**先頭に集中**する (長い
     `--time-ms` では平均 NPS 影響が薄まる)．また **CPU コア数に余裕がある
     環境 (例: DevContainer 4C で MCTS 2T + dfpn 1T) では競合が起きず影響は
     ノイズ内**．影響が顕在化するのは vCPU が少ない Colab GPU 実機．
     持続的競合を測りたい場合は短めの窓 or dfpn 予算の拡大 (別レバー)．
7. **診断粒度の注記**: `maou search` の Stats は見出し NPS + warmup_ms のみ．
   fill%/collisions/gc などの掃引診断が要るときは §4 後半の onnx_bench
   (リポジトリビルド) 経路を使う．

### 提案する Colab セル列 (承認後に §4 へ反映)

```python
# 0. GPU プリフライト (未選択なら Runtime > Change runtime type > GPU)
!nvidia-smi -L
```

```python
# 1. wheel + GPU 推論依存を pip 一発で取得 (wget 不要)．
#    maou[tensorrt-infer] が onnxruntime-gpu==1.22.* (provider .so を capi/ に
#    同梱) と tensorrt-cu12==10.* (libnvinfer.so.10) を引く．版は ort 静的コア
#    (onnxruntime 1.22) と一致させてある．
#    (maou search 自体は torch も google-cloud も使わない — provider 供給が目的)
import json, sys, urllib.request
rel = json.load(urllib.request.urlopen(
    "https://api.github.com/repos/dousu/maou/releases/tags/latest-gpu"))
pytag = f"cp{sys.version_info.major}{sys.version_info.minor}"   # Colab の Python 版
whl = next(a["browser_download_url"] for a in rel["assets"]
           if a["name"].endswith(".whl") and pytag in a["name"])
print("wheel:", whl.rsplit("/", 1)[-1])
!pip install -q "maou[tensorrt-infer] @ {whl}"
```

```python
# 2. pip 同梱の .so を loader パスに載せる (ldconfig)．
#    provider 3 .so は onnxruntime/capi/，libnvinfer は tensorrt_libs/，
#    CUDA/cuDNN は Colab の torch 同梱 nvidia libs から解決する．
import glob
dirs = (glob.glob("/usr/local/lib/python3*/dist-packages/onnxruntime/capi")
        + glob.glob("/usr/local/lib/python3*/dist-packages/tensorrt_libs")
        + glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/*/lib"))
with open("/etc/ld.so.conf.d/maou.conf", "w") as f:
    f.write("\n".join(dirs) + "\n")
!ldconfig
# 両方 1 行以上出ること (出なければ provider/TensorRT が未解決)
!ldconfig -p | grep -E "libonnxruntime_providers_shared|libnvinfer.so.10"
```

> フォールバック (pip provider が静的コアと ABI 不一致だった場合): 従来の
> provider tarball 経路 — Release `latest-gpu` の
> `onnxruntime-gpu-providers.tar.gz` を wget/展開し，その dir を上の `dirs`
> 先頭に足す (CI は当面この tarball も生成し続ける)．

```python
# 3. 実モデル (_fp16 ONNX) を /content/model_fp16.onnx に置く (prerequisite)
#    例) from google.colab import drive; drive.mount("/content/drive")
#        !cp "/content/drive/MyDrive/.../model_fp16.onnx" /content/model_fp16.onnx
#    or  !gdown <FILE_ID> -O /content/model_fp16.onnx
#    or  左のファイルペインへドラッグ&ドロップ
```

```python
# 4. North-star NPS 計測 (単発で正しい NPS．engine build は warmup_ms に分離)
#    実配備と同条件: startpos / 2T / b256 / root-dfpn あり (dfpn 併走の影響込み)
!maou search \
    --sfen "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1" \
    --model-path /content/model_fp16.onnx \
    --tensorrt --cuda --threads 2 --batch-size 256 --time-ms 30000 --root-dfpn \
    --trt-cache-dir /content/trt_cache
# Stats: ... nps=<計測> elapsed_ms=~30000 warmup_ms=<engine build> stop=time_limit
# 期待: stop=time_limit / warmup_ms は初回のみ大．
# dfpn 併走の影響を測るには --root-dfpn 有無で A/B を取る (41K は無しの系列)．
```

## What this enables

- 事前ビルド wheel + `maou search` で，Rust ビルドなしに単発で正しい NPS を
  計測できる (warmup_ms 分離により初回計測が無効化されない)．
- **インストールが `pip install "maou[tensorrt-infer] @ <wheel>"` + ldconfig の
  2 ステップに簡素化** (provider tarball の wget を廃止)．
- root_dfpn 併走の CPU/メモリ競合が MCTS の NPS に与える影響を，Rust ベンチ /
  `maou search` の両方で `--root-dfpn` 有無の A/B として計測できる．

## What this constrains

- `SearchStats`/`SearchResult`/`run.py` の Stats 出力は `warmup_ms` を含む
  形を維持する (バージョン: maou_search 0.9.0 / maou_rust 0.11.0 / maou 0.24.0)．
- GPU 推論 extra (`onnx-gpu-infer`/`tensorrt-infer`) は onnxruntime-gpu を
  **ort 静的コアと同版 (1.22.*)** に固定し続ける (ort bump 時は追随)．
  tensorrt-cu12 系は 10.* 固定 (libnvinfer.so.10)．
- North-star 計測は root_dfpn 併走を既定条件に含める (実配備同条件; user 指示)．

## Rollback plan

- コード: 3 crate/package の version bump と `search.rs` の並べ替え・
  `warmup_ms` フィールド追加を revert (計測は再び build を含むようになる —
  非推奨)．
- docs: 本 review の変更を戻す (search.md の warmup_ms 行削除，benchmarking.md
  §4 を旧手順へ)．
