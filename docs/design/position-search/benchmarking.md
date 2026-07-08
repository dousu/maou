# NPS ベンチマーク手順 (maou_search)

1局面探索エンジンの NPS 計測手順．計測規律は
[index.md §10.2](index.md) が正 (release 必須，North-star への計上は
Colab GPU × 実モデルのみ)．

## 1. ベンチの種類

| ベンチ | 評価器 | 用途 |
|---|---|---|
| `examples/nps_bench.rs` | MockEvaluator (擬似乱数) | NN 抜きの探索コア上限，GC/バッチ挙動の相対比較 |
| `examples/onnx_bench.rs` | OnnxEvaluator (ONNX Runtime) | 実推論込みの NPS．North-star 計測はこれ (GPU × 実モデル) |

主なオプション (両ベンチ共通; `--help` 参照):
`--sfen` / `--threads` / `--batch` / `--playouts` / `--time-ms` / `--capacity` /
`--cpuct` / `--fpu` / `--keep-ratio` / `--no-gc`．
onnx_bench 専用: `--model PATH` (必須) / `--ort-threads` / `--cuda` /
`--tensorrt` / `--trt-cache DIR` / `--pad N`．

## 2. テスト用 ONNX モデル

実 checkpoint がなくても，I/O 契約 (board int32 (B,9,9) + hand f32 (B,14) →
policy (B,1496) + value (B,1)) 互換の極小モデルで配線と探索側スループットを
検証できる (torch 不要，onnx パッケージのみ):

```bash
uv run --with onnx python rust/maou_search/tests/make_tiny_onnx.py /tmp/tiny.onnx
```

**注意**: 極小モデルの NPS は配線検証用の参考値．North-star はモデルサイズに
依存するため，実モデル (学習パイプラインが export した .onnx) で計測すること．
学習パイプラインは FP16 変換済みモデル (`_fp16` サフィックス) も出力しており，
GPU 計測では **`_fp16` を使う** (A100 実測で FP32 比の大幅な NPS 向上を確認済み)．

## 3. ローカル (DevContainer) — 相対比較専用

```bash
# mock ベンチ
cargo build --release -p maou_search --example nps_bench
/tmp/cargo-target/release/examples/nps_bench --threads 4 --batch 16 --time-ms 5000

# ONNX ベンチ (CPU)
cargo build --release -p maou_search --features onnx --example onnx_bench
/tmp/cargo-target/release/examples/onnx_bench \
    --model /tmp/tiny.onnx --threads 2 --batch 32 --ort-threads 2 --time-ms 8000
```

DevContainer は物理 2 コア × SMT (4 vCPU) であり，スレッドスケーリングの
検証には使えない (worklog 2026-07-08 の pinning 実験参照)．

## 4. Colab (GPU) — North-star 計測

以下は Colab で検証済みのセル列 (2026-07-08，T4 相当)．

```python
# 1. リポジトリ取得 (ブランチは適宜)
!git clone --branch feat/position-search https://github.com/dousu/maou.git
%cd maou

# 2. Rust toolchain + リンカ (.cargo/config.toml が clang + lld を要求する)
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
!apt-get -qq install -y clang lld
import os
os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + os.environ["PATH"]
# .cargo/config.toml の jobs = 1 (DevContainer 省メモリ設定) を上書きする
os.environ["CARGO_BUILD_JOBS"] = str(os.cpu_count())
```

```python
# 3. (推奨) 特徴量/ラベル移植の parity テスト — GPU 不要
!cargo test -p maou_search -- --test-threads=1
```

```python
# 4. ベンチのビルド (ort が onnxruntime GPU バイナリを自動ダウンロードする)
!cargo build --release -p maou_search --features onnx-cuda --example onnx_bench
```

```python
# 5. 実行時ライブラリパス (重要 — §5 参照)
import glob
import os

repo = "/content/maou"
nvidia_libs = glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/*/lib")
os.environ["LD_LIBRARY_PATH"] = ":".join(
    [f"{repo}/target/release/examples", f"{repo}/target/release"]
    + nvidia_libs
    + [os.environ.get("LD_LIBRARY_PATH", "")]
)
```

```python
# 6a. 配線検証 (極小モデル)
!pip -q install onnx
!python rust/maou_search/tests/make_tiny_onnx.py /content/tiny.onnx
!target/release/examples/onnx_bench --model /content/tiny.onnx --cuda \
    --threads 2 --batch 128 --time-ms 15000

# 6b. North-star 計測 (実モデル)
!nvidia-smi -L
!target/release/examples/onnx_bench --model /content/model.onnx --cuda \
    --threads 2 --batch 128 --time-ms 30000
```

`--cuda` は CUDA Execution Provider の初期化失敗時に即エラーで落ちる設計
(静かな CPU フォールバックで GPU 計測を誤らせないため)．CPU で動かす場合は
`--cuda` を外す．

### TensorRT EP を使う場合

CUDA EP 比でカーネル融合による高速化を狙う (A100/ViT-19.8M/FP16 実測で
CUDA EP 比 約 2.2×)．feature `onnx-tensorrt` でビルドし，TensorRT ランタイム
(libnvinfer) を pip で導入する:

```python
# TensorRT ランタイム．onnxruntime バイナリが要求するのは 10 系
# (libnvinfer.so.10)．pip の最新は 11 系なので pin が必須 (2026-07 検証済み)
!pip install "tensorrt-cu12==10.*"

# ライブラリを ldconfig に登録する (環境変数伝播に依存しない確実な方法．
# SONAME で登録されるためファイル名の版番号差異も吸収される)
import glob
dirs = (
    glob.glob("/usr/local/lib/python3*/dist-packages/tensorrt_libs")
    + glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/*/lib")
    + ["/content/maou/target/release", "/content/maou/target/release/examples"]
)
with open("/etc/ld.so.conf.d/maou.conf", "w") as f:
    f.write("\n".join(dirs) + "\n")
!ldconfig
!ldconfig -p | grep libnvinfer.so.10   # 1 行以上出ること

# ビルド (CUDA EP へのフォールバック用に onnx-cuda も同時に有効化)
!cargo build --release -p maou_search \
    --features onnx-cuda,onnx-tensorrt --example onnx_bench

# 実行 (--tensorrt --cuda: TensorRT 優先，非対応ノードは CUDA へ)
!target/release/examples/onnx_bench --model /content/model_fp16.onnx \
    --tensorrt --cuda --threads 2 --batch 256 --time-ms 30000
```

- **初回実行はエンジンビルドで数分かかる** (`trt_cache/` にキャッシュされ，
  2 回目以降は数秒でロード)．NPS 計測は 2 回目以降の実行で行うこと．
- `--tensorrt` 時はバッチが探索バッチサイズへ自動 padding される
  (`--pad N` で上書き可)．TensorRT は入力 shape ごとにエンジンを構築する
  ため，可変バッチのまま渡すと shape の数だけビルドが走る．

## 5. トラブルシューティング

### `libonnxruntime_providers_shared.so: cannot open shared object file`

onnxruntime GPU 版は provider ライブラリを実行時に **dlopen で名前解決**する．
ort はビルド時に `target/<profile>/`，`target/<profile>/deps/`，
`target/<profile>/examples/` へ .so をコピーするが，バイナリの隣にあるだけでは
探索対象にならないため，`LD_LIBRARY_PATH` に載せる必要がある (§4 セル 5)．
場所が違う場合は `find target -name "libonnxruntime_providers_shared.so"` で
実体を探し，そのディレクトリを先頭に足す．

### `libcudnn.so.9` 等の CUDA 系ロードエラー

CUDA EP は cuDNN 9 / cuBLAS 等を要求する．Colab では torch 同梱の nvidia pip
パッケージ (`.../dist-packages/nvidia/*/lib`) にあるため，§4 セル 5 の
`nvidia_libs` で解決する．

### `libnvinfer.so.10: cannot open shared object file`

onnxruntime の TensorRT EP はビルド時ダウンロードされた onnxruntime バイナリと
**同じメジャー版の TensorRT** を要求する (現行 10 系)．pin なしの
`pip install tensorrt-cu12` は **11 系が入り** (`tensorrt_libs/` 内が
`*.so.11.*` になる)，どうパスを通しても解決しない．
`pip install "tensorrt-cu12==10.*"` で 10 系に入れ替えること
(`ls .../tensorrt_libs/ | grep libnvinfer.so.10` で確認できる)．

パス解決は `LD_LIBRARY_PATH` より **ldconfig 登録が確実** (§4 参照)．
`ldconfig` 実行時に既存ライブラリへの `... is not a symbolic link` 警告が
大量に出るが無害．VM リセット後は ldconfig のセルだけ再実行する．

### コピペ時のパス分断

パス途中に空白や改行が入ると `python rust/maou_search/tests /make_tiny_onnx.py`
のように別引数と解釈され `can't find '__main__' module` になる．

## 6. 統計の読み方とパラメータ掃引

出力の主指標:

| 指標 | 意味 | 目安 |
|---|---|---|
| `NPS` | playouts/s (葉評価スループット) | North-star: GPU 実測 100万 |
| `avg_batch` / fill % | 評価バッチの平均充填率 | **GPU 効率の主指標**．低いと GPU が遊ぶ |
| `collisions` | 展開競合によるロールバック率 | batch × threads が木の広さに対して過大だと急増 |
| `gc_runs` / `gc_freed_nodes` | プール GC の回数/解放量 | 高頻度なら `--capacity` か `--keep-ratio` を調整 |
| `stop` | 停止理由 | 計測は TimeLimit で終わること (PoolExhausted は容量不足) |

掃引の指針:

- `--batch 128/256/512` × `--threads 2/4` を格子で計測する．
  **fill % が維持できる範囲で batch を上げる**のが基本 (fill 20% を切るような
  batch は virtual loss が探索を歪めるだけで得がない)．
- 1 ルートからの virtual loss バッチ収集で埋められる葉数には限界があるため，
  batch を上げるほど衝突率が上がり fill が下がる (tiny model では 2T/batch128
  で fill 17%．実モデルは 1 推論が重く供給が追いつくため同条件で fill 99%)．
- **per-batch 時間がバッチサイズに対して線形なら GPU は既にスループット飽和**
  しており，batch 増でも NPS は伸びない (A100/FP16/ViT-19.8M の掃引で確認．
  この状態では per-call オーバーヘッド削減より TensorRT 等の per-item
  推論効率が効く)．倍にして per-batch 時間が倍未満なら batch 増が有効．
- GPU 律速時は threads 2 で十分．増やしても衝突が増えるだけで NPS は不変．
- 現状 ort の Session::run は Mutex で直列化されている (PoC 制約)．fill が
  十分なのに NPS が伸びない場合はここが容疑者 (per-thread session /
  IoBinding が改善候補)．

## 7. 記録規律

計測値は worklog / compass (campaign 中) に記録し，本ドキュメントには
手順と読み方だけを置く (数値は陳腐化が速いため)．
