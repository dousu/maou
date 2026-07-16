---
title: wheel 配布の一本化 (通常 wheel と GPU wheel の統合)
date: 2026-07-16
status: applied
applied_in: 6d9f0f4
# user 承認 2026-07-16．workflow 統合 = 6d9f0f4，docs = a41af07 (index.md /
# benchmarking.md / search.md) + analyze_game.md (PR #387 マージ取り込み後の
# 後続コミット) で全 target 適用済み．latest-gpu Release の削除は本 PR
# マージ後，統合 latest ビルドの初回成功を確認してから実施する．
target:
  - docs/design/position-search/index.md
  - docs/design/position-search/benchmarking.md
  - docs/commands/search.md
  - docs/commands/analyze_game.md
risk: medium
reversibility: moderate
---

# 提案: 単一 wheel 化 — `latest` に onnx-cuda + onnx-tensorrt を同梱し `latest-gpu` を廃止

## 背景 (user 課題, 2026-07-16)

1. main へ push しても GPU wheel がビルドされない (build-gpu-wheel.yml は
   workflow_dispatch のみ．latest-gpu は 0.33.0 で放置，latest は 0.42.0)
2. 通常 wheel と GPU wheel の分離によるストレージコスト
3. 2 種類の wheel を別々に管理したくない

現状の実態 (2026-07-16 調査):

- リポジトリは public のため Release ストレージは無課金・上書き運用で蓄積なし
  (計 ~48MB)．実害は「鮮度乖離」と「二重管理」
- 通常 wheel は cargo features なし = **onnx 非搭載で NN 推論不可** (mock のみ)．
  「CPU wheel vs GPU wheel」ではなく「データパイプライン wheel vs フル wheel」
  になっている
- GPU wheel は ONNX Runtime core 1.22 を静的リンク (+9MB)．CUDA/TensorRT の
  provider .so は wheel 非同梱で，実行時に pip の `onnxruntime-gpu==1.22.*`
  (`maou[tensorrt-infer]` が版 pin) から dlopen．EP 有効化は実行時フラグ
  (`use_cuda`/`use_tensorrt`, default off) — **runtime gate は既に二段構え**

## 検証結果 (2026-07-16 実測 — 本提案の根拠)

**(a) manylinux_2_28 コンテナで統合ビルド可能** (CI probe run 29473257950):
`--features pyo3/extension-module,onnx-cuda,onnx-tensorrt` +
`yum install openssl-devel` (ort のバイナリ取得用 TLS 依存) のみで
`maou-*-manylinux_2_28_x86_64.whl` (16MB, cp311/cp312) がビルド・監査を通過．
現行 GPU wheel の manylinux_2_35 より**可搬性は向上** (glibc 2.28 まで下がる)．

**(b) CPU-only 環境で GPU-feature ビルドが完全動作** (DevContainer 実測):
- ONNX **CPU 推論が動作** (SearchEngine + ONNX モデルで maou search /
  analyze-game e2e 完走)
- `use_cuda=True` は provider dlopen 失敗の**明示的 RuntimeError**
  (silent CPU fallback なし = 計測を誤らせない既存設計のまま)
- ロード時の動的依存は libc 系 + libstdc++ のみ (CUDA 非依存)

## 提案内容

### 1. workflow 統合 (.github/workflows/ — reviews 対象外だが計画を記録)

- **build-wheel.yml**: maturin args に
  `--features pyo3/extension-module,onnx-cuda,onnx-tensorrt` を追加し，
  before-script に `openssl-devel` を追加．manylinux 2_28 は維持．
  Release notes にランタイム要件 (GPU 利用時は `maou[tensorrt-infer]` で
  onnxruntime-gpu 1.22 + tensorrt-cu12 10 系) を記載
- **build-gpu-wheel.yml を削除**，Release **latest-gpu を廃止** (削除)．
  トリガは main push (既存) に一本化 — 課題 1〜3 すべて解消
- ローカル開発の `maturin develop` は従来どおりデフォルト (pure Rust)．
  features は wheel ビルド時のみ付与

### 2. docs 改定 (approve 対象)

- **docs/design/position-search/index.md §2.2** (wheel 可搬性):
  「ONNX/CUDA/TensorRT は optional feature / 別 extra として分離」→
  「**配布 wheel は単一** (onnx-cuda + onnx-tensorrt 同梱, manylinux_2_28)．
  HW 依存は従来どおり実行時フラグで opt-in し，CUDA/TensorRT の実行時
  ライブラリは pip extra (`onnx-gpu-infer` / `tensorrt-infer`) で供給する．
  デフォルト**ビルド** (ローカル dev / crate default) は pure Rust を維持」
- **docs/design/position-search/benchmarking.md §4**: Colab の wheel 取得を
  `latest-gpu` → `latest` に変更 (手順は同型)
- **docs/commands/search.md / analyze_game.md**: 「model_path には onnx
  feature 付き wheel が必要」の記述を「配布 wheel (latest) はそのまま利用可，
  自前ビルド時のみ features 指定が必要」に更新

### 3. VETO の改定 (user 承認をもって compass に反映)

現行: 「CUDA EP / TensorRT 推論は optional extra / **feature 分離**で
デフォルト wheel を可搬に保つ」
改定後: 「**wheel は単一** (onnx-cuda + onnx-tensorrt 同梱, manylinux_2_28)．
HW 有効化は runtime gate のみ (native CPU 命令ビルド棄却は不変)．CUDA /
TensorRT の実行時ライブラリは pip extra で供給し，デフォルト wheel は
CPU-only 環境でそのまま動作すること (検証 b を回帰条件とする)」

## 影響と互換性

- **wheel 利用者**: latest が 7.4MB→16MB (+9MB)，機能は上位互換
  (CPU でも NN 推論が可能になる)．glibc 下限は 2.28 で不変
- **Colab GPU 手順**: 取得元タグの変更のみ (`maou[tensorrt-infer]` 併用は不変)
- **CI**: main merge ごとのビルド時間が増える (ort 取得 + リンク; sccache で
  緩和．probe 実測 ~9 分)．GPU workflow の手動運用は消滅
- **REFUTED との整合**: 棄却済みの「Python extra による GPU/CPU **wheel
  切替**」の再導出ではない — wheel は 1 本になり，extra の役割は現行どおり
  実行時ライブラリの供給のみ

## ロールバック

build-gpu-wheel.yml は git 履歴から復元可能．latest-gpu Release も workflow
再実行で再生成できる (アーカイブ目的の保存は不要)．
