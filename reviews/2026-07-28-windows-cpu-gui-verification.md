---
title: Windows での USI エンジン動作要件 (VC++ 再頒布可能パッケージ) と CPU wheel の入手経路を docs に追加
date: 2026-07-28
status: pending
target:
  - docs/commands/usi.md
  - docs/design/usi-engine/verification.md
risk: low
reversibility: easy
---

# 提案: Windows CPU 機での GUI 実機検証に必要な前提を docs に残す

## Trigger

GUI 実機検証 (verification.md §8，未実施のまま carry-over されていた) を
Windows CPU 機で始めたところ，**docs に書かれていない前提が 2 つ**あって
どちらも手が止まった．どちらも「知っていれば 1 分，知らないと原因不明」
の類なので committed docs 側に残したい．

### 1. Windows wheel が存在しなかった

`.github/workflows/build-wheel.yml:22,29` は `ubuntu-latest` /
`x86_64-unknown-linux-gnu` 固定で，Release `latest` の資産も manylinux
2 本のみ．`docs/rust-build-optimization.md:105` にも「Windows x86_64 →
当面不要」とある．

→ `.github/workflows/build-wheel-windows.yml` を追加した (9361a12 /
039747c)．`windows-latest` / `x86_64-pc-windows-msvc` / Python 3.11・3.12
の matrix で **Actions artifact** として出す．**Release `latest` の配布
方針 (単一 manylinux wheel — reviews/2026-07-16-unified-wheel-build.md)
には手を付けていない**．

**実測**: ort の ONNX Runtime 静的リンクは MSVC で問題なく通った
(`.pyd` 44.5MB / wheel 13.8MB)．`onnxruntime.dll` への依存は無い．
「Windows は当面不要」は技術的障害があったのではなく未着手だっただけ，
という位置づけになる．

### 2. VC++ 再頒布可能パッケージが要る (これが本題)

クリーンな Windows 機で `maou-usi` を起動すると:

```
ImportError: DLL load failed while importing _rust:
指定されたモジュールが見つかりません。
```

Windows は**欠けた DLL の名前を出さない**ので，これだけでは特定できない．
`objdump -p _rust.cp312-win_amd64.pyd` の import table を読むと:

| DLL | 供給元 |
|---|---|
| `MSVCP140.dll` / `MSVCP140_1.dll` | **VC++ 再頒布可能パッケージ** |
| `VCRUNTIME140.dll` / `VCRUNTIME140_1.dll` | VC++ 再頒布可能パッケージ (Python も同梱) |
| `DirectML.dll` / `d3d12.dll` / `dxgi.dll` / `dxcore.dll` | ONNX Runtime の DirectML EP (Windows 10 1903+ に同梱) |

**`MSVCP140.dll` (C++ 標準ライブラリ) は Python にも Windows にも同梱
されない**．ONNX Runtime が C++ なのでこれが要る．
`winget install Microsoft.VCRedist.2015+.x64` で解決することを実機で確認．

**CI では原理的に検出できない** — GitHub runner には最初から入っている．
これが「CI green なのに実機で動かない」の発生源になる．

## 提案する変更

### A. `docs/commands/usi.md` § Engine registration in a GUI

Windows の登録手順 (:60-61) の直後に前提を 1 項足す:

```markdown
- **Windows prerequisite**: the wheel links ONNX Runtime (C++) statically,
  so it needs the Microsoft Visual C++ Redistributable 2015-2022 (x64) —
  `MSVCP140.dll` ships with neither Python nor Windows. Without it the
  engine dies at startup with `ImportError: DLL load failed while
  importing _rust` (which does *not* name the missing DLL). Install with
  `winget install Microsoft.VCRedist.2015+.x64`. CI cannot catch this —
  GitHub's Windows runners have it preinstalled.
```

### B. `docs/design/usi-engine/verification.md` §7

§7 の inline スクリプトを `scripts/usi_smoke.py` へ差し替える (現状は
Colab の `%%writefile` 前提で書かれており，GUI 機でそのまま流せない)．
スクリプトは既定で mock 評価器なのでモデル転送前に疎通確認ができ，
起動失敗時は欠落 DLL を名指しする．§7 の「一括 pipe は使わない」の
注意はスクリプト側の docstring に移してある．

### C. `docs/design/usi-engine/verification.md` §8

環境要件 (:635-637) に Windows CPU 機での入手経路を追記する:
Actions artifact `maou-windows-wheel-py3.1x` → venv へ pip install →
`scripts/usi_smoke.py` で疎通 → GUI 登録．
GUI は **winget 経由の入手**を推奨として明記する (`winget install
sunfish-shogi.shogihome` / `shogixyz.ShogiGUI`)．ブラウザ DL だと
Mark of the Web が付いて SmartScreen が出るが，winget は自分で DL して
manifest の SHA256 で検証するため出ない．将棋 GUI はどれも未署名なので
GUI を変えても回避できず，**回避すべきは入手経路**という整理．

## 検討したが採らない案

- **wheel に VC ランタイム DLL を同梱する** (app-local deployment は
  Microsoft のライセンス上許容される)．棄却理由: wheel が肥大し，
  ランタイムの更新 (セキュリティ修正) が我々の再ビルド待ちになる．
  1 コマンドで入る依存を明記する方が保守が軽い．
- **Windows wheel を Release `latest` に添付する**．棄却理由: 配布方針は
  reviews/2026-07-16-unified-wheel-build.md (applied) の決定事項で，
  変えるなら独立した提案が要る．今は GUI 検証という一時的な目的しか
  無いので Actions artifact (90 日) で足りる．**GUI 検証が終わってから
  改めて判断する**．

## Risks

- **DirectML.dll への依存**: 低 / 可搬性 / Windows 10 1903+ に同梱なので
  実用上は問題ないが，**我々は DirectML EP を使っていないのに依存が乗って
  いる** (ort の prebuilt 由来)．より古い Windows では起動しない可能性が
  ある．今回の実機では欠落していない (VCRedist 導入だけで起動した)．
- **CPU の最適 `batch_size` が未測定**なまま Windows 手順を docs 化する:
  低 / 既定 8 のままなので新たな誤りは持ち込まない．GPU 推奨の 64 を
  CPU に流用しない旨は既に compass の invariant にある．
