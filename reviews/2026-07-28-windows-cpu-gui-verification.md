---
title: Windows ビルド・実行の知見を docs に残し，Windows wheel CI は休眠させる
date: 2026-07-28
status: pending
target:
  - docs/rust-build-optimization.md
  - docs/commands/usi.md
  - docs/design/usi-engine/verification.md
risk: low
reversibility: easy
---

# 提案: Windows の知見を残す (配布対象にはしない)

## 位置づけ

**maou の主対象は高性能 GPU 環境で，現状は主に Colab** (user, 2026-07-28)．
Windows は配布対象ではない．この提案は Windows 対応を増やすものではなく，
**一度だけ実際に踏破した経路の知見を，次に必要になったときに再発見しなくて
済む形で残す**ためのもの．

`.github/workflows/build-wheel-windows.yml` は追加済みだが
**`workflow_dispatch` のみの休眠状態**にしてある (通常の push では走らない)．
Release `latest` の配布方針 (単一 manylinux wheel —
reviews/2026-07-16-unified-wheel-build.md) には手を付けていない．

## Trigger

GUI 実機検証 (verification.md §8) が「GUI を動かせる環境が無い」ため
carry-over され続けていた．Windows CPU 機で実施したところ，**docs に
無い前提で 2 回止まった**．どちらも「知っていれば 1 分，知らないと原因
不明」の類だった．結果として §8 のうちエンジン登録と実対局の開始までは
到達した (maou が実 GUI で対局した最初の記録)．

## 実証された事実 (2026-07-28)

### 1. Windows ビルドは通る

`docs/rust-build-optimization.md:105` は「Windows x86_64 | 当面不要」と
だけ書いており，技術的に可能かどうかは未知だった．**可能**．

- `windows-latest` / `x86_64-pc-windows-msvc` / maturin / Python 3.11・3.12
- **ort の ONNX Runtime 静的リンクは MSVC で問題なく通る**．
  `onnxruntime.dll` への依存は無い (`.pyd` 44.5MB / wheel 13.8MB)
- 必須フラグ 2 つ (どちらも既定では付かない):
  - `--features pyo3/extension-module,onnx` — 省くと pure Rust ビルドに
    なり `ModelPath` が使えない (`pyproject.toml:130` の既定は
    `["pyo3/extension-module"]` のみ)
  - `--release` — `pyproject.toml:131` が `profile = "dev"` なので
    省くと debug ビルドになる
- 初回ビルド約 35 分，sccache 有効後は大幅に短縮

### 2. 実行機に VC++ 再頒布可能パッケージが要る

クリーンな Windows 機で `maou-usi` を起動すると:

```
ImportError: DLL load failed while importing _rust:
指定されたモジュールが見つかりません。
```

Windows は**欠けた DLL の名前を出さない**ので，これだけでは特定できない．
`objdump -p _rust.cp312-win_amd64.pyd` の import table:

| DLL | 供給元 |
|---|---|
| `MSVCP140.dll` / `MSVCP140_1.dll` | **VC++ 再頒布可能パッケージ** |
| `VCRUNTIME140.dll` / `VCRUNTIME140_1.dll` | VC++ 再頒布可能パッケージ (Python も同梱) |
| `DirectML.dll` / `d3d12.dll` / `dxgi.dll` / `dxcore.dll` | ONNX Runtime の DirectML EP (Windows 10 1903+ に同梱) |

**`MSVCP140.dll` (C++ 標準ライブラリ) は Python にも Windows にも同梱
されない**．ONNX Runtime が C++ なのでこれが要る．
`winget install Microsoft.VCRedist.2015+.x64` で解決することを実機で確認．

**CI では原理的に検出できない** — GitHub runner には最初から入っている．
「CI green なのに実機で動かない」の発生源になる型の欠落．

### 3. GUI の入手経路 (SmartScreen)

将棋 GUI (将棋所 / ShogiGUI / ShogiHome) はいずれも未署名なので，ブラウザで
DL すると Mark of the Web が付いて SmartScreen の警告が出る．**GUI を変えても
避けられない — 変えるべきは入手経路**．winget はクライアント自身が DL して
manifest の SHA256 で検証するため警告が出ない:

- `winget install sunfish-shogi.shogihome` (ShogiHome)
- `winget install shogixyz.ShogiGUI` (ShogiGUI)

また `maou-usi` (引数なしエントリポイント) は Windows で
`<venv>\Scripts\maou-usi.exe` になるので，**bat ラッパーは不要**で GUI に
そのまま .exe を登録できる．

## 提案する変更

### A. `docs/rust-build-optimization.md:105` — 知見の主たる置き場

「Windows x86_64 | Windows 開発者 | 当面不要」の行に脚注を付け，上記
「実証された事実 1・2」を短く残す (必須フラグ 2 つ / VCRedist 必須 /
静的リンクは通る / workflow は `build-wheel-windows.yml` に休眠)．
**当面不要という判断自体は変えない**．

### B. `docs/commands/usi.md` § Engine registration in a GUI

このファイルは **:59-61 で既に Windows の登録手順を書いている**ので，
Windows を配布対象にするかに関わらず前提が抜けている状態になっている．
1 項足す:

```markdown
- **Windows prerequisite**: the wheel links ONNX Runtime (C++) statically,
  so it needs the Microsoft Visual C++ Redistributable 2015-2022 (x64) —
  `MSVCP140.dll` ships with neither Python nor Windows. Without it the
  engine dies at startup with `ImportError: DLL load failed while
  importing _rust`, which does *not* name the missing DLL. Install with
  `winget install Microsoft.VCRedist.2015+.x64`. CI cannot catch this —
  GitHub's Windows runners have it preinstalled. Prebuilt Windows wheels
  are not distributed; build on demand via the `Build Windows Wheel`
  workflow (manual dispatch).
```

### C. `docs/design/usi-engine/verification.md` §7 / §8

- §7 の inline スクリプトを `scripts/usi_smoke.py` への参照に差し替える．
  現状は Colab の `%%writefile` 前提で書かれており，他環境でそのまま
  流せない．**このスクリプトは Windows 専用ではなく Colab でも使う**
  (既定は mock 評価器なのでモデル転送前に疎通確認ができ，`KeepAlive` の
  空行を数えるので未決 2 の非空性チェックにも使える)．
- §8 に GUI の入手経路 (winget) と，`maou-usi.exe` を直接登録できること
  を追記する．

## 検討したが採らない案

- **Windows wheel を Release `latest` に添付する**．棄却理由: 主対象は
  GPU 環境 (Colab) で Windows は配布対象でない．配布方針は
  reviews/2026-07-16-unified-wheel-build.md (applied) の決定事項でもある．
- **wheel に VC ランタイム DLL を同梱する** (app-local deployment は
  Microsoft のライセンス上許容される)．棄却理由: 配布対象でないものの
  ために wheel を肥大させる意味が無く，ランタイムのセキュリティ更新が
  我々の再ビルド待ちになる．
- **workflow ごと削除する**．棄却理由: 再構築のコストが実測で約 35 分 +
  必須フラグ 2 つの再発見で，休眠させておくコストはゼロ
  (`workflow_dispatch` のみなので通常の push では走らない)．

## Risks

- **休眠 workflow が腐る**: 中 / 保守 / 走らないので actions のバージョン
  更新や maturin の非互換に気付けない．次に起こしたとき赤いところから
  始まる可能性がある．許容 — 上の「35 分 + 再発見」より安い．
- **DirectML.dll への依存**: 低 / 可搬性 / **DirectML EP を使っていないのに
  依存が乗っている** (ort の prebuilt 由来)．Windows 10 1903+ には同梱
  なので今回の実機では問題にならなかったが，より古い Windows では起動
  しない可能性がある．
