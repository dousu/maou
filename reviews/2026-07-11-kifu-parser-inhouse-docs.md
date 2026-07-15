---
status: applied
applied_in: 6b2e3ed
title: CSA/KIF パーサ独自実装に伴う docs 同期 (hcpe_convert.md 要件 + rust-backend.md モジュール表)
target: docs/commands/hcpe_convert.md, docs/rust-backend.md
---

# CSA/KIF パーサ独自実装に伴う docs 同期

## Trigger

campaign「production の cshogi 依存除去」(user 指示 2026-07-11) の実装で，
CSA/KIF パーサを Rust (maou_shogi::kifu) の完全独自実装に置換した
(maou_shogi 5.7.0 / maou_rust 0.17.0 / maou 0.34.0)．これにより
`hcpe-convert` は base install で動作するようになり，CLI の lazy loader
から `PackageRequirement("cshogi", ("hcpe",))` を撤去した
(src/maou/infra/console/app.py)．CLAUDE.md の documentation MUST
(CLI 変更時の docs/commands/ 同期) に従い docs 更新を起票する．

## 提案する docs 変更

### 1. docs/commands/hcpe_convert.md — Requirements 節の書き換え

現行 (stale — cshogi/hcpe extra が必須という記述):

> - CSA/KIF parsing uses `cshogi`, which is **not** installed by default (it pins
>   `numpy<1.27` on Python 3.12; production search/inference uses the Rust backend
>   and does not need it). Install the `hcpe` extra before running this command:
>   `uv sync --extra hcpe` (dev) or `pip install 'maou[hcpe]'` (wheel). Running
>   `hcpe-convert` without it raises an `ImportError` pointing here.

置換案:

> - CSA/KIF parsing uses the in-house Rust backend (`maou_shogi::kifu`,
>   exposed as `maou._rust.maou_shogi.parse_csa_str` / `parse_kif_str`).
>   No extra dependency is required — `hcpe-convert` works on a base
>   install. The parser is parity-verified against the previous
>   cshogi-based implementation (`rust/maou_shogi/tests/kifu_parity.rs`).
>   The legacy `hcpe` extra (cshogi) is retained only for backward
>   compatibility and fixture regeneration; production code does not
>   import it.

### 2. docs/rust-backend.md — maou_shogi モジュール表への kifu/ 追加

モジュールツリー (moves.rs 行の付近) に追記:

```
        ├── kifu/           # CSA/KIF 棋譜パーサ (cshogi parity 検証済み)
        │   ├── csa.rs      #   CSA V2.2 (P+/P-/AL は spec 準拠の独自拡張)
        │   ├── kif.rs      #   KIF 柿木形式 (不成対応，BOD は明示エラー)
        │   └── record.rs   #   GameRecord + cshogi 互換 move エンコード
```

「cshogi 互換性に関する設計判断」節に追記:

- **棋譜パーサ**: CSA/KIF パーサは独自実装 (maou_shogi::kifu)．出力
  moves は cshogi の 32-bit エンコーディング (to | from<<7 |
  promote<<14 | 移動前駒種<<16 | 捕獲駒種<<20，打は from=81+idx) と
  bit-exact．golden fixture parity (tests/kifu_parity.rs，oracle =
  cshogi 0.9.7) で置換の等価性を実証済み．

## 意図的な cshogi との差分 (docs には注記のみ，挙動は実装済み)

1. KIF「不成」を正しく解釈 (cshogi は黙って読み飛ばし以降の盤面が壊れる)
2. KIF の BOD (局面図) 初期局面は明示エラー (cshogi は黙って平手扱い →
   誤った学習データ混入を防止)
3. KIF `scores()`/`comments()` は moves と同長に整列 (cshogi 版では
   KIF の HCPE 変換が 1 行も出力されない不具合があった)
4. KIF 開始日時ヘッダから partitioningKey を導出 (従来 None 固定)
5. CSA P+/P-/持駒 00/AL は V2.2 spec 準拠で対応 (cshogi は segfault)

## Evidence

- Rust parity: `cargo test -p maou_shogi --test kifu_parity` (21 fixtures)
- Python 交差検証: tests/maou/domain/parser/test_{csa,kif}_parser.py の
  TestCshogiCrossCheck (cshogi と直接比較)
- KIF end-to-end: tests/maou/app/converter/test_hcpe_converter.py::
  TestHCPEConverter::test_successfull_kif_conversion
- QA: ruff/isort/mypy/pytest (777 passed) / cargo clippy 0 warnings
