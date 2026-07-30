---
status: approved
title: analyze-game / analyze-gui に TensorRT teardown 回避を入れる
date: 2026-07-30
target:
  - docs/commands/analyze_game.md
  - docs/commands/analyze_gui.md
risk: low
reversibility: easy
---

# 提案: TensorRT 終了時の挙動を analyze 系 doc に明記する

## Trigger

GPU で `maou analyze-game --tensorrt` を実行した際，レポート出力の**直後**に

```
JSON report: report_g6.json
corrupted double-linked list
```

で落ちた．既知の「heap 破壊は TensorRT EP 固有」(compass invariant) の
発現で，`exit_skipping_teardown` による回避が **`analyze_game.py` と
`analyze_gui.py` にだけ入っていなかった**．

`floodgate.py` / `selfplay.py` / `usi.py` / `search_board.py` /
`evaluate_board.py` は全て呼んでいる．推論を行う console コマンドのうち
analyze 系 2 つだけが漏れていた (pre-existing)．

**結果は書き終わってから落ちるので気付きにくい**．今回も JSON は無事
だった．終了コードだけ見ている自動化があれば誤検知する．

## 変更 (コード側，doc 対象外)

- `analyze_game.py`: レポート出力後に `exit_skipping_teardown(tensorrt=...)`
- `analyze_gui.py`: サーバ停止後に同上
- `scripts/check-cli-docs.sh` の `CLI_DOC_MAP` に `analyze_game.py` /
  `analyze_gui.py` を登録 (未登録で，触ると "has no mapping" で
  pre-commit が必ず落ちる状態だった — `usi.py` / `selfplay.py` と同じ穴)

## 提案する doc 変更

`docs/commands/usi.md` が既に持っている「終了時の挙動」行と同じ形式で，
`analyze_game.md` と `analyze_gui.md` の CLI 表に 1 行ずつ追加する:

> TensorRT 有効時はレポート書き出し後に **destructor を経由せず終了**する．
> TensorRT EP の teardown が glibc heap を壊して abort するため
> (`verification.md §8.5`)．結果は既に書かれており，終了コードは 0 のまま．

## なぜ doc に書くか

`os._exit` は異常に見える実装なので，理由が doc に無いと将来「行儀の悪い
コード」として消される．usi.md に同じ行があるのはそのため．
