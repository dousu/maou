---
title: "dfpn テストは --test-threads=1 必須 (並列は OOM→SIGTERM) を CLAUDE.md に明記"
status: applied
date: 2026-06-22
branch: feat/tsume-solver
applied_in: 7f1eec0
---

## Trigger

worklog/2026-06-22-085309.md: `cargo test --release -p maou_shogi` (デフォルト並列) が
`signal: 15, SIGTERM` で死亡した．当初 V4PROF 変更の回帰を疑ったが，ユーザ指摘「シングルスレッドで
テストしていますか?」で真因判明 = **並列実行が dfpn の大 TT alloc で memory 制約 DevContainer (8GB) を
OOM させ kill**．`--test-threads=1` で **203 pass / 0 fail** (回帰無し)．

この罠は再発性が高い (assertion failure でなく SIGTERM ゆえコード回帰と誤認しやすい) ため，
CLAUDE.md の常設ルールに昇格して将来の混乱を防ぎたい．

## Proposal

CLAUDE.md の「### 重いテスト (Rust dfpn) — release ビルド必須」セクション (または Quick Reference の
Rust tests) に以下を追記する:

> dfpn テストは各々が大きな置換表 (TT) を alloc するため，**MUST `--test-threads=1`** で実行すること．
> default の並列実行は memory 制約 DevContainer (8GB) で OOM → `signal: 15 SIGTERM` となり，
> assertion failure でなくプロセス kill として現れる (コード回帰と誤認しやすい)．
>
> ```bash
> cargo test --release -p maou_shogi -- --test-threads=1
> ```

既存の `cargo test -p maou_shogi` 系の例にも `-- --test-threads=1` を付すことを検討．

## Risk / Scope

- ドキュメントのみ (CLAUDE.md)．コード非変更．
- 既存の SHOULD「Serena 並列禁止」「Agent Teams ≤2 (memory)」と同じ memory-constraint 系の運用ルール．
- リスク低．唯一の判断点 = MUST にするか SHOULD にするか (本提案は MUST)．

## Status note

status: pending — ユーザが CLAUDE.md を編集・commit し，SHA を applied_in に記入して applied 化する．
