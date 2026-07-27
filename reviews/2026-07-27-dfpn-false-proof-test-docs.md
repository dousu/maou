---
title: CLAUDE.md の [SLOW] テスト表に偽証明ハンターを追加し，soundness スイープの回し方を明記
date: 2026-07-27
status: pending
applied_in:
target:
  - CLAUDE.md
risk: low
reversibility: trivial
---

# 提案: dfpn 偽証明ハンターと mate1ply soundness スイープを CLAUDE.md へ反映

## Trigger

`fix(dfpn): 王手手が守備の歩を取ると二歩が解けて偽 1 手詰になる不具合を修正`
(f967499) で以下の 2 つが増えた．どちらも **次に偽証明を追う人が最初に必要とする
道具**であり，見つけられないと同じ探索を一からやり直すことになる．

1. `rust/maou_shogi/src/dfpn/tests.rs::test_false_proof_hunt` — `**[SLOW]**` /
   `#[ignore]` の新規テスト．CLAUDE.md の `[SLOW]` 表は「`[SLOW]` フラグが
   ついているテストは全て `#[ignore]`」と宣言しているため，表に載らないと
   **宣言と実体が食い違う**．
2. `MATE1PLY_VERIFY=1` の検証カバレッジ修復 — 従来は production が通る 3 経路
   (fused look-ahead / near2 / cached full scan) を素通りしており，実際に
   soundness 違反が起きていたのに検出できなかった．今回全経路に挿したので，
   **偽 1 手詰の広域スイープが回せるようになった**ことを書き残す価値がある．

## ドキュメント変更内容 (本レビューの承認対象)

### CLAUDE.md §「重いテスト (Rust dfpn) — release ビルド必須」

**(a) `[SLOW]` テスト表に 1 行追加**:

| テスト名 | バジェット | 備考 |
|---|---|---|
| `test_false_proof_hunt` | 50 nodes/局面 (leaf-mate 相当) | 偽証明 (`STRICT VERIFY None`) ハンター．ランダム対局から局面生成．env `SEED`/`GAMES`/`PLIES`/`NODES`/`MAXHITS` |

**(b) 表の直後に soundness スイープの回し方を 3 行追記**:

> dfpn の **偽 1 手詰 (soundness 違反)** を疑うときは `MATE1PLY_VERIFY=1` を付ける．
> 1 手詰と申告した手を実 replay で検証し，偽なら `[mate1ply] FALSE MATE site=... m=... sfen=...`
> を出す (production が通る fused / near2 / cached full scan の全経路をカバー)．
> `test_false_proof_hunt` と併用すると広域スイープになる:
>
> ```bash
> SEED=1 GAMES=200 MATE1PLY_VERIFY=1 cargo test --release -p maou_shogi -- \
>   --test-threads=1 --ignored --nocapture dfpn::tests::test_false_proof_hunt
> ```

## 代替案と棄却理由

- **docs/design/tsume-solver/ 側に書く**: 診断 env を列挙した節が現状 docs/ に
  存在せず，新設すると本 PR の範囲を超える．CLAUDE.md の `[SLOW]` 表は既に
  「テストの回し方」を担っている節なので，そこへ寄せるのが整合的．
- **何も書かない**: (1) の表は「全て」と宣言しているため，放置すると規約と実体の
  乖離になる．棄却．

## リスクと理由

- **risk: low** — CLAUDE.md への追記のみ．既存の MUST 規約の変更・削除は無い．
- **reversibility: trivial** — 追加した表 1 行と直後の 3 行を消すだけ．

## ロールバック

CLAUDE.md の当該表から `test_false_proof_hunt` 行を削除し，直後に追記した
`MATE1PLY_VERIFY` の節を削除する．
