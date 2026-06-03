---
status: applied
applied_in: 54b3af5
date: 2026-06-03
target: [docs/memory-architecture.md, CLAUDE.md, .claude/commands/resume-context.md, .claude/commands/checkpoint-context.md]
risk: low
reversibility: trivial
---

# Campaign Compass — 常時ロード・curated な不変則＋北極星指標レイヤ

## Trigger

数セッションをまたぐ「絶対に忘れてはいけない指針」が脱落する構造的欠陥．具体例:

1. `mid_v2 頭打ち` を過去 worklog に記録済みにもかかわらず，数セッション後に再び
   mid_v2 での改善を提示してしまった（ユーザ報告）．`scratchpad/current.md:71-73`
   の「## 構造分析の訂正 (ユーザ指摘)」がこの是正を記録している:
   「mid_v2 を捨てたのは unit-16 頭打ちが理由 → 『mid_v2 に戻る』は誤り」．
2. 各セッションで成果をドライブした比較指標（KH gap, unique-count 等）が，その
   セッションで重視されないと worklog に残らず失われる．

根本原因: `/resume-context` は worklog を **最新 1 ファイルしか読まない**ため，古い
worklog にある結論は二度とロードされない．`current.md` は毎 checkpoint **全上書き**
されるため，横断的な不変則は毎回再タイプしない限り脱落する．現に `current.md` には
「## ⚠ 鉄則」「## 構造分析の訂正」という *ad-hoc な invariant セクション* が手動で
維持されており，正しい置き場所が無いことの証左になっている．

## Proposed change

新レイヤ `scratchpad/compass.md`（gitignored, working-memory ファミリ）を追加し，
常時ロード・curated（上書きでなく剪定）で運用する．本体は seed 済み
（North-star スコアボード＋Invariants の 2 セクション，上限 ~45 行）．以下 4 ファイルを編集:

### 1. `docs/memory-architecture.md`

`## Files` テーブルに 1 行追加（`worklog` 行の直前）:

```
| `scratchpad/compass.md` | Curated campaign invariants + north-star metrics. Always loaded; prunable. | no — `.gitignore`d |
```

新サブセクションを `## current.md shape` の後に追加:

```markdown
## compass.md — the curated layer

`current.md`（毎回全上書き）でも worklog（immutable・1 つだけロード）でもない
**第 3 のモード = curated**．常時ロードされ，編集・削除で剪定される．

設計空間 (mutable↔immutable) × (常時ロード↔オンデマンド) の，これまで空いていた
「mutable・常時ロード・campaign を貫く durable」セルを埋める．

2 セクションのみ:
- **North-star**: goal / target / active metric / 現状最良 / 残ギャップの性質．
  毎 checkpoint で数値を更新するか "unchanged" を明記する（指標を失わないため）．
- **Invariants**: 「破ると 1 セッション無駄になる」do-not-redo 結論．各項に Est.（出所）
  を付け，evidence で覆ったら **delete/edit** する（append しない＝墓場を作らない）．

不変則 2 種の寿命の違い: Invariants は *sticky だが evidence で上書き可*，
North-star の値は *毎回更新される*．

肥大化対策（必須）:
1. ハード上限 ~45 行 / 不変則 ~12 件．超えたら最も load-bearing でない項目を evict．
2. スロット正当化: 「破ると 1 セッション無駄になる」もののみ．豆知識は worklog/MEMORY 行き．
3. curation は checkpoint の必須ステップ（後述）．
4. supersede = 削除．覆れた不変則は消す．「覆れた事実」を残すのは将来また再追加されて
   しまう場合のみ 1 行で．
```

`## Anti-patterns` に追加:

```
- compass.md を append-only で扱う（剪定せず積み上げる）．
- 覆れた不変則を残して「墓場」を作る．
- North-star の数値を checkpoint で更新し忘れる（"unchanged" すら書かない）．
```

`## Token budget guidance` テーブルに 1 行追加:

```
| `scratchpad/compass.md` | ~45 | always loaded |
```

### 2. `CLAUDE.md`

`## Repository-Centric Memory Architecture` の Files テーブルに 1 行追加:

```
| `scratchpad/compass.md` | Curated campaign invariants + north-star metrics. Always loaded; prunable. | no (`.gitignore`d) |
```

同セクションの `### MUST rules` に 2 項追加:

```
- MUST load `scratchpad/compass.md` at `/resume-context` and treat its
  Invariants as binding guardrails（評価は SHA 基準で値の staleness を意識）．
- MUST curate `scratchpad/compass.md` at every `/checkpoint-context`:
  North-star 数値を更新（or "unchanged"），新規 do-not-redo 結論を追加，覆れた
  不変則を delete/edit，上限超過時は evict．append-only にしない．
```

### 3. `.claude/commands/resume-context.md`

`### 2. Read in parallel` のリストに追加:

```
- `scratchpad/compass.md`（full, if it exists — 常時ロード）
```

`### 5. Emit a compact brief` のテンプレに `### Focus` の直後へブロック追加:

```
### Charter in force [from compass.md]
- Active metric + current best + gap（値は 1 checkpoint stale の可能性）
- Invariants（再実行禁止の guardrail を 1 行ずつ）
```

`### 4. Classify` に注記: compass.md の Invariants は binding guardrail として扱い，
North-star の値は Assumed（SHA で staleness 判定）．

### 4. `.claude/commands/checkpoint-context.md`

`### 3. Refresh scratchpad/current.md` の後に新ステップを挿入（以降を繰り下げ）:

```markdown
### 3.5. Curate scratchpad/compass.md

上書きではなく**剪定**する:
- North-star: active metric の現在値・現状最良・KH gap を更新．変化なしなら "unchanged" を明記．
- Invariants: 今セッションが do-not-redo 結論を確立したら追加（Est. に worklog/SHA）．
  今セッションが既存不変則を **覆した** なら該当行を delete/edit（append しない）．
- 上限 ~45 行 / 不変則 ~12 件を超えたら，最も load-bearing でない項目を evict．
- 末尾 `## Last curated:` を現 SHA @ JST 日付に更新．
```

`### 6. Print compact status` に 1 行追加:

```
- compass.md curated: scoreboard updated / N invariants added / N removed
```

## Motivation

worklog の「1 つだけロード」と current.md の「全上書き」という既存 2 レイヤの性質が，
campaign を貫く不変則を構造的に脱落させる（Trigger の mid_v2 再提案がその実害）．
比較指標も同様にセッション間で散逸する．compass は「常時ロード・curated・prunable」
という欠けていたセルを 1 ファイルで埋め，再提案バグと指標散逸の両方を断つ．

## Alternatives considered

1. **current.md に「鉄則」節を正式化**: 既に手動で存在するが，current.md は毎回全上書き
   かつ意味的に「現在の状態」．cold セッションがゼロから再導出しない限り不変則は脱落する．
   curated（部分剪定）モードを current.md に混ぜると「全上書き」契約が壊れる．→ 棄却．
2. **CLAUDE.md / reviews に書く**: CLAUDE.md は常時ロードだが編集に reviews 経由の承認＋
   ユーザ apply が要り，metric を毎 checkpoint 更新する摩擦に耐えない．かつ project 全体
   ルールであって transient campaign のスコアボードではない．reviews は on-demand で
   常時ロードされない．→ 棄却．
3. **worklog にセクション追加**: worklog は immutable かつ resume が 1 つしか読まない．
   不変則を入れても古い worklog 化して消える（＝現状の欠陥そのもの）．→ 棄却．

## What this enables

- 数セッション後でも「mid_v2 に戻らない」等の確定結論が常時ロードされ，再提案を防ぐ．
- 北極星指標（KH 19,270 / unique-count gap）が単一スコアボードに集約され，どのセッション
  が何で進んだかが散逸しない．

## What this constrains

- checkpoint で compass の curation が**必須**になる（怠ると stale 化）．
- 不変則の追加・削除には Est.（出所）と「覆す条件」を書く規律が要る．
- compass は ~45 行に強制され，肥大したら evict を迫られる（意図的な摩擦）．

## Rollback plan

- `scratchpad/compass.md` を削除（gitignored, 他へ影響なし）．
- 4 ファイルの追記を revert（いずれも追記のみ・削除なし，trivial）．
- 既存の resume/checkpoint フローは compass 行を無視すれば従前通り動作する．
