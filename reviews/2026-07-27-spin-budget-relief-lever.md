---
status: pending
applied_in:
date: 2026-07-27
target:
  - docs/commands/selfplay.md
  - docs/commands/search.md
  - docs/design/usi-engine/verification.md
risk: low
reversibility: trivial
---

# 提案: 空回りの予算開放レバー (`--spin-relief` / `--ab-mode spin`) を docs へ反映する

## Trigger

空回りを分離計上したことで「空回りが予算を食っている」ことが数値で見えた
(終盤では消費予算の 9 割超)．そこで **空回りを playout 予算から外すレバー**を
実装した (既定 off，A/B 用)．CLI とサマリの表面が増えるため反映が要る．

停止性: 空回りを予算から外すと，終端しか無い領域 (深さ上限超過は
`mark_terminal` しないため証明で畳めない) で葉収集ループが永久に回る．
連続空回りが `SPIN_STREAK_LIMIT = 4096` に達したら新 `StopCause::SpinExhausted`
で止める設計にした (`test_spin_budget_relief_stops_instead_of_spinning_forever`
で pin)．

**CPU 事前スクリーニング (重要)**: レバーは発火する (空回り 10 倍) が，
**実 playout はほとんど増えない** (1 局 40 手 / 400 playouts / 開局 3 通りで
+1.34%，ばらつき +1.2〜+1.4%)．
空回りは「予算を奪っている」のではなく「その時点で開ける葉が無い」ことの
症状であり，予算を足しても葉は増えないため — 詳細は §4.5 (下記 (c))．

## ドキュメント変更内容 (本レビューの承認対象)

### (a) `docs/commands/selfplay.md` — CLI オプション表に 2 行

| Flag | Required | Description |
| --- | --- | --- |
| `--spin-relief/--no-spin-relief` | default off | Exclude terminal spin from the playout budget so `--playouts` counts real search volume only. Bounded by a consecutive-spin limit, so a frontier made entirely of terminals still stops (`stop=spin_exhausted`). Fixed-budget runs only. |

`--ab-mode` の説明へ `spin` を追加 (A = relief on / B = off，`--clock-ms` とは
併用不可 — 持ち時間モードは時計が拘束条件なので会計を変えても消費 wall clock が
変わらない)．

### (b) `docs/commands/search.md` — Stats 例と停止理由

- Stats 行の例へ `terminal_backprops=N` を追加．
- 停止理由の列挙へ `spin_exhausted` を追加 (`playout_limit` / `time_limit` /
  `pool_exhausted` / `root_terminal` / `root_proven` / `spin_exhausted`)．

### (c) `docs/design/usi-engine/verification.md` — §4.5 として新設

「空回りの予算開放 — CPU スクリーニングで棄却」節を追加する．内容:

- 測定表 (mock / 1 局 40 手 / 400 playouts/手 / 開局 3 通り):
  relief off = 実 playout 12,660 / 空回り 953，relief on = 実 playout 12,829
  (**+1.34%**) / 空回り 9,524 (**10 倍**)．
- レバーは発火しているが実 playout が増えない理由 (空回りは予算を奪っている
  のではなく「開ける葉が無い」ことの症状)．
- 期待効果 +1.34% ≈ **+1.2 Elo** で n=40 の検出限界 ~150 Elo の 2 桁下 →
  **GPU A/B を回さない判断**．走査 10 倍による wall clock +4% で期待値は負．
- 次に試すレバー (選択側の proven 子除外 = MCTS-Solver 相当) と，深さ上限超過
  だけは `mark_terminal` しないため別扱いが要ること．**実装後は同じ手順で
  発火量を先に測る**こと．

## 代替案と棄却理由

- **レバーを実装せず設計だけ書く**: 棄却．「効果が無い」ことは実装して測って
  初めて言える．実際，机上では「予算の 90% が空回り → 開放すれば実探索量が
  10 倍」と読めるが，実測は +1.34% だった．測定器としてのレバーは残す価値がある
  (選択側の対策を入れた後の再測定にも使う)．
- **レバーを既定 on にする**: 棄却．実 playout が増えないうえ走査が 10 倍に
  なり wall clock が伸びる (実測 +4%)．持ち時間モードでは純損失．
- **`--ab-mode spin` を持ち時間モードでも許可する**: 棄却．時計が拘束条件なので
  会計変更は消費 wall clock を変えず，「効果なし」を誤った理由で結論づける
  regime になる．CLI とバインディングの両方でエラーにした．

## リスクと理由

- **risk: low** — 既定 off の計測用トグル．既定経路の挙動は不変
  (静かな局面では空回り 0 = 完全な no-op であることを確認済み)．
- **reversibility: trivial** — フラグと `AbMode::Spin` を削るだけ．

## ロールバック

`--spin-relief` / `--ab-mode spin` / `SearchOptions::spin_budget_relief` /
`StopCause::SpinExhausted` を削除し，docs の該当記述を戻す．
