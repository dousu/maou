---
status: applied
applied_in: d6097e0
date: 2026-07-27
target:
  - docs/design/usi-engine/verification.md
risk: low
reversibility: trivial
---

# 提案: 偽証明の GPU 追認結果を verification.md へ反映する

## Trigger

`fix(dfpn)` f967499 (PR #408) の後，**前回 126 件を出したのと同一コマンド**で
`MATE1PLY_VERIFY=1` を付けた 40 局を Colab L4 で実行し，偽証明が消えたことを
実測した (2026-07-27 19:35-20:37 JST)．

```
sound_checks TOTAL  : 28,177,735     (検証器の発火件数 = 非空性)
FALSE_MATE / FALSE MATE 行 : 0 / 0
STRICT VERIFY None  : 0              (同一コマンドで前回 126 件)
```

これにより verification.md §8.5 の記述が **2 箇所とも事実と食い違う**状態に
なった．

1. 「**原因候補は TT 再利用による GHI**」— 誤り．TT は solve をまたいで再利用
   されず (毎 solve 新規確保)，再現は fresh solver・2 ノードで単独成立した．
   真因は `can_interpose_bb` の二歩判定が**移動前**の歩ビットボードを見ていた
   staleness (王手手自身が守備の歩を取るとその筋の二歩が解け，取られた歩を
   数えて「歩合は非合法」と誤判定 → 偽の 1 手詰)．
2. 「**未解決**」— 解決済み．しかも「実害は出ていない」も不正確で，同じ
   staleness が移動合駒の要員を過大計上して**本物の詰みを取りこぼして**いた
   (回帰局面で `L*1d` の 3 手詰を回収)．

同じ run で **throughput の水増しが従来記録より桁で悪い**実測も得られたため，
既に §4.2 にある tripwire に実測値を添える (下記 (c))．

## ドキュメント変更内容 (本レビューの承認対象)

### (a) §8.5 見出しの変更

`## 8.5 既知の課題 (検証中に判明．未解決)`
→ `## 8.5 既知の課題 (検証中に判明)`

理由: 3 項目のうち 1 つが解決したため，見出しの「未解決」が全体にかかる形で
不正確になる．各項目側に状態を持たせる．

### (b) §8.5 の dfpn 項目を差し替え

現行:

> - **dfpn の偽証明アラート** — `[dfpn] STRICT VERIFY None (偽証明/不完全)`
>   が 40 局で 9 件，同設定の追試で 126 件．**実害は出ていない** (STRICT 検証
>   が最終権威なので `Unknown` に落ち，偽の詰みは指されない) が，探索が pn=0 を
>   誤って立てる頻度としては看過できない．原因候補は TT 再利用による GHI．

変更後:

> - **dfpn の偽証明アラート — 解決済み (f967499 / PR #408)**．
>   `[dfpn] STRICT VERIFY None (偽証明/不完全)` が 40 局で 9 件，同設定の追試で
>   126 件出ていた．真因は `can_interpose_bb` の二歩判定が**移動前**の歩
>   ビットボードを見ていたこと (王手手が守備の歩を取るとその筋の二歩が解ける)
>   で，TT 再利用による GHI ではない (TT は solve をまたいで再利用されない)．
>   soundness 違反 (偽の 1 手詰) と完全性の欠落 (本物の詰みの取りこぼし) の
>   両方を起こしていた．**追認 (2026-07-27, 同一コマンドで 40 局)**:
>   `STRICT VERIFY None` **0 件** / `FALSE MATE` **0 件** /
>   検証器の発火 `sound_checks` **28,177,735 件**．
>   再発の監視は `MATE1PLY_VERIFY=1` を付けて自己対局を回す
>   (**`sound_checks` の総和が 0 でないことを必ず確認する** — 検証器が
>   発火していなければ「0 件」は無意味)．統計行は solve ごとに出るので
>   総和を取ること．`DFPN_STATS=1` は併用しない (成功時の行が solve ごとに
>   出て log が溢れる)．

### (c) §4.2 の throughput tripwire へ実測値を追記

現行の引用ブロック末尾 (「**サマリの `throughput:` が … 水増しを疑うこと**」)
の直後に 1 段落追加:

> 実測 (2026-07-27, L4 / 40 局 / 30s+0.5s / `checkmate` 39 局): `throughput:`
> **668,449 playouts/秒** = 物理上限 (約 11,000) の **約 61 倍**．実 NN 評価から
> 逆算すると本物の playout は全体の **約 1%** しかない．同じ分母を使う
> `subtree reuse:` の割合も **0.8%** へ潰れる (実測済みの 18-20% と乖離) ので，
> **reuse 率の異常な低さは水増しの二次シグナルとして使える**．

### (d) §1 に wheel 入れ替えの注意を追記

`--threads 1 が最適` の箇条書きの前に 1 項目追加:

> - **wheel を入れ替えるときは `--force-reinstall --no-deps` を併用する**．
>   Rust のみの修正では `pyproject.toml` の版数が動かないため，Release
>   `latest` が更新されても pip が「同版数」と見なして**入れ替えないことが
>   ある**．修正前バイナリで計測して誤結論を出すのを防ぐため，dfpn の挙動が
>   関わる計測では先に preflight を通すこと (モデル不要):
>
>   ```
>   position sfen 1g1+N+N1+P1l/4+B4/4Np+P2/l1p1p1ppk/3PsnP1p/+r4P3/1pPG3PL/p2S1SGbP/SRK6 b GLPp 141
>   go mate 10000
>   ```
>
>   f967499 以降は `checkmate 5b4a 1d1c L*1d` を返す．`checkmate timeout` が
>   返ったら**修正前の wheel** なので入れ替えをやり直す (この局面は
>   `test_no_false_proof_when_check_capture_clears_nifu` で pin 済み)．

## 代替案と棄却理由

- **§8.5 の dfpn 項目を丸ごと削除する**: 棄却．126 件 → 0 件という追認の
  数値と，再発監視の回し方 (非空性の確認を含む) が失われる．偽証明は
  soundness 事案なので，解決の**根拠**が辿れる形で残す価値が大きい．
- **(c) を §5 (aggregator) 側に書く**: 棄却．水増しは throughput を読む
  すべての節に関わるゲートで，既に §4.2 に tripwire がある．同じ場所に
  実測値を足すのが整合的．
- **(d) を benchmarking.md §4 (インストール手順の本体) に書く**: 一案として
  有力だが棄却．§4 は `maou search` の North-star 計測向けで，preflight の
  局面は dfpn 固有．verification.md 側に置き，手順の参照関係 (§1 が
  benchmarking.md §4 を参照する形) は変えない．
- **何も書かない**: 棄却．「原因候補は TT 再利用による GHI」は **REFUTED
  済みの仮説**であり，放置すると次に読む人が最初にそこを掘る．

## リスクと理由

- **risk: low** — 1 ファイルの記述更新のみ．手順・既定値・コードは変えない．
- **reversibility: trivial** — 差し替えた段落を戻すだけ．

## ロールバック

`docs/design/usi-engine/verification.md` の §8.5 見出し・dfpn 項目を旧文へ戻し，
§4.2 の実測段落と §1 の箇条書き 1 項目を削除する．
