---
title: 自己対局の棋譜を CSA で残し analyze-game に繋ぐ
date: 2026-07-30
status: pending
target:
  - docs/commands/selfplay.md
  - docs/commands/analyze_game.md
  - docs/design/usi-engine/index.md
risk: low
reversibility: easy
---

# 提案: selfplay の CSA 棋譜出力を doc に反映する

user 指示「selfplay 時の棋譜を csa か kif かで残せるようにして，analyze-game
での分析を行い，どこで差がついたのかを検証できるようにしてほしい」．

背景は手数カーブ (`--ab-mode timecurve`) の A/B が **A 13.5/40 = 33.8%，
−117 Elo [−229, −5]，t = −2.37** で負けたこと．**時間切れ負けは 0 件**
(resign 37 / checkmate 2 / repetition 1，終局時残り A 45.0s / B 71.5s) なので，
負け筋は「終盤の探索が薄くなった」か「中盤の追加探索が勝率に変換されなかった」
のどちらか．手ごとの分析でしか区別できない．

コード側は実装済み．doc 側の未反映分を本提案で埋める．

---

## なぜ KIF でなく CSA か

**KIF では往復できないから**．KIF パーサは初期局面を `手合割：` でしか読めず
BOD は明示的に非対応 (`rust/maou_shogi/src/kifu/kif.rs`)．`selfplay --sfen` で
任意局面から始めた対局は KIF に書いても読み戻せない．CSA は `P1..P9` +
`P+`/`P-` 持駒行で任意局面を表現でき，`analyze-game` / `analyze-gui` /
`hcpe_convert` の 3 つともが既に食える．

## 実装の要点

1. **writer はパーサと同じ場所に置いた** (`rust/maou_shogi/src/kifu/csa.rs`)．
   `parse_csa_str` が独立実装として往復テストの相手になる．任意局面 (持駒あり
   ・後手番)・成り・駒打ちを含む往復を固定した．
2. **`move_to_csa` の二重実装を解消した**．floodgate 実装時に
   `maou_usi/src/csa/protocol.rs` に置いた指し手 → CSA 変換を `maou_shogi` へ
   移し，transport 側は再輸出にした．**パーサ・writer・transport で 3 つ目の
   表記実装を持たない** (compass invariant「新 transport でパーサを二重に
   作らない」の writer 版)．golden parity は変更後も通る．
3. **終局行は勝敗が読み戻しても一致するように選んだ**．パーサは `%TORYO` /
   `%TIME_UP` / `%ILLEGAL_MOVE` を「手番側の負け」，`%KACHI` を「手番側の
   勝ち」と読む．driver の winner はいずれも「手番側が負ける」形で決まるので
   そのまま対応づく．連続王手の千日手は王手をかけ続けた側 (= 手番側) の負け
   なので `%ILLEGAL_MOVE`，最大手数は driver が課す上限であって持将棋では
   ないので `%CHUDAN`．
4. **1 局 1 ファイル**．`analyze-game` は複数局 CSA を拒否する．
5. **手ごとの計測を分けた**．CSA には `T<n>` (秒) と `'** <score>` を書き，
   **正確なミリ秒と手ごとの playout は JSONL 側**に置いた
   (`move_times_ms` / `move_playouts` / `move_scores`)．棋譜は標準形式のまま
   保ち，計測は失わない．棋譜本文は `--kifu-dir` 指定時に JSONL から取り除く
   (二重に持たない)．

## 分析でできるようになること

`analyze-game` のレポートは 1 手ごとに `record_time_s` / `record_score` を
載せるので，**対局時のエンジン評価**と**再解析の評価**を並べられる．
今回の A/B なら:

- 終盤 (ply 100 以降) に `winrate_loss` が A 側に偏るか → 「終盤の探索が薄い」
- 中盤に時間を積んだのに `winrate_loss` が改善していないか → 「中盤の追加
  探索が無駄」

**注意**: `record_time_s` は秒に丸まる．配分そのものを見るときは JSONL の
`move_times_ms` を使うこと．`parallel > 1` では壁時計が CPU 競合で歪むので，
時間配分の分析は `parallel = 1` で採った対局に限る．

---

# 提案する doc 変更

## 1. `docs/commands/selfplay.md`

- options 表に `--kifu-dir` を追加 (1 局 1 ファイル・CSA・analyze-game /
  analyze-gui / hcpe-convert が直接食える旨)．
- `## Output` のレコード一覧に `move_times_ms` / `move_playouts` /
  `move_scores` を追加し，**棋譜は秒・JSONL はミリ秒**という分担を書く．
  `--kifu-dir` 指定時に `csa` キーが JSONL に残らないことも明記．
- 分析への導線を Example に 1 つ足す (selfplay → analyze-game)．

## 2. `docs/commands/analyze_game.md`

- 入力の説明に「`maou selfplay --kifu-dir` の出力をそのまま解析できる」を
  追記．`record_time_s` / `record_score` が自己対局時のエンジン申告値で
  埋まることを書く (再解析値との比較が分析の主目的になるため)．

## 3. `docs/design/usi-engine/index.md` (selfplay driver の節)

- 「成果物は driver + 棋譜出力 + smoke まで」の記述を実装済みに更新し，
  **CSA を選んだ理由 (KIF は任意局面を往復できない)** と，
  **手ごとの計測を棋譜と JSONL に分けた理由**を残す．
- `move_to_csa` を `maou_shogi` へ移して transport と writer で共有した
  ことを invariant として書く．

---

# 未決 / 次

- **A/B の敗因はまだ特定していない**．本 PR は分析の道具までで，結論は
  棋譜を取り直して `analyze-game` にかけてから．
- 手数カーブの既定は **off のまま**．`-117 Elo` は「現行の一律配分の方が
  強い」という測定結果であって，カーブという発想自体の棄却ではない
  (山の位置・振幅は未探索)．結論を出すのは分析の後．
- 今回の A/B で **terminal spin が消費予算の 94.8%** だった．レバーとは
  独立の現象だが「中盤に時間を積んでも探索の質が伸びない」理由の候補で，
  分析時に併せて見る価値がある．
