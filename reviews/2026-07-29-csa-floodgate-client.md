---
title: CSA サーバ transport (floodgate 対局) の追加とドキュメント
date: 2026-07-29
status: pending
target:
  - docs/commands/floodgate.md
  - docs/design/usi-engine/index.md
risk: low
reversibility: easy
---

# 提案: CSA サーバ transport を durable doc に反映する

## Trigger

user 指示「maou コマンドで floodgate での対局をできるようにする．最低でも
ログイン名を取る．パスワード未指定ならランダム生成して出力，指定ありなら
それを使う (再開の意味)．連続対局に対応．1 局ごとに floodgate から
ログアウトされるので注意」．

設計 `docs/design/usi-engine/index.md:106` は当初から

> 電竜戦本戦は CSA → 将来 CSA transport を agent 無変更で追加できる

と書いており，`rust/maou_usi/src/lib.rs:13` にも同じ根拠が置かれていた．
今回その「将来」を実装したので，**予定として書かれている記述を実在する
機能の記述へ更新する**必要がある．

## 実装 (051243a)

`rust/maou_usi/src/csa/` を `stdio.rs` と同じ層に追加した．`Agent` は
**無変更**で，設計の想定どおり transport だけが増えた．

- `csa/protocol.rs` — CSA 行 ⇔ 型付きメッセージ (pure)
- `csa/client.rs` — TCP セッション + 対局ループ + 連続対局の再接続
- `maou floodgate` コマンド (console → interface → app → PyO3)

## 一次資料で確定した floodgate 仕様

実装前に <http://wdoor.c.u-tokyo.ac.jp/shogi/> と CSA プロトコル
ver 1.2.1 (<http://www.computer-shogi.org/protocol/tcp_ip_server_121.html>)
から確定させた．**記憶で書くと外す部分**なので docs に残す価値がある:

| 事項 | 内容 |
|---|---|
| 接続先 | `wdoor.c.u-tokyo.ac.jp:4081` |
| ログイン名 | 任意 (**事前登録不要**)．重複回避のためオリジナルな名前を |
| パスワード欄 | **`floodgate-300-10F,<trip>`** — ゲーム名を埋め込む規約．trip は同名ユーザを区別する仕組み |
| 対局の組まれ方 | **毎時 0 分と 30 分** |
| 対局後 | **ログアウト状態に戻る → 再接続する** |
| 持ち時間 | 300 秒 + 10 秒加算 (Fischer)．**512 手で引き分け** (2024-01-07 に 256 手から拡張) |
| レーティング | 15 試合程度で計算される |

「ログイン名 + trip が同一性の単位」という user の理解は正しく，それが
floodgate 公式の `trip` の定義そのものだった．

## 実機検証 (2026-07-29 実施．goal 達成)

DevContainer (CPU 推論 / ViT 19.8M fp16 / batch 8 / threads 1) から実 floodgate へ
接続し，**2 局連続**で対局した．

| | 対局 1 | 対局 2 |
|---|---|---|
| 開始 | 09:30:06 | 10:00:07 |
| 相手 | `910` | `komadokun_depth5` |
| 手番 | 先手 | 先手 |
| 結果 | **勝ち** (#RESIGN) | **勝ち** (#RESIGN) |
| 手数 | 91 | 81 |
| 所要 | 789 秒 | 683 秒 |

三者一致で盤面同期とプロトコル適合を確認した:

1. クライアントの報告 — 91 手 / win (#RESIGN)
2. サーバの棋譜 — `'summary:toryo:maou_test win:910 lose`
3. **自前の CSA パーサ** (`parse_csa_str`) で再読込 — 91 手 / `%TORYO` / win=1

確認できた挙動:

- `LOGIN <name> floodgate-300-10F,<trip>` が通り `LOGIN:maou_test OK` を受領．
- 待機中の keep-alive (空行) にサーバが空行で応答する．
- **対局後にログアウトして再接続する連続対局が実機で回った** (`> LOGOUT` →
  次の接続 → 再ログイン → 次の枠で対局)．
- 対局条件のパースが実際の floodgate 通知と一致
  (300 秒 / 加算 10 秒 / 秒読み 0 / 最大手数 512)．
- 時間配分: 初手 33 秒 (hard 予算) → 以降 16 秒/手に収束．1 手 16 秒消費・
  10 秒加算で残り時間は減るが，予算式 (`残り/40 + 加算`) が残量に比例して縮み，
  `ceiling = 残り − マージン` が上限を押さえるため**時間切れは起きなかった**．

### 環境の罠 (再発防止)

`uv run pytest` 等の**暗黙 re-sync が maturin ビルドの `.so` を onnx feature
なしで上書きする**．最初の実行が
`this build has no onnx feature; ModelPath is unavailable` で落ちた．
compass の「暗黙リビルドは debug」と同種で，**feature も落ちる**．
対策は再ビルド後に `uv run` を挟まず `.venv/bin/maou` を直接叩くこと．

## 設計判断 (durable にする価値があるもの)

- **持ち時間の責務分界は USI と同一**．CSA は対局開始時に持ち時間規定を
  一括通知し，以後は指し手に消費時間 `,T<n>` を付ける (USI の毎手
  `go btime wtime` とは別形式)．transport が残り時間を追跡して
  `ClockParams` へ写し，予算配分は既存 `TimeStrategy` が行う．
  VETO「持ち時間の消費計画は別レイヤー」に整合．
- **消費時間はサーバ計測が正**．クライアントの実測ではない (遅延時間の
  控除等をサーバが行う)．仕様 3.2.2 が「クライアントはサーバが示した
  消費時間を時間計算に使用すればよい」と明示している．
- **`clock_margin_ms` を CSA 経路にも適用した**．compass invariant
  「自己対局の対処が USI へ横展開されていなかった前例あり．片方を直したら
  必ずもう片方を見る」に従い，**3 本目の経路**として最初から入れた．
- **局面パーサを二重に作らない**．`BEGIN Position` は既存の CSA 棋譜パーサ
  (`maou_shogi::kifu::parse_csa_str`) に委譲する．golden 検証済みの実装を
  再利用し，CSA 局面表記の 2 つ目の実装を持たない．
- **指し手の解決は合法手照合**．USI 経路の
  `generate_legal_moves(...).find(|m| m.to_usi() == usi)` と同じ規約を
  CSA 表記で行う．非合法手・盤面ずれがその場で検出される．
- **keep-alive は仕様下限 30 秒を機構的に守る**．CSA 仕様 3.4 は
  「30 秒を経ずして送ってはならない．違反はサーバが反則負けにできる」と
  定めるので，設定値が下回っても切り上げる．
- **ponder は使わない**．CSA には `ponderhit` に相当する信号がない．

## 提案する docs 変更

1. **`docs/commands/floodgate.md` (新規．051243a で追加済み)** —
   `check-cli-docs` フックが新規 CLI と同一コミットでの追加を機構的に
   要求するため先行して入っている．本 review はその審査を兼ねる．
2. **`docs/design/usi-engine/index.md`** — §4 の
   「将来 CSA transport を agent 無変更で追加できる」を**実装済みの記述へ
   更新**し，`csa` モジュールをレイヤー表に追加する．上記「一次資料で確定
   した floodgate 仕様」表と責務分界を短く記載する．

## Risks

- **floodgate は非公式仕様に依存する**: 低 / 保守 / ゲーム名
  `floodgate-300-10F` と対局時刻はサーバ運用者が変更し得る．CLI オプション
  で上書きできるようにしてあるので追随は容易．
- **持ち時間の残量がサーバとずれ得る**: 低 / 棋力 / 加算の適用時点の解釈
  差．サーバの `,T<n>` を正として引くので蓄積はしない．
- **CPU 実行では棋力が低い**: 中 / レーティング / DevContainer で 22 p/s．
  レーティングは名前ごとなので，計測用の名前と本番用の名前を分ける運用が要る．

## 判定

user 承認後に `docs/design/usi-engine/index.md` を更新して applied 化する．
