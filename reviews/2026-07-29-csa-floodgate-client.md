---
title: CSA サーバ transport (floodgate 対局) の追加とドキュメント
date: 2026-07-29
status: applied
applied_in: 8221edd
target:
  - docs/commands/floodgate.md
  - docs/design/usi-engine/index.md
  - docs/rust-build-optimization.md
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

### レーティングの所在 (CSA プロトコルは運ばない)

user 質問「レーティングは何から何になったか．CSA サーバから送られてくるか」への実測回答:

- **live の CSA プロトコルはレーティングを一切運ばない**．`Game_Summary` に
  該当フィールドがなく，仕様 ver 1.2.1 にも定義がない．
- レーティングが載るのは (a) **公開棋譜のコメント行** `'black_rate:` /
  `'white_rate:` と (b) wdoor のレーティングページ．前者は既存の CSA パーサが
  `GameRecord.ratings` に読み込む．
- 今回の 2 局とも **`'black_rate:` (maou_test 側) の行が無い** = 対局時点で
  **未レーティング**．相手側は記録があり `910:2098` / `komadokun_depth5:2045`．
  レーティングページ (`players-floodgate-20260729.html`) にも maou_test は未掲載．
  公式が「15 試合程度でレーティングが計算される」と書いているとおり，2 局では付かない．
- **識別子はサーバ側で `<ログイン名>+md5(<trip>)`**．棋譜の `'rating:` 行が
  `maou_test+5cb2fc16ffe6289288402b1441a046ee` で，これは生成した trip の
  MD5 と一致した (実測)．**同一性が継続しているかを外部から検証できる**．

### Python CLI 層のレスポンス影響 (測定: 影響なし)

user 質問「Python CLI を介すオーバーヘッドは自動対局のレスポンスに影響しないか」:

- Python は `run_csa` を**起動時に 1 回呼ぶだけ**で，PyO3 が `py.detach` で
  **GIL を解放**して以降は戻らない．ソケット I/O・プロトコル解析・探索・着手送信は
  全て Rust 側で完結し，**1 手ごとの経路に Python は存在しない**．
- 起動コスト (import + click) は **0.09-0.10 秒**で対局開始前に 1 回だけ．
- 実測の裏取り: サーバが計測した消費時間 `T` と内部予算 (`TimeStrategy` の
  soft/hard) を対局 1 の自分 46 手で突き合わせたところ，**hard 超過は 1 手・
  最大 0.1 秒**だった．Python が手番ごとの経路にいれば `T` は系統的に予算を
  上回るはずで，そうなっていない．

### 生成 trip の出力先を stdout に変更

当初 stderr に出していたが，**`> out.txt` で通信ログを捨てる運用で
生成 trip が消える**．trip は失うと同じ識別子に戻れない唯一の出力なので
stdout へ移した (docs/commands/floodgate.md も追随．回帰テストで固定)．

### 環境の罠と，その機構的な解消 (user 指示 2026-07-29)

`uv run pytest` 等の**暗黙 re-sync が maturin ビルドの `.so` を onnx feature
なしで上書きする**．最初の実行が
`this build has no onnx feature; ModelPath is unavailable` で落ちた．
compass の「暗黙リビルドは debug」と同種で，**feature も落ちる**．

user 指示「maturin ビルドはデフォルトで onnx feature を入れる．その方が
暗黙的な事故を減らせる」に従い，**`pyproject.toml` の
`[tool.maturin] features` を `["pyo3/extension-module", "onnx"]` に変更**した．
検証: 事故が起きた経路 (`uv run maou floodgate --model-path ...`) を再実行し，
モデルのロードが通って TCP 接続失敗まで到達することを確認した．

- CI は features を明示指定しており (linux は `onnx-cuda,onnx-tensorrt`，
  windows は `onnx`)，いずれも既定と同じか上位集合なので影響しない．
- `cargo test` / `cargo clippy` は pyproject を読まないので従来どおり
  feature なしで速いまま．
- cuda/tensorrt は HW 依存なので**既定に入れない** (実行時 opt-in を維持)．
  可搬性 VETO は「HW/EP は runtime gate のみ」であり，ONNX の CPU 実行を
  既定に含めることはこれに抵触しない (配布 wheel は既に onnx 込みの単一 wheel)．

**この変更が生む副作用を明示する**: 従来 onnx なしのビルドは
「ModelPath 使用時に大きな声で落ちる」状態だったが，既定に入れたことで
「動くが debug のまま約 6 倍遅い」に変わり得る．**声の大きい失敗を
静かな性能劣化に取り替えない**ため，`run_csa` の起動時に
`cfg!(debug_assertions)` で debug ビルドを名指し警告するようにした
(compass の TRIPWIRE「性能数値を報告する前に release か確認」の機構化)．

## 追加で提案する docs 変更

3. **`docs/rust-build-optimization.md:115`** — 「`--features
   pyo3/extension-module,onnx` (省くと pure Rust ビルドになり…)」は
   maturin 経由では既定に含まれるようになったため記述を更新する
   (`cargo` 直叩きでは依然 feature 指定が要る点は残す)．

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

**user 承認 (2026-07-29)．8221edd で適用済み**:

- `docs/design/usi-engine/index.md` — レイヤー表に `csa` を追加し，
  「将来 CSA transport を追加できる」を「追加できた (agent 変更 0 行)」へ更新．
  §4.1 として floodgate の接続仕様と責務分界を追記
- `docs/rust-build-optimization.md` — maturin の既定 features が onnx を
  含むようになったため「features は既定では付かない」を実態へ更新
- `docs/commands/floodgate.md` — 051243a で先行追加したものを本 review が審査

同コミットで `.claude/skills/gh-pr/SKILL.md` の禁止事項も実態へ修正した
(Co-Authored-By トレーラと PR 本文の footer はどちらもリポジトリの確立した
慣行であり，禁止は誤りだった)．**`AGENTS.md:247-248` に同じ誤った規定が
残っている** — governance doc なので別途承認を得てから直す．
