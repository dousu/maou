---
title: USI エンジン (対局エージェント) の設計
date: 2026-07-17
status: approved
# user 承認 2026-07-17 (未決事項 1-6 すべて提案どおり — 本文末尾「承認記録」参照)．
# docs/design/usi-engine/index.md + position-search/index.md の相互参照は承認直後に適用．
# docs/commands/usi.md は M1 実装 PR 内で適用予定．
target:
  - docs/design/usi-engine/index.md
  - docs/design/position-search/index.md
  - docs/commands/usi.md
risk: low
reversibility: moderate
---

# 提案: `maou usi` — USI 対局エージェント

## 背景 (user 指示 2026-07-17)

> - USI エンジンは対局エージェントとして作成する (プロトコル仕様と対局
>   エージェントを分離するかは設計判断)．持ち時間戦略や千日手戦略といった
>   戦略的なロジック実装が必要
> - コンピュータ将棋の大会での特殊ルール対応 (先手の持ち時間ハンデを実現
>   するために初期局面でお互い玉の無駄な移動をする，手数制限など) も戦略の
>   一部として対局エージェントで扱う
> - 目標は将棋所や ShogiGUI を使って対人間・対 USI エンジンで対局できる
>   状態．自己対局にも使えるようにパフォーマンス面での設計を行う
> - (user 主観) ponder を考えると Python は薄いラッパーでほぼ Rust 完結が
>   よさそう．探索以外のオーバーヘッドで棋力を落としたくない．どこまで
>   Python にするかは設計判断

USI 対応により自己対局・他将棋 AI との対局・人間との対局の土台を作る
(campaign goal，compass North-star)．

## USI プロトコル調査 (2026-07-17, ウェブ)

### 基本仕様

標準入出力の行指向テキストプロトコル．時間は全てミリ秒．

| 方向 | コマンド | 意味 |
|---|---|---|
| GUI→E | `usi` | 初期化要求．エンジンは `id name/author` + `option` 宣言列 + `usiok` を返す |
| GUI→E | `isready` | 対局準備要求．重い初期化 (モデルロード等) はここで行い `readyok` を返す |
| GUI→E | `setoption name X value Y` | オプション設定．`usiok`〜`isready` 間が基本だが再送もあり得る |
| GUI→E | `usinewgame` | 新規対局開始通知 |
| GUI→E | `position startpos [moves ...]` / `position sfen <sfen> [moves ...]` | 局面設定 (思考開始はしない) |
| GUI→E | `go btime B wtime W [byoyomi T \| binc b winc w]` | 思考開始 (持ち時間は先手 B/後手 W ms) |
| GUI→E | `go ponder ...` / `go infinite` / `go mate T` | 先読み思考 / 無制限 (検討) / 詰み探索 |
| GUI→E | `stop` | 思考中断．エンジンはその時点の `bestmove` を返す |
| GUI→E | `ponderhit` | ponder 的中通知．先読みを本探索に切替えて継続 |
| GUI→E | `gameover win\|lose\|draw` | 対局終了通知 |
| GUI→E | `quit` | 終了 |
| E→GUI | `bestmove <move> [ponder <move>]` | 指し手 (+ 相手の予想手)．特殊値 `resign` (投了) / `win` (入玉宣言勝ち) |
| E→GUI | `info depth/nodes/nps/time/score cp\|mate/pv/string ...` | 思考情報 (探索中に随時) |

ponder のシーケンス: エンジンが `bestmove X ponder Y` を返す → GUI は相手
番中に `position ... moves ... X Y` + `go ponder` を送る → 相手が Y を
指せば `ponderhit` (思考継続，時間計測はここから)，別の手なら `stop`
(bestmove を返させて読み捨て，新しい position/go をやり直す)．
やねうら王拡張では `ponderhit` に btime/wtime 等の残り時間を付加できる．

### 実装差・方言 (エンジン側で防御的に扱う点)

- **`info` の `pv` は行末尾に置くことが必須** — 電竜戦で pv が末尾でない
  エンジンのパース障害が実際に発生した
- `gameover` を送らず次の `position` を送る GUI がある (ShogiGUI の連続
  対局)．将棋所は `gameover` → `isready` → `usinewgame` の順．どの経路
  でも状態が回復するように作る
- `stop` への `bestmove` 応答は GUI 側で読み捨てられ得る．`stop` 受信は
  「即時中断シグナル」として扱い，どんな状態でも短時間で bestmove を返す
- 開始局面の `position startpos` は `moves` を含まない (将棋所実挙動)
- 検討機能で `stop` 後に `setoption` が再送される (ShogiGUI)
- エンコーディングは GUI により UTF-8/Shift_JIS が混在 (ShogiHome は
  UTF-8)．プロトコル出力は ASCII 安全に保つ (`info string` の日本語は
  避ける)
- `isready` の readyok までの目安は 30 秒程度 (USI2.0 案)．TensorRT の
  エンジンビルドはこれを超え得るため対策が要る (§設計 7)
- GUI へのエンジン登録は実行ファイル指定 (将棋所は起動引数を渡せない)．
  .bat/.cmd ラッパーでの登録が確立した慣行 → 引数なしで起動できる
  エントリポイントを用意し，設定は `setoption` で受けるのが標準的

## 大会特殊ルール調査 (電竜戦，2026-07-17 ウェブ)

第4回電竜戦本戦大会ルール (dr4_rule.pdf) より:

- **最大手数 512 手**．超えて勝敗未決なら引き分け (最大手数時の詰みも
  引き分け，宣言勝ち成立は勝ち)．引き分けは先手 0.4 勝/後手 0.6 勝
- **千日手は指し直しなしの引き分け**
- **入玉宣言 (27 点法，%KACHI)**: 宣言側の手番/玉が敵陣三段目以内/大駒
  5 点小駒 1 点で先手 28 点・後手 27 点以上 (対象は持ち駒+敵陣三段目以内
  の駒，玉除く)/敵陣三段目以内に玉を除き 10 枚以上/王手されていない/
  持ち時間が残っている — 全充足で勝ち，1 つでも欠けると宣言側の負け
- 持ち時間は**先手 5 分+2 秒加算/後手 10 分+2 秒加算の非対称フィッシャー**
  (先手有利補正)
- 本戦の対局は CSA サーバプロトコル ver 1.2.1 (shogi-server-dr) 経由．
  USI エンジンはブリッジクライアントで接続するのが通例
- **ハードウェア統一戦**では先手時間ハンデを「**最初の 4 手 =
  ▲58玉→△52玉→▲59玉→△51玉**」(通称: 玉の屈伸運動) で実現しており，
  この手順は**参加プログラム側で対応が必須** (指定局面戦 TSEC と同様)

→ user 指示の「玉の無駄な移動」「手数制限」はいずれも実在の大会規定．
汎用化すると (a) 強制序盤手順，(b) 最大手数，(c) 入玉宣言，(d) 引き分け
価値の非対称性，の 4 つが戦略パラメータとして必要．

## 既存資産と gap (2026-07-17 リポジトリ調査)

### 流用できる資産

- **千日手履歴規約は実装済み**: 「初期局面 SFEN + USI 経路」→
  `build_board_and_history` (rust/maou_rust/src/maou_search.rs:183)．
  探索中の千日手検出は `find_repetition` (rust/maou_search/src/repetition.rs:88，
  フル Zobrist 一致，距離 4 から 2 手刻み)．**連続王手の千日手 (かけた側の
  負け) も実装済み** — 探索が反則手を自然に回避する
- **詰み探索併走**: root-dfpn + leaf-mate が default on
  (rust/maou_search/src/search.rs:1023-1099)．詰みを見つけたら root が
  proven WIN 化して即停止 → 終盤の即詰み逃しに強い
- **永続評価器**: `SearchEngine` (PyO3) はモデル 1 回ロードで連続探索．
  `py.detach` で GIL 解放済み (maou_rust/src/maou_search.rs:540-546)
- **バッチ推論 + TRT キャッシュ**: `evaluate_batch` + `pad_to` 固定
  batch + `with_engine_cache` (rust/maou_search/src/onnx.rs)
- 最終手選択は robust child + 負け確定手除外 (search.rs:273)

### gap (新規実装が必要)

1. **探索の外部キャンセル API がない**: 停止は playout/time_ms 到達か
   内部条件のみ (`Shared::stop` は内部 AtomicBool, search.rs:367)．USI
   `stop`/`quit` 即応には外部から止める注入口が要る
2. **探索中の進捗/PV スナップショット取得がない**: `SearchResult` は
   完了後一括生成．USI `info` の随時出力に口がない
3. **木/TT の再利用がない**: 呼び出しごとに `NodePool::new`
   (search.rs:954)．ponder 的中時・連続手番での subtree 再利用は未実装
   (docs/design/position-search/index.md:254 が対局レイヤー導入後の主機構
   候補と明記)
4. **リアルタイム時間管理がない**: 既存 `BudgetAllocator`
   (src/maou/app/analysis/game_analyzer.py:94) は局面数既知の棋譜解析用
   等分配分．btime/byoyomi からの動的 1 手配分は新規レイヤー
5. **終局ルールロジックがない**: 入玉宣言判定 (27 点法)・最大手数・
   resign 判断はいずれも repo に存在しない
6. ONNX 推論は `Mutex<Session>` 直列化 (onnx.rs:75)．自己対局の並列度を
   上げる際のボトルネック候補 (GPU 律速なら影響小)
7. USI プロトコルループは repo に一切存在しない — 完全新規

## 設計

### 1. プロトコル層と対局エージェントの分離 — 分離する (設計判断)

新規 Rust クレート `rust/maou_usi/` を 3 モジュールに分ける:

| モジュール | 責務 | 依存 |
|---|---|---|
| `protocol` | USI 行 ⇔ 型付きコマンド (`GuiCommand`/`EngineCommand`) の parse/serialize のみ．IO なし・戦略なし | なし (pure) |
| `agent` | 対局エージェント = 状態機械 + 戦略モジュール群 + 探索セッション．イベント (型付きコマンド) を受けて応答を返す．transport 非依存 | maou_shogi, maou_search |
| `stdio` | stdin 読取りスレッド + stdout 書込み (行バッファ+flush) + dispatch | protocol, agent |

分離の根拠:
- `agent` を stdio なしで直接駆動できる → **自己対局 driver が同一
  エージェントを in-process 再利用** (user 要件のパフォーマンス設計の核)
- 電竜戦本戦は CSA プロトコル → 将来 CSA transport を agent 無変更で追加
  できる
- GUI 方言 (§調査) を protocol/stdio に隔離し，agent は clean に保つ
- 状態機械・戦略を fake transport で完全に単体テストできる

### 2. Rust/Python 境界 — プロトコル・エージェント・時間管理は全て Rust (設計判断)

user 方針どおり **Python は薄いラッパー**とする:

- Python 側は `maou usi` (click) が config (モデルパス・探索初期値・EP
  フラグ) を組み立てて `maou._rust.maou_usi.run_usi(config)` を呼ぶだけ．
  GIL を解放して Rust が stdin/stdout を専有し，`quit` で戻る
- **stdout はプロトコル専用**．Python logging は `maou usi` 起動時に
  stderr へ向ける (エンジンの絶対規約)
- GUI 登録用に引数なしエントリポイント **`maou-usi`** (console script) を
  追加する (`maou usi` のデフォルト構成で起動する alias)．モデルパス等は
  USI `setoption` で受けられるため引数なしで成立する

根拠: `stop`/`ponderhit` の即応は Rust reader スレッドで機構的に保証
(Python を経由すると GIL/GC がテールレイテンシに入る)．タイマー・ponder
のスレッド制御・毎手のループが全て Rust に閉じ，探索以外のオーバーヘッドが
ほぼゼロになる．自己対局も同じ理由で Rust ループが最速．Python は導入 UX
(uv/wheel/CLI) と将来の学習パイプライン統合だけを持つ．

### 3. maou_search への拡張 (前提工事)

1. **stop token 注入**: `SearchLimits` に `stop: Option<Arc<AtomicBool>>`
   を追加し，worker ループの既存 `stopped()` 判定に OR で合流させる．
   `StopCause::External` を追加
2. **無期限探索モード**: `SearchLimits { max_playouts: None, time_ms:
   None }` を「stop token でのみ停止」と再定義 (現行は
   DEFAULT_MAX_PLAYOUTS に丸められる)．`go ponder`/`go infinite` 用
3. **進捗スナップショット**: worker が一定 playout 間隔で root 統計
   (playouts/nps/max_depth/best 系列 PV/winrate) を `ArcSwap` 相当の
   snapshot 領域へ発行し，呼び出し側スレッドが随時読める口を作る →
   `info` 随時出力と，時間延長判断 (§5.1) の入力に使う
4. **千日手評価値の可変化**: 終端 Draw の 0.5 固定 (search.rs:681-693 →
   terminal_value) を `SearchOptions::draw_value` として可変にする
   (エージェントが手番視点に変換して渡す，§5.2)
5. (M3) **subtree 再利用**: root 前進時に旧木の該当 subtree を引き継ぐ．
   position-search 設計 doc の未決レバーを対局レイヤー要件として実装する
   (`Board::hash_after` が下地)．M3 で効果計測してから採否確定

1〜4 は movegen/dfpn の意味論に触れない見込みだが，search.rs に触れる
ため実装 PR では STRICT-VERIFY canonical (29te/39te) を RAN して照合する
(TRIPWIRE 遵守)．

### 4. エージェント状態機械

状態: `Booting → Idle → EngineReady → InGame { AwaitingGo, Thinking,
Pondering }`．主な遷移:

- `usi` → `id name maou <version>` + option 宣言列 + `usiok`
- `isready` → 評価器構築 + warmup (TRT ビルド含む) → `readyok`．初回
  以降の `isready` は健全性確認のみで即答．ビルドが長い場合に備え
  readyok まで 5 秒ごとの keep-alive 空行送出をオプションで用意
  (default off，USI2.0 案準拠．将棋所実機で挙動確認してから default 判断)
- `position` → `build_board_and_history` 規約そのまま (既存資産)
- `go` → SpecialRules 前処理 (宣言勝ち判定・強制手順，§5.3) →
  TimeStrategy が予算決定 (§5.1) → 探索スレッドへ依頼 → 完了/停止で
  `bestmove [+ ponder]`
- `stop` → stop token を立て，探索スレッドの合流を待って `bestmove`
  (目標: 受信から 100ms 以内)
- `go ponder`/`ponderhit` → §5.4
- `gameover` → ponder 停止・対局状態破棄．**来ない GUI があるため，
  `usinewgame`/新しい対局と矛盾する `position` でも自己回復する**
- `quit` → 探索停止 → プロセス終了 (取りこぼしなし)

探索は専用スレッドで実行し，reader スレッドは常にコマンドを受理できる
(stop/quit 即応の機構的保証)．

### 5. 戦略モジュール (agent 内 trait，全て Rust)

#### 5.1 時間管理 TimeStrategy

compass VETO「持ち時間の消費計画は別レイヤー — 1 局面探索は与えられた
予算内まで」に完全整合: TimeStrategy が clock → 予算変換の上位レイヤーで，
探索は受け取った予算を消費するだけ．

- 入力: `ClockState { my_time, opp_time, byoyomi, my_inc, opp_inc }`，
  手数，(M2 以降) 探索スナップショット
- 出力: `TimeBudget { soft_ms, hard_ms }`．soft = 通常の思考打切り目標，
  hard = 絶対上限 (時間切れ安全マージン込み)
- default 実装は三態: 秒読み型 (残時間/想定残り手数 + byoyomi − margin)，
  フィッシャー型 (残時間/想定残り手数 + inc − margin)，切れ負け型
  (残時間/想定残り手数 − margin，安全バッファ厚め)．非対称持ち時間
  (電竜戦の先手 5 分/後手 10 分) は my_time ベースで自然に扱える
- `NetworkDelay` オプション (ms) を margin に算入 (サーバ対局の伝送遅延は
  自分の消費時間に含まれる — 電竜戦ルール第 22 条)
- (M2) **時間延長**: soft 到達時に root best が不安定 (上位 2 手の訪問数
  拮抗・直近の best 交代) なら hard まで延長．判断材料は §3-3 の
  スナップショット．延長判断も TimeStrategy 側 = レイヤー不変
- 定数 (想定残り手数カーブ等) は実装時に自己対局で調整し worklog に記録

#### 5.2 千日手戦略 RepetitionPolicy

- 検出・反則 (連続王手) は探索実装済み (§資産)．戦略の実体は
  **引き分け価値の設定**: `DrawValueBlack` / `DrawValueWhite` オプション
  (千分率 spin，default 500 = 0.5)．エージェントが自分の手番・先後から
  探索へ渡す draw_value (§3-4) を決める
- 電竜戦の引き分け 0.4/0.6 勝はそのまま DrawValue 400/600 で表現できる
  (先手は千日手を避け，後手はやや許容する，が探索評価に一貫して効く)
- 優等/劣等局面の扱いは position-search 設計の未決レバーのまま据え置き
  (本設計のスコープ外)

#### 5.3 大会特殊ルール SpecialRules

- **OpeningScript** (`string` オプション，例 `"5i5h 5a5b 5h5i 5b5a"`):
  対局経路が script の prefix と一致している間は次の script 手を探索なしで
  即指し (時間消費最小)．相手が script を外れたら以後無効化．電竜戦 HWT の
  玉の屈伸・指定局面系の強制手順に対応．script 手が非合法な局面に来た場合
  も無効化して通常探索に落ちる (安全側)
- **MaxMovesToDraw** (`spin`，default 0 = 無効): 最大手数ルール．
  (a) 手数がリミットに到達する局面では宣言可否を必ず確認し，可能なら
  `bestmove win`．(b) リミットが近い (残り数手) 場合，探索予算を絞る
  (どうせ引き分けになる木に時間を使わない)．探索内で「リミット以降を
  Draw 終端扱い」する in-search 対応は効果を見て M4 で判断 (未決事項 4)
- **入玉宣言 DeclarationWin**: 27 点法チェッカーを `maou_shogi` に新規
  実装 (`Board::nyugyoku_declarable() -> bool`，宣言側手番・敵陣三段目・
  点数 28/27・枚数 10・王手なしの 5 条件．時間条件は agent 側)．自分の
  手番開始時 (go 受信時) に判定し，成立していれば探索せず `bestmove win`．
  将棋所も同条件 (CSA ルール準拠) で宣言勝ちを判定する
- **Resign** (`ResignValue` spin 千分率，default 0 = 投了しない):
  探索後の root winrate が閾値未満の状態が `ResignConsecutive` 手続いたら
  `bestmove resign`．対人間では GUI 側の投了操作もあるため default off

#### 5.4 ponder PonderPolicy

- `USI_Ponder` (check，default true) 宣言．GUI が off にすれば
  `bestmove` に ponder を付けない
- 予想手 = 自探索 PV の 2 手目 (PV 長 < 2 なら ponder なし)
- `go ponder`: 無期限探索 (§3-2) を開始．`ponderhit` → その時点から
  TimeStrategy で予算を計算し，soft/hard デッドラインを張り直して**探索
  継続** (木・playout はそのまま活きる = ponder の主利得)．やねうら王拡張
  の `ponderhit` 時刻パラメータも受理する
- `stop` (ponder 外れ) → 探索停止して bestmove 返却 (GUI は読み捨て)．
  外れ後の新 position は M1-M3 時点では作り直し探索，subtree 再利用
  (§3-5) が入れば旧木から引き継ぐ
- 将棋所文書にある「ponder を返さず勝手に先読み」方式 (stochastic
  ponder) は採らない (時間管理が GUI と乖離するため)．ただし agent を
  transport 非依存にしてあるので将来差し替え可能

### 6. 自己対局 (パフォーマンス設計)

- `maou_usi::selfplay` に in-process driver を置く: 1 対局 = agent 2 個
  (先後) を **stdio/プロセスなし**で直接駆動．プロトコル文字列の
  parse/serialize すら通らない (agent はイベント駆動なので)
- 評価器 (ONNX session + TRT キャッシュ) はプロセス内で 1 個を全対局で
  共有 (`Evaluator: Send + Sync` 前提は既存)．モデルロード/warmup は
  1 回きり
- 並列度: T スレッドで G 対局を並行．現状の `Mutex<Session>` 直列化
  (gap 6) が並列時の上限になるため，M4 で「複数探索からの評価要求を
  まとめてバッチ推論する aggregator」を効果計測付きで検討 (pad_to 固定
  batch と親和)．**先行投資はしない** (REFUTED の IoBinding 教訓 —
  計測してから)
- 終局判定 (宣言・千日手・最大手数・投了) は SpecialRules/Repetition を
  そのまま流用 = USI 対局と自己対局で意味論が一致する
- 学習データ (HCPE) 生成への接続は次 campaign (本設計は driver の骨組み
  + 棋譜出力まで)

### 7. CLI と USI オプション

- `maou usi` (click サブコマンド) + `maou-usi` (引数なし console script)．
  CLI フラグは `maou search`/`analyze-game` と同名の探索・EP passthrough
  (--model-path/--threads/--batch-size/--cuda/--tensorrt/--trt-cache-dir/
  root-dfpn 系/leaf-mate 系)．**CLI フラグ = 初期値，`setoption` が上書き**
- USI オプション (宣言順): `ModelPath` (filename) / `Threads` /
  `BatchSize` / `NodeCapacity` (spin，USI_Hash からの自動換算はせず独立．
  USI_Hash は受理して NodeCapacity 未指定時の換算に使う — ノード実サイズ
  から係数を実装時に決める) / `UseCuda` / `UseTensorRT` (check) /
  `TrtCacheDir` (string) / `USI_Ponder` (check) / `NetworkDelay` (spin) /
  `DrawValueBlack` `DrawValueWhite` (spin) / `ResignValue` (spin) /
  `MaxMovesToDraw` (spin) / `OpeningScript` (string) / dfpn・leaf-mate 系
- モデル未指定で `isready` が来たら **mock 評価器 + `info string
  mock evaluator (development only)` 明示** (analyze-gui で確立した
  user 承認済み慣例 2026-07-17 に合わせる)
- `info` 出力: `depth` (max_depth) / `nodes` (playouts) / `nps` /
  `score cp` (winrate から Ponanza 定数系の対数変換，proven は
  `score mate`) / **`pv` は必ず末尾**．出力は時間間隔 gate (デフォルト
  ~1 秒) で流量制御．MultiPV は非対応 (将来拡張)
- wheel VETO 整合: maou_usi は pure Rust + 既存 onnx feature 経由．新規
  HW 依存なし・単一 wheel 不変・CPU-only 動作維持

### 8. テスト計画

- protocol: parse/serialize 単体 + 方言 golden (startpos に moves なし /
  setoption 再送 / go 引数省略 / ponderhit 時刻付き / 未知コマンド無視)
- agent: fake transport の台本テスト — 通常対局 1 局 / ponder 的中 /
  ponder 外れ / thinking 中 stop / gameover 省略 GUI / usinewgame なしの
  再対局 / quit 中の探索破棄
- TimeStrategy: 三態 (秒読み/フィッシャー/切れ負け) + 残り 1 秒級の
  boundary + 非対称持ち時間
- 入玉宣言チェッカー: 27 点法 5 条件の境界 golden (電竜戦第 23 条の条件
  列挙をそのまま fixture 化)
- E2E (Python): subprocess で `maou-usi` を起動し USI セッション台本を
  流す (mock 評価器)．bestmove の合法性・stop 応答 100ms・quit の
  クリーン終了を assert
- 自己対局 smoke: mock で 1 局完走 + 各終局理由 (宣言/千日手/最大手数/
  投了) の再現
- 実 GUI 検証: user 環境の 将棋所/ShogiGUI/ShogiHome への登録と対局
  (Colab GPU wheel と同経路)．performance 数値は release ビルド +
  実測明示 (TRIPWIRE)

### 9. マイルストーンと版数計画

| M | 内容 | 完了条件 |
|---|---|---|
| M1 | maou_usi crate (protocol+agent+stdio) / maou_search stop token+無期限モード / `maou usi`+`maou-usi` / 簡易時間管理 (秒読み−margin) / mock+onnx | GUI に登録して対局が完走する |
| M2 | TimeStrategy 完全版 (三態+延長) / DrawValue / 入玉宣言 / resign / MaxMovesToDraw 最小 / info 随時出力 (スナップショット口) | 電竜戦系ルールでの実戦運用可 |
| M3 | ponder 一式 (go ponder/ponderhit/stop) / subtree 再利用の実装と効果計測 | ponder 的中時に木が引き継がれる |
| M4 | OpeningScript / 自己対局 driver (並列+評価共有，バッチ aggregator は計測後判断) | HWT 想定の強制手順対局 + 自己対局 smoke |

PR は M ごとに分割 (analyze-gui campaign の 2 PR 分割の慣行を踏襲)．

版数 (実装時): `rust/maou_usi` 0.1.0 新設 / `maou_search` 0.17.3 →
0.18.0 (stop token・無期限・snapshot・draw_value) / `maou_shogi` 5.8.2 →
5.9.0 (入玉宣言チェッカー) / `maou_rust` 0.22.0 → 0.23.0 (run_usi
expose) / Python `maou` 0.46.0 → 0.47.0 (M1)．

## 提案する docs 変更 (本 review の approve 対象)

1. **docs/design/usi-engine/index.md (新規)**: 本設計の常設化 (章立て:
   目的とスコープ / プロトコル調査と方言 / 大会特殊ルール / 既存資産と
   gap / レイヤー構成 (protocol・agent・stdio・selfplay) / Rust-Python
   境界 / maou_search 拡張 / 状態機械 / 戦略モジュール / 自己対局 /
   CLI と USI オプション / テスト / マイルストーンと未決事項)
2. **docs/design/position-search/index.md**: 未決事項の「subtree 再利用」
   「対局レイヤー」項に usi-engine 設計への相互参照を 1 行追記
3. **docs/commands/usi.md (新規)**: CLAUDE.md MUST に従い M1 実装 PR 内で
   既存フォーマット (Overview / CLI options / Example / GUI 登録手順)

## 未決事項 (user 確認ポイント)

1. エントリポイントの 2 本立て (`maou usi` サブコマンド + GUI 登録用の
   引数なし `maou-usi` console script) でよいか
2. `USI_Ponder` の default true (ponder は GUI 側で off にできる) で
   よいか．M3 まで実装されない間は option 宣言自体を出さない予定
3. 投了 (`ResignValue`) default off (= 投げない) でよいか
4. MaxMovesToDraw の in-search 対応 (探索木内でリミット以降を Draw 終端に
   する) は M4 で効果を見てから採否判断，でよいか
5. 定跡 (opening book) は本 campaign のスコープ外 (OpeningScript は大会の
   強制手順専用で定跡ではない) でよいか
6. 自己対局は「driver + 棋譜出力 + smoke」まで．学習データ生成 (HCPE)
   への接続は次 campaign，でよいか

## 承認記録 (2026-07-17)

user は本提案を**無修正で承認**した．未決事項 1-6 はすべて提案どおり:

1. エントリポイントは `maou usi` + 引数なし `maou-usi` の 2 本立て
2. `USI_Ponder` default true (M3 まで option 宣言自体を出さない)
3. 投了は default off (`ResignValue` 0 = 投了しない)
4. MaxMovesToDraw の in-search 対応は M4 で効果計測後に採否判断
5. 定跡 (opening book) は本 campaign のスコープ外
6. 自己対局は driver + 棋譜出力 + smoke まで．HCPE 生成接続は次 campaign

設計の正は docs/design/usi-engine/index.md (適用済み)．

## 根拠

- **VETO 整合**: 時間管理は TimeStrategy = 探索の上位レイヤーに新設
  (「持ち時間の消費計画は別レイヤー」そのまま)．wheel 可搬性・単一 wheel・
  CPU-only 回帰条件に影響なし．main 直コミットなし (feat/usi-engine)
- **user 方針との一致**: Rust ほぼ完結 + Python 薄ラッパー．stop/
  ponderhit 即応とオーバーヘッド排除を機構 (Rust reader スレッド + GIL
  解放済み探索) で保証
- **プロトコル/エージェント分離**は自己対局 (in-process 直結) と将来の
  CSA 対応の両方を 1 つのエージェント実装で賄うための最小の抽象
- 既存資産 (千日手規約・詰み併走・永続評価器・バッチ推論) を全て流用し，
  gap 7 点に絞って新規実装する — 探索本体 (PUCT/dfpn) の意味論には
  M1-M2 では触れない

## 出典

- [将棋所 USI プロトコル (原典，本調査時は接続不可のため二次資料で補完)](http://shogidokoro.starfree.jp/usi.html)
- [USI (Universal Shogi Interface) の現状調査 — 方言・互換性](https://qiita.com/sunfish-shogi/items/3efcd3a727c04ada020d)
- [USI 2.0 仕様案 (MyShogi/docs/USI2.0.md)](https://github.com/yaneurao/MyShogi/blob/master/MyShogi/docs/USI2.0.md)
- [USIプロトコルとは (Zenn)](https://zenn.dev/tarinaihitori/articles/101c268d7b7d36)
- [やねうら王 USI 拡張コマンド (ponderhit 時刻等)](https://github.com/yaneurao/YaneuraOu/wiki/USI%E6%8B%A1%E5%BC%B5%E3%82%B3%E3%83%9E%E3%83%B3%E3%83%89)
- [第4回世界将棋AI電竜戦本戦 大会ルール (PDF)](https://denryu-sen.jp/dr4/dr4_rule.pdf)
- [電竜戦 HWT の玉の屈伸運動 (理事長解説)](https://x.com/katsudonshogi/status/1736749670708498680)
- [電竜戦ハードウェア統一戦](https://denryu-sen.jp/hd2/)
- [ONNX Policy プレイヤーの .cmd 登録例 (将棋所/ShogiGUI)](https://gist.github.com/mizar/3335181fb6a88d40fcbe3d1b463e9d8d)
