# USI エンジン (対局エージェント) 設計

> 提案・承認の経緯は reviews/2026-07-17-usi-engine-design.md (approved
> 2026-07-17，未決事項 1-6 すべて提案どおり)．
> 各節に「実装済み」「設計方針 (未実装)」「未決」のいずれかを明記する．
> 本ドキュメント起草時点では全節が設計方針 (マイルストーン M1-M4 で実装)．

## 1. 目的とスコープ

- 目的: maou を USI (Universal Shogi Interface) エンジンとして動作させ，
  将棋所・ShogiGUI・ShogiHome 等から**対人間・対 USI エンジンの対局**を
  可能にする．同じ対局エージェントを **in-process の自己対局**にも使う
  (対局の土台 = 自己対局・他将棋 AI 対局・人間対局)
- USI エンジンは単なるプロトコル変換ではなく**対局エージェント**として
  設計する: 持ち時間戦略・千日手戦略・大会特殊ルール (強制序盤手順/
  最大手数/入玉宣言) を戦略モジュールとして持つ
- スコープ外: 定跡 (opening book)，MultiPV，学習データ (HCPE) 生成への
  接続 (次 campaign)，優等/劣等局面の千日手一般化 (position-search 未決の
  まま)

## 2. USI プロトコルと方言 (調査結果 2026-07-17)

標準入出力の行指向テキストプロトコル．時間は全てミリ秒．

| 方向 | コマンド | 意味 |
|---|---|---|
| GUI→E | `usi` | 初期化要求．`id name/author` + `option` 宣言列 + `usiok` を返す |
| GUI→E | `isready` | 対局準備要求．重い初期化 (モデルロード等) はここで行い `readyok` |
| GUI→E | `setoption name X value Y` | オプション設定 (再送あり得る) |
| GUI→E | `usinewgame` | 新規対局開始通知 |
| GUI→E | `position startpos [moves ...]` / `position sfen <sfen> [moves ...]` | 局面設定 |
| GUI→E | `go btime B wtime W [byoyomi T \| binc b winc w]` | 思考開始 |
| GUI→E | `go ponder ...` / `go infinite` / `go mate T` | 先読み / 無制限 (検討) / 詰み探索 |
| GUI→E | `stop` | 思考中断．即時に `bestmove` を返す |
| GUI→E | `ponderhit` | ponder 的中．先読みを本探索へ切替え継続 |
| GUI→E | `gameover win\|lose\|draw` | 対局終了通知 |
| GUI→E | `quit` | 終了 |
| E→GUI | `bestmove <move> [ponder <move>]` | 指し手．特殊値 `resign` / `win` (入玉宣言勝ち) |
| E→GUI | `info depth/nodes/nps/time/score cp\|mate/pv/string ...` | 思考情報 |

ponder シーケンス: `bestmove X ponder Y` → GUI が相手番中に
`position ... moves ... X Y` + `go ponder` → 的中なら `ponderhit`
(時間計測はここから)，外れなら `stop` (bestmove は読み捨て)．
やねうら王拡張では `ponderhit` に btime/wtime 等を付加できる (受理する)．

### エンジン側で防御的に扱う方言

- **`info` の `pv` は行末尾に置く** (末尾でないエンジンのパース障害が
  電竜戦で実際に発生)
- `gameover` を送らない GUI がある (ShogiGUI の連続対局)．
  `usinewgame`/矛盾する `position` でも状態が自己回復するように作る
- `stop` への `bestmove` は GUI 側で読み捨てられ得る．「即時中断
  シグナル」として扱い，どんな状態でも短時間で bestmove を返す
- 開始局面の `position startpos` は `moves` を含まない (将棋所実挙動)
- `stop` 後の `setoption` 再送 (ShogiGUI 検討機能) を受理する
- プロトコル出力は ASCII 安全に保つ (GUI のエンコーディングが UTF-8 /
  Shift_JIS 混在のため `info string` の日本語は避ける)
- GUI へのエンジン登録は実行ファイル指定 (将棋所は起動引数を渡せない)．
  引数なしエントリポイント + `setoption` 設定が標準

出典: [USI 現状調査](https://qiita.com/sunfish-shogi/items/3efcd3a727c04ada020d) /
[USI2.0 仕様案](https://github.com/yaneurao/MyShogi/blob/master/MyShogi/docs/USI2.0.md) /
[やねうら王 USI 拡張](https://github.com/yaneurao/YaneuraOu/wiki/USI%E6%8B%A1%E5%BC%B5%E3%82%B3%E3%83%9E%E3%83%B3%E3%83%89)

## 3. 大会特殊ルール (電竜戦，調査結果 2026-07-17)

[第4回電竜戦本戦ルール](https://denryu-sen.jp/dr4/dr4_rule.pdf) /
[HWT](https://denryu-sen.jp/hd2/) より:

- 最大手数 512 手で引き分け (最大手数時の詰みも引き分け，宣言勝ち成立は
  勝ち)．引き分けは先手 0.4 勝/後手 0.6 勝
- 千日手は指し直しなしの引き分け
- 入玉宣言 (27 点法，CSA `%KACHI` / USI `bestmove win`): 宣言側の手番/
  玉が敵陣三段目以内/大駒 5 点小駒 1 点で先手 28 点・後手 27 点以上
  (持ち駒 + 敵陣三段目以内の駒，玉除く)/敵陣三段目以内に玉を除き 10 枚
  以上/王手されていない/持ち時間が残っている
- 持ち時間は先手 5 分+2 秒加算/後手 10 分+2 秒加算の非対称フィッシャー
- ハードウェア統一戦は先手時間ハンデを「最初の 4 手 =
  ▲58玉→△52玉→▲59玉→△51玉」(玉の屈伸運動) で実現．**参加プログラム側で
  対応必須**
- 電竜戦本戦の接続は CSA サーバプロトコル ver 1.2.1 (USI エンジンは
  ブリッジ経由が通例)

→ 戦略パラメータへの一般化: (a) 強制序盤手順 (OpeningScript)，
(b) 最大手数 (MaxMovesToDraw)，(c) 入玉宣言，(d) 引き分け価値の非対称性
(DrawValueBlack/White)．

## 4. レイヤー構成 (設計方針)

新規 Rust クレート `rust/maou_usi/`:

| モジュール | 責務 | 依存 |
|---|---|---|
| `protocol` | USI 行 ⇔ 型付きコマンド (`GuiCommand`/`EngineCommand`) の parse/serialize のみ．IO・戦略なし | なし (pure) |
| `agent` | 対局エージェント = 状態機械 + 戦略モジュール + 探索セッション．transport 非依存 | maou_shogi, maou_search |
| `stdio` | stdin 読取りスレッド + stdout 書込み (行バッファ+flush) + dispatch | protocol, agent |
| `selfplay` | in-process 自己対局 driver (M4) | agent |

プロトコル層とエージェントを分離する根拠:

- 自己対局 driver が **agent を stdio/プロセスなしで直接駆動**できる
  (性能要件の核．プロトコル文字列の parse/serialize すら通らない)
- 電竜戦本戦は CSA → 将来 CSA transport を agent 無変更で追加できる
- GUI 方言を protocol/stdio に隔離し agent を clean に保つ
- fake transport で状態機械・戦略を完全に単体テストできる

## 5. Rust / Python 境界 (設計方針)

**プロトコル・エージェント・時間管理・探索制御は全て Rust．Python は
薄いラッパー** (user 方針 2026-07-17):

- `maou usi` (click) が config (モデルパス・探索初期値・EP フラグ) を
  組み立て `maou._rust.maou_usi.run_usi(config)` を呼ぶ．GIL を解放して
  Rust が stdin/stdout を専有，`quit` で戻る
- **stdout はプロトコル専用**．Python logging は起動時に stderr へ向ける
- GUI 登録用の引数なしエントリポイント **`maou-usi`** (console script)
  を追加 (`maou usi` のデフォルト構成起動)．設定は `setoption` で受ける

根拠: `stop`/`ponderhit` 即応を Rust reader スレッドで機構的に保証
(Python 経由は GIL/GC がテールレイテンシに入る)．毎手のループ・タイマー・
ponder スレッド制御が Rust に閉じ，探索以外のオーバーヘッドをほぼゼロに
する．Python は導入 UX (uv/wheel/CLI) と将来の学習パイプライン統合のみ．

## 6. maou_search への拡張 (設計方針)

1. **stop token 注入**: `SearchLimits` に外部 `Arc<AtomicBool>` を追加し
   worker ループの既存 `stopped()` 判定に OR 合流．`StopCause::External`
   (M1 実装済み)
2. **無期限探索モード** (`go ponder`/`go infinite` 用): `max_playouts =
   u64::MAX` + stop token で表現する (予算未指定時の既定丸め
   DEFAULT_MAX_PLAYOUTS を変えると既存呼び出しが無限探索になるため，
   既定は維持した — M1 実装済み)
3. **進捗スナップショット**: worker が一定 playout 間隔で root 統計
   (playouts/nps/max_depth/PV/winrate) を snapshot 領域へ発行．`info`
   随時出力と時間延長判断の入力 (M2)
4. **千日手評価値の可変化**: 終端 Draw の 0.5 固定を
   `SearchOptions::draw_value` に (エージェントが手番視点へ変換して渡す)
5. **subtree 再利用** (M3): root 前進時に旧木の該当 subtree を引き継ぐ．
   [position-search §7](../position-search/index.md) の未決レバーを対局
   レイヤー要件として実装し，効果計測してから採否確定

1〜4 は movegen/dfpn の意味論に触れない見込みだが search.rs に触れるため，
実装 PR では STRICT-VERIFY canonical (29te/39te) を RAN して照合する．

## 7. エージェント状態機械 (設計方針)

状態: `Booting → Idle → EngineReady → InGame { AwaitingGo, Thinking,
Pondering }`．

- `usi` → `id name maou <version>` + option 宣言列 + `usiok`
- `isready` → 評価器構築 + warmup (TRT ビルド含む) → `readyok`．2 回目
  以降は健全性確認のみで即答．readyok まで 5 秒ごとの keep-alive 空行を
  オプションで用意 (default off．将棋所実機確認後に default 判断)
- `position` → 「初期局面 SFEN + USI 経路 = 千日手履歴」規約
  (`build_board_and_history`) をそのまま使う
- `go` → SpecialRules 前処理 (宣言勝ち・強制手順) → TimeStrategy が予算
  決定 → 探索スレッドへ依頼 → `bestmove [+ ponder]`
- `stop` → stop token → 探索合流 → `bestmove` (目標: 受信から 100ms 以内)
- `gameover` → ponder 停止・対局状態破棄 (来ない GUI でも自己回復)
- `quit` → 探索停止 → クリーン終了

探索は専用スレッド，reader スレッドは常時コマンド受理 (即応の機構的保証)．

## 8. 戦略モジュール (設計方針，agent 内 trait)

### 8.1 時間管理 TimeStrategy

「持ち時間の消費計画は別レイヤー — 1 局面探索は与えられた予算内まで」
(user 決定 2026-07-07) に整合: TimeStrategy が clock → 予算変換の上位
レイヤー，探索は予算を消費するだけ．

- 入力: `ClockState { my_time, opp_time, byoyomi, my_inc, opp_inc }`，
  手数，(M2) 探索スナップショット
- 出力: `TimeBudget { soft_ms, hard_ms }` (soft = 通常打切り目標，hard =
  絶対上限，時間切れ安全マージン込み)
- default 実装は三態: 秒読み型 (残時間/想定残り手数 + byoyomi − margin)，
  フィッシャー型 (残時間/想定残り手数 + inc − margin)，切れ負け型
  (安全バッファ厚め)．非対称持ち時間は my_time ベースで自然に扱える
- `NetworkDelay` (ms) を margin に算入 (伝送遅延は自分の消費時間)
- (M2) 時間延長: soft 到達時に root best が不安定 (上位 2 手の訪問数
  拮抗・直近 best 交代) なら hard まで延長．延長判断も TimeStrategy 側
- 定数 (想定残り手数カーブ等) は実装時に自己対局で調整し worklog に記録
  (未決)

### 8.2 千日手戦略 RepetitionPolicy

- 検出 (フル Zobrist，SFEN+USI 経路規約) と連続王手の千日手 (かけた側
  負け) は探索実装済み．戦略の実体は**引き分け価値**:
  `DrawValueBlack`/`DrawValueWhite` (千分率，default 500)．エージェントが
  先後・手番から探索の draw_value へ変換して渡す
- 電竜戦の引き分け 0.4/0.6 勝は DrawValue 400/600 でそのまま表現できる

### 8.3 大会特殊ルール SpecialRules

- **OpeningScript** (string，例 `"5i5h 5a5b 5h5i 5b5a"`): 対局経路が
  script の prefix と一致する間は次の script 手を探索なしで即指し．外れ
  たら以後無効化．script 手が非合法なら無効化して通常探索 (安全側)
- **MaxMovesToDraw** (spin，default 0 = 無効): リミット到達局面では宣言
  可否を必ず確認し可能なら `bestmove win`．リミットが近ければ探索予算を
  絞る．in-search 対応 (リミット以降を Draw 終端) は M4 で効果計測後に
  採否判断 (user 承認 2026-07-17)
- **入玉宣言**: 27 点法チェッカーを `maou_shogi` に新規実装
  (`Board::nyugyoku_declarable()`，手番・敵陣三段目・点数 28/27・枚数
  10・王手なしの 5 条件．時間条件は agent 側)．`go` 受信時に判定し成立で
  `bestmove win`
- **Resign** (`ResignValue` 千分率，default 0 = 投了しない — user 承認):
  root winrate が閾値未満の状態が `ResignConsecutive` 手続いたら
  `bestmove resign`

### 8.4 ponder PonderPolicy

- `USI_Ponder` (check，default true — user 承認)．**M3 実装まで option
  宣言自体を出さない**
- 予想手 = 自探索 PV の 2 手目 (PV 長 < 2 なら ponder なし)
- `go ponder` = 無期限探索．`ponderhit` → その時点から予算計算して探索
  継続 (木・playout が活きる = ponder の主利得)．やねうら王拡張の時刻
  付き ponderhit も受理
- `stop` (外れ) → bestmove 返却 (読み捨て)．外れ後は M3 の subtree
  再利用が入るまで作り直し探索
- 「ponder を返さず勝手に先読み」方式は採らない (時間管理が GUI と乖離)

## 9. 自己対局 (設計方針，M4)

- `maou_usi::selfplay`: 1 対局 = agent 2 個 (先後) を stdio/プロセス
  なしで直接駆動．評価器 (ONNX session + TRT キャッシュ) はプロセス内
  1 個を全対局共有，モデルロード/warmup 1 回
- 並列度: T スレッド G 対局．現状の `Mutex<Session>` 直列化が上限になる
  ため，複数探索の評価要求をまとめる**バッチ aggregator は効果計測して
  から**検討 (先行投資しない)
- 終局判定 (宣言/千日手/最大手数/投了) は USI 対局と同一実装 = 意味論
  一致
- 成果物は driver + 棋譜出力 + smoke まで．HCPE 生成接続は次 campaign
  (user 承認 2026-07-17)

## 10. CLI と USI オプション (設計方針)

- `maou usi` (click) + `maou-usi` (引数なし console script)．CLI フラグは
  `maou search`/`analyze-game` と同名の探索・EP passthrough．
  **CLI フラグ = 初期値，`setoption` が上書き**
- USI オプション: `ModelPath` (filename) / `Threads` / `BatchSize` /
  `NodeCapacity` (spin．`USI_Hash` は受理し NodeCapacity 未指定時の換算に
  使う — 係数はノード実サイズから実装時に決定) / `UseCuda` /
  `UseTensorRT` / `TrtCacheDir` / `USI_Ponder` (M3) / `NetworkDelay` /
  `DrawValueBlack` `DrawValueWhite` / `ResignValue` / `MaxMovesToDraw` /
  `OpeningScript` / dfpn・leaf-mate 系
- モデル未指定で `isready` → mock 評価器 + `info string mock evaluator
  (development only)` 明示 (analyze-gui の user 承認済み慣例)
- `info`: `depth` (max_depth) / `nodes` (playouts) / `nps` / `score cp`
  (winrate から対数変換，proven は `score mate`) / **`pv` 末尾**．時間
  間隔 gate (~1 秒) で流量制御．MultiPV 非対応
- wheel 可搬性: maou_usi は pure Rust + 既存 onnx feature 経由．新規 HW
  依存なし・単一 wheel・CPU-only 動作維持

## 11. テスト (設計方針)

- protocol: parse/serialize 単体 + 方言 golden (startpos に moves なし /
  setoption 再送 / go 引数省略 / ponderhit 時刻付き / 未知コマンド無視)
- agent: fake transport 台本 — 通常対局 / ponder 的中 / ponder 外れ /
  thinking 中 stop / gameover 省略 GUI / usinewgame なし再対局 / quit
- TimeStrategy: 三態 + 残り 1 秒級 boundary + 非対称持ち時間
- 入玉宣言: 27 点法 5 条件の境界 golden (電竜戦第 23 条を fixture 化)
- E2E (Python): subprocess で `maou-usi` に USI 台本 (mock)．bestmove
  合法性・stop 応答 100ms・quit クリーン終了
- 自己対局 smoke: mock 1 局完走 + 各終局理由の再現
- 実 GUI 検証: user 環境の将棋所/ShogiGUI/ShogiHome．性能数値は release
  ビルド + 実測明示

## 12. マイルストーンと未決事項

| M | 内容 | 完了条件 |
|---|---|---|
| M1 | maou_usi crate (protocol+agent+stdio) / maou_search stop token+無期限モード / `maou usi`+`maou-usi` / 簡易時間管理 / mock+onnx | GUI に登録して対局が完走 |
| M2 | TimeStrategy 完全版 / DrawValue / 入玉宣言 / resign / MaxMovesToDraw 最小 / info 随時出力 | 電竜戦系ルールで実戦運用可 |
| M3 | ponder 一式 / subtree 再利用の実装と効果計測 | ponder 的中で木が引き継がれる |
| M4 | OpeningScript / 自己対局 driver (並列+評価共有) | 強制手順対局 + 自己対局 smoke |

PR は M ごとに分割．版数: `maou_usi` 0.1.0 新設 / `maou_search` minor /
`maou_shogi` minor (入玉宣言，M2) / `maou_rust` minor / Python `maou`
minor (M1 で 0.47.0)．

### 未決事項

| # | 未決 | 決め方 |
|---|---|---|
| 1 | TimeStrategy の定数 (想定残り手数カーブ・margin 既定値) | 実装時に自己対局で調整，worklog 記録 |
| 2 | keep-alive 空行の default | 将棋所実機で挙動確認後 |
| 3 | USI_Hash → NodeCapacity 換算係数 | NodePool のノード実サイズから実装時決定 |
| 4 | MaxMovesToDraw の in-search 対応 | M4 で効果計測後 |
| 5 | バッチ aggregator (自己対局並列時) | M4 で計測後 |
| 6 | subtree 再利用の採否 | M3 で効果計測後 |
