# USI エンジン (対局エージェント) 設計

> 提案・承認の経緯は reviews/2026-07-17-usi-engine-design.md (approved
> 2026-07-17，未決事項 1-6 すべて提案どおり)．
> 各節に「実装済み」「設計方針 (未実装)」「未決」のいずれかを明記する．
> 本ドキュメント起草時点では全節が設計方針 (マイルストーン M1-M4 で実装)．
> 2026-07-27: M1-M4 実装完了 (M1=#393/#394, M2=#395, M3=#397, M4=#401)．
> 2026-07-28: **未決事項はすべて決着** (2 を ShogiHome 実機で確認)．
> (現状は §12 の表，検証手順と既知の課題は [verification.md](verification.md))．

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
| `csa` | CSA サーバプロトコル transport (`csa::protocol` = 行 ⇔ 型付きメッセージ (pure) / `csa::client` = TCP セッション + 対局ループ + 連続対局の再接続) | protocol, agent, time |

プロトコル層とエージェントを分離する根拠:

- 自己対局 driver が **agent を stdio/プロセスなしで直接駆動**できる
  (性能要件の核．プロトコル文字列の parse/serialize すら通らない)
- 電竜戦本戦は CSA → **CSA transport を agent 無変更で追加できた**
  (2026-07-29 実装．`agent` の変更 0 行で floodgate 対局が動いた —
  この分離の設計上の狙いが実証された形)
- GUI 方言を protocol/stdio に隔離し agent を clean に保つ
- fake transport で状態機械・戦略を完全に単体テストできる

### 4.1 CSA transport (floodgate 対局)

コマンドは [`maou floodgate`](../../commands/floodgate.md)．接続先の既定は
floodgate (wdoor)．一次資料 (<http://wdoor.c.u-tokyo.ac.jp/shogi/> と
CSA プロトコル ver 1.2.1) から確定した仕様:

| 事項 | 内容 |
|---|---|
| 接続先 | `wdoor.c.u-tokyo.ac.jp:4081` |
| ログイン名 | 任意 (**事前登録不要**)．重複回避のためオリジナルな名前を |
| パスワード欄 | **`floodgate-300-10F,<trip>`** — ゲーム名を埋め込む規約 |
| 同一性 | 「ログイン名 + trip」．サーバ側の識別子は `<ログイン名>+md5(<trip>)` で公開棋譜の `'rating:` 行に現れる |
| 対局の組まれ方 | **毎時 0 分と 30 分** |
| 対局後 | **ログアウト状態に戻る** → 再接続する (連続対局 = 1 接続 1 対局) |
| 持ち時間 | 300 秒 + 10 秒加算 (Fischer)．**512 手で引き分け** |
| レーティング | 15 試合程度で計算される．**プロトコルは運ばない** (公開棋譜の `'black_rate:` / `'white_rate:` と wdoor のレーティングページにのみ存在) |

責務分界は USI 経路と同一に保つ:

- CSA は対局開始時に持ち時間規定を一括通知し，以後は指し手に消費時間
  `,T<n>` を付ける (USI の毎手 `go btime wtime` とは別形式)．この差は
  **transport が吸収**して残り時間を追跡し，USI と同じ `ClockParams` に写す．
  1 手の予算配分は §8.1 の TimeStrategy がそのまま行う
- **消費時間はサーバ計測が正**．クライアントの実測ではない (仕様 3.2.2 が
  「サーバが示した消費時間を時間計算に使用すればよい」と定める)
- 自己対局 driver の `clock_margin_ms` を CSA 経路にも適用する
  (**時間制の経路は USI / 自己対局 / CSA の 3 本**．片方を直したら他も見る)
- 局面 (`BEGIN Position`) は既存の CSA 棋譜パーサ
  (`maou_shogi::kifu::parse_csa_str`) に委譲し，CSA 局面表記の 2 つ目の
  実装を持たない．指し手の解決は合法手照合 (USI 経路の `to_usi()` 照合と
  同じ規約) で，非合法手・盤面ずれがその場で検出される
- **ponder は使わない** (CSA に `ponderhit` 相当の信号が無い)

keep-alive は仕様 3.4 の下限 (同一クライアントからの受信後 30 秒) を機構的に
守る — 違反はサーバが反則負けにできる．

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
  **手数** (`GameState::move_number`．実装済み)，(M2) 探索スナップショット
- 出力: `TimeBudget { soft_ms, hard_ms }` (soft = 通常打切り目標，hard =
  絶対上限，時間切れ安全マージン込み)
- default 実装は三態: 秒読み型 (残時間/想定残り手数 × w + byoyomi −
  margin)，フィッシャー型 (残時間/想定残り手数 × w + inc − margin)，
  切れ負け型 (安全バッファ厚め)．非対称持ち時間は my_time ベースで
  自然に扱える
- `NetworkDelay` (ms) を margin に算入 (伝送遅延は自分の消費時間)
- (M2) 時間延長: soft 到達時に root best が不安定 (上位 2 手の訪問数
  拮抗・直近 best 交代) なら hard まで延長．延長判断も TimeStrategy 側
- 定数 (想定残り手数・カーブのパラメータ) は自己対局で調整し worklog に
  記録する (未決．手順は [verification.md §4](verification.md))

#### 8.1.1 手数カーブ `w(手数)` — 変換期重み付け (既定 on)

`w` を `peak_ply` 頂点の山型の折れ線とし，**変換期** (優勢を勝ちに変える
局面帯) に配分を寄せる．**既定 on**．既定値は 頂点 ply 100 / 半幅 55 /
頂点 2.5 倍 / 序盤の底 0.3 倍 / 終盤の底 1.2 倍．

設計上の要点は 3 つ:

1. **乗数は裁量枠 (`残時間 / horizon` = バンクの取り崩し) にのみ掛け，
   毎手戻ってくる時間 (秒読み・フィッシャー加算) には掛けない**．
   秒読みでは終盤の重みを下げても秒読み分が必ず残り，フィッシャーでは
   終盤が加算ペースに収束したうえでバンクを寄せられる．三態の分岐を
   増やさずに両方が正しくなる．
2. **底は序盤側と終盤側で別に持つ**．自己対局 + 中立な再解析で
   **序盤 (ply 9-30) の平均 winrate loss は 0.0004** (相手 0.0013) と
   両者ほぼ完璧 = 序盤の探索時間はほぼ無価値だった．一方**変換期の時間は
   中盤より価値が高い** (中盤 +12% が loss を Δ0.0022 改善するのに対し，
   変換期 −10% は Δ0.0077 悪化させた)．だから序盤を原資に，終盤側の底は
   一律配分より**上** (1.2 倍) に置く．
3. **自己正規化**．配分の土台が常にその時点の残時間の分数なので，カーブは
   配分を歪めるだけで総消費を増やす方向には倒れない．

**山を中盤に置いた最初の形は失敗した** (ply 55 / 頂点 1.8 / 序盤 0.7 /
終盤 1.0 = 一律と同値)．50 局で 34%・−117 Elo．敗因は「終盤の底を 1.0 に
すれば終盤は守られる」という誤りで，**同じ乗数でも掛ける先のバンクが
中盤で減っていれば絶対量は減る**．実測でも ply 91-110 の 1 手あたりが
B の 0.90 倍まで痩せ，勝勢からの逆転負けがその帯に集中した．
山を変換期へ移して 20 局 65%・+108 Elo へ反転した (§4.4.1)．

**実効ピークは `peak_ply` より手前に来る**: 重みが上がる一方で掛ける先の
残り時間が減るので，積が先に頂点を迎える (GPU 実測で配分比の最大は
ply 61-90 帯)．パラメータを読むときはこのずれを織り込むこと．

**フィッシャーでは増加分がカーブを薄める**．floodgate 相当 (300 秒 +
10 秒) では，序盤の裁量枠 7.5 秒に対し増加分が 10 秒あり，1 手予算の
過半が乗数の掛からない部分になる．公称 0.3〜2.5 倍でも実効は
**0.75〜1.25 倍**程度にしかならない (実測)．

なお `horizon_moves` 固定に由来する**未消化**は別問題として残る．50 手
指すと `(39/40)^50 ≈ 0.28` が構造的に残る．`horizon` を 20 に縮める
(= 速く使う) 方向は過去の A/B で **−89 Elo** なので「速く使えば強い」では
ない ([verification.md §4.4](verification.md))．

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

### 9.1 棋譜出力 (`--kifu-dir`) — 実装済み

A/B は「どちらが勝ったか」しか返さない．**どこで差がついたか**を見るには
1 手ごとの記録が要る (手数カーブの A/B が −117 Elo で負けた際，時間切れは
0 件だったため「終盤の探索が薄い」のか「中盤の追加探索が無駄」なのかを
区別できなかったのが動機)．

- **形式は CSA**．KIF パーサは初期局面を `手合割：` でしか読めず BOD 非対応
  なので，`--sfen` で始めた対局が**往復できない**．CSA は `P1..P9` +
  持駒行で任意局面を表現でき，`analyze-game` / `analyze-gui` /
  `hcpe_convert` の 3 つともが既に読める
- **1 局 1 ファイル** (`analyze-game` が複数局 CSA を拒否する)
- **writer はパーサと同じ場所** (`maou_shogi::kifu::csa`)．`parse_csa_str`
  が独立実装として往復テストの相手になる
- **指し手 → CSA 表記は単一実装**．floodgate transport が持っていた
  `move_to_csa` を `maou_shogi` へ移し，transport は再輸出にした
  (パーサ・writer・transport で 3 つ目の表記実装を持たない)
- **終局行は勝敗が読み戻しても一致する対応で選ぶ**．パーサは `%TORYO` /
  `%TIME_UP` / `%ILLEGAL_MOVE` を「手番側の負け」，`%KACHI` を「手番側の
  勝ち」と読み，driver の winner はいずれも「手番側が負ける」形で決まる
- **計測は棋譜と JSONL で分ける**．CSA には `T<n>` (秒) と `'** <score>`，
  正確なミリ秒と手ごとの playout は JSONL (`move_times_ms` /
  `move_playouts` / `move_scores`)．棋譜を標準形式に保ったまま分解能を
  失わないための分担．**`parallel > 1` では壁時計が CPU 競合で歪む**ので
  時間配分の分析は `parallel = 1` に限る

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

| # | 未決 | 決め方 | 現状 (2026-07-26) |
|---|---|---|---|
| 1 | TimeStrategy の定数 (想定残り手数カーブ・margin 既定値) | 実装時に自己対局で調整，worklog 記録 | **決着 — 既定 `horizon_moves = 40` を据え置き**．Colab L4 / ViT 19.8M / 30s+0.5s で **40 vs 20 = +89 Elo (paired t=+1.75)**，**60 vs 40 = 40 側が +61 Elo (paired t=-2.10)** と両側から検証．終局時の残り持ち時間は horizon 20/40/60 で 1.6s / 5.5-6.5s / 10.1s と単調に増え，配分が実際に効いていることを確認済み．40 vs 50 級の細かい調整は n≈400 局を要し未検証 |
| 2 | keep-alive 空行の default | GUI 実機で挙動確認後 | **決着 (既定 on = 5000ms)** — **ShogiHome** の実機で `KeepAlive 200` を指定して `isready` 中に空行 2 行が流れ，GUI はそれを無害に無視して対局へ進み正常終了 (`close=0`) することを確認．発火 (空行 2 行) と無害性の両方を観測した上での反転．`isready` が速い環境では 1 行も出ないので無音が正常．他の GUI は未確認 — 壊れる GUI が見つかったらここに名前を残す |
| 3 | USI_Hash → NodeCapacity 換算係数 | NodePool のノード実サイズから実装時決定 | **決着** — 実測レイアウト由来の 808 B/node (Node 48B + 分岐 62 × Edge 12B + 諸経費)．旧 512B は 1.6 倍の過小評価 |
| 4 | MaxMovesToDraw の in-search 対応 | M4 で効果計測後 | **決着 (on)** — 発火は上限直前の探索深さ分のみで 60 局の勝敗は不変．採否の根拠は棋力ではなく「最大手数時の詰みも引き分け」という電竜戦ルールへの適合．既定 0 では bit-identical |
| 5 | バッチ aggregator (自己対局並列時) | M4 で計測後 | **決着 (現行構成では採用しない)** — GPU でも並列は `parallel 1/2/4/8 = 4.7k/5.6k/5.9k/5.9k playouts/秒` と **1.26 倍で頭打ち**．ただし単発の長い探索は 10.9k 出るため律速は GPU 飽和ではなく**バッチ充填**で，対局をまたぐ aggregator には約 2 倍の伸びしろがある → **次 campaign の課題として起票**．CPU では `Mutex<Session>` が上限で 64/65/65 と完全に平坦だった |
| 6 | subtree 再利用の採否 | M3 で効果計測後 | **決着 (on 継続)** — 探索手の 90% で reroot 成功，引き継ぎは playout の 18-20% (予算 64/256/800 で一定)．勝率 40 局は +44 Elo (CI が 0 を含む) で，予算較正の予測と整合．GPU 実挙動の確認: [verification.md §6](verification.md) |

GPU (Colab) / GUI 実機での検証手順は [verification.md](verification.md) に
分離した．A/B ハーネスは `maou selfplay --ab-mode`
([docs/commands/selfplay.md](../../commands/selfplay.md)) として配布 wheel に
入っており，Rust example `selfplay_ab` は同じ `maou_usi::ab` を呼ぶ．
