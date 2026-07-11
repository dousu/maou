# 1局面探索エンジン設計ドキュメント (草案)

> **状態: 草案 (living document)**．実装の進行に合わせて本ドキュメントを更新する．
> 各節に「実装済み」「設計方針 (未実装)」「未決」のいずれかを明記する．
> 実装済み記述の正は常にコード (`rust/maou_search/`) であり，乖離を見つけたら
> 本ドキュメント側を直す．

## 1. 目的とスコープ

与えられた局面 (SFEN) に対して探索を行い，**最有力手と評価値 (手番側勝率) を返す**
1局面探索エンジン．今後の将棋 AI (対局機能等) の基盤となるためパフォーマンス重視で
設計する．

- **最終目標 (North-star)**: GPU 環境 (Colab) での実測 100万 NPS
  (モデルパラメータ数に依存．ベンチを整備し実測ベースで較正する)．
- **スコープ外**: 持ち時間の消費計画．本機能は「与えられた予算 (時間/ノード数) 内で
  1 局面を探索する」までを責務とし，時間管理は上位レイヤーで設計する (binding，
  user 決定 2026-07-07)．

## 2. 全体構成

### 2.1 crate 配置 (実装済み)

```
maou (Python)
  └─ maou._rust (PyO3 cdylib = maou_rust)     ← 将来: 探索 API を公開
       └─ maou_search (pure Rust)              ← 本エンジン．MCTS + Evaluator trait
            └─ maou_shogi (pure Rust)          ← Board / 合法手生成 / dfpn
```

探索エンジンは **pure Rust の独立 crate `maou_search`** とする (PyO3 非依存，
依存は maou_shogi のみ)．maou_shogi 内 module 案は以下の理由で棄却済み:

1. 既存 dfpn/TT は明示的に単一スレッド設計 (`dfpn/tt/mod.rs` 冒頭) であり，
   マルチスレッド + バッチ推論前提の MCTS と設計前提が衝突する．
2. NN 評価を trait で抽象化することで，ルール crate (maou_shogi) と探索 crate を
   pure Rust に保ったまま推論実装を差し替えられる．
3. 再コンパイル影響の局所化と crate 独立バージョニング．

### 2.2 wheel 可搬性 (binding)

デフォルトビルドは可搬な pure Rust を維持する．ONNX Runtime / CUDA EP / TensorRT
などの platform 依存推論は **optional feature / 別 extra として分離**する
(user veto: native build 禁止，HW 命令は runtime gate のみ)．Python 側には既に
`cpu-infer` / `onnx-gpu-infer` / `tensorrt-infer` の extras 分離パターンがある．

### 2.3 モジュール構成 (実装済み)

| ファイル | 責務 |
|---|---|
| `src/evaluator.rs` | `Evaluator` trait (NN 推論の抽象境界) + `MockEvaluator` |
| `src/tree.rs` | 固定容量 `NodePool`，`Node`/`Edge`，lock-free 統計 |
| `src/search.rs` | PUCT 探索本体，バッチ収集，予算管理，結果集計 |
| `examples/nps_bench.rs` | NPS ベンチ CLI |

## 3. 探索アルゴリズム (実装済み)

基本は **PUCT ベースの MCTS** (AlphaZero/dlshogi 系)．将棋固有の工夫 (詰み探索統合，
勝敗確定伝播等) は §8 で段階的に追加する．

### 3.1 選択 (PUCT)

ノード n の子辺 e について:

```
score(e) = Q(e) + c_puct × P(e) × √N(n) / (1 + N(child(e)))
```

- `Q(e) = wins(child) / visits(child)`．**ノード統計は「親の手番側から見た」意味論**で
  保持する — Q がそのまま親手番の勝率になり，virtual loss が単純化する (§3.3)．
- 未訪問の子は `Q = fpu` (first play urgency，デフォルト 0.5 = 中立)．
- デフォルト: `c_puct = 1.5`．いずれもベンチ/対局で要チューニング (未決)．

### 3.2 評価と展開 (evaluate-then-expand)

葉に到達したら合法手を生成し，evaluator が返す priors を辺に設定して展開，
value を経路に沿ってバックプロパゲーションする (視点は 1 手ごとに反転:
`W += 1 - v; v = 1 - v`)．

### 3.3 virtual loss とバッチ収集

**GPU バッチ推論が最大のボトルネックになる想定**のため，バッチ収集を first-class で
設計する — これが本エンジンの中心的な設計判断である．

- 降下中，経路上の全ノードの visits を**前置インクリメント**する．wins が付かない
  in-flight 訪問は Q を押し下げ，virtual loss として働く (visits と別のカウンタを
  持たない — 親手番視点の統計だからこそ N+=1 だけで済む)．
- 各探索スレッドは葉を最大 `batch_size` 個収集してから
  `Evaluator::evaluate_batch` に一括で渡す (per-thread batching)．
- 将来: 複数スレッドの葉を単一 GPU バッチに束ねる global collector は，
  同じ `Evaluator` trait の背後に実装できる (未決 — per-thread で足りるか実測で判断)．

### 3.4 展開の同期 (マルチスレッド)

ノード状態は `UNEXPANDED → EXPANDING → EXPANDED | TERMINAL_LOSS` の一方向遷移．

- `UNEXPANDED → EXPANDING` の CAS に勝ったスレッドだけが評価・展開の所有権を持つ．
- 他スレッドが `EXPANDING` に到達したら**衝突 (collision)**: 前置 visits を
  ロールバックし，手持ちのバッチを即時 flush する．
- edges は `OnceLock` で一度だけ設定し，state の Release/Acquire で公開する．
- 子ノード生成の CAS 競合に負けた場合，確保済みプール slot は捨てる
  (leak，統計 `leaked_nodes` で監視．4T 実測で ~3.7% — 許容と判断済み)．

衝突率とバッチ充填率は「バッチ収集 vs 探索効率」トレードオフの主観測指標として
`SearchStats` に出す (4T/batch16 実測: 衝突 5.2%，充填 71%)．

### 3.5 木の表現とメモリ

- **固定容量 `NodePool`** を探索開始時に一括確保し，index の単調増加で lock-free に
  割り当てる．容量到達時は GC で低訪問サブツリーを刈り取って継続する (§7)．
- `Node`: visits (AtomicU64 — u32 は 1M NPS × 約 71 分で飽和するため) /
  wins (AtomicU64，**2^16 固定小数点** — [0,1] の勝率を `fetch_add` だけで加算，
  精度 1.5e-5) / state (AtomicU8) / edges (OnceLock)．
- 局面は木に保存しない．シミュレーションごとにルート Board (~640B POD) を clone し
  `do_move` で葉まで再生成する．transposition (DAG 化) は未対応 (未決 —
  `Board::hash_after` で do_move なしに子ハッシュが取れるため将来の TT 統合は可能)．

### 3.6 終端処理

- 合法手なし = 手番側の負け (`TERMINAL_LOSS`，v=0)．将棋にステイルメイトはない．
- 千日手 = 経路依存の終端 (`TERMINAL_DRAW` / `TERMINAL_WIN` / `TERMINAL_LOSS`，
  §9)．木に合流が無いため判定結果をノードに焼き付けられる．
- 終端の確定値は AND-OR 伝播 (§8.3) で祖先の内部ノードへ連鎖昇格し，
  確定ノードは以後降下せず固定値で短絡する．
- `max_ply` (デフォルト 512) 到達 = 引き分け v=0.5 で打ち切り
  (千日手にならないまま伸び続ける経路の無限降下ガード)．

## 4. Evaluator 境界 (実装済み — mock + ONNX)

```rust
pub struct EvalItem  { pub board: Board, pub moves: Vec<Move> }
pub struct EvalResult { pub priors: Vec<f32>, pub value: f32 }  // moves と同数・同順 / 手番側勝率 [0,1]
pub trait Evaluator: Send + Sync {
    fn evaluate_batch(&self, items: &[EvalItem]) -> Vec<EvalResult>;
}
```

探索側には「合法手ごとの事前確率 + 手番側勝率」だけを渡す．**モデル固有の変換は
すべて evaluator 実装側の責務**とする:

- 特徴量エンコード: 現行 NN 入力は board ID `(B,9,9)` int32 + hand `(B,14)` f32
  (Python 実装 `feature.py`)．**Rust 移植済み** (`maou_search/src/feature.rs`．
  maou_shogi の 104 プレーン `feature.rs` は legacy で現行 NN 未使用)．
- move→policy label: 1496 ラベル (`label.py` の `make_move_label`)．
  **Rust 移植済み** (`maou_search/src/label.rs`)．
- 出力変換: policy logits は合法手分を gather して softmax，value logit は
  sigmoid で勝率化 (評価値表示は `600 × logit` の Ponanza 換算)．

移植の正しさは Python 正実装から生成した golden fixture (全 (from, to, promo)
盤上手 + 全駒打ちの網羅ラベル表 13,689 ケース + 実局面 10 面) との一致で
担保する (`tests/parity.rs`．fixture 再生成手順は worklog 2026-07-08 参照)．

ONNX export は実装済み (`model_io.py`): 入力 `board`/`hand`，出力 `policy (B,1496)`
logits + `value (B,1)` logit，opset 20，batch 可変．**この契約が OnnxEvaluator の
I/F 仕様**である．

評価器の実装は 2 つ:

- `MockEvaluator` (デフォルトビルド): zobrist hash × splitmix64 の決定論的擬似乱数．
  NN 抜きで探索コアのスループット上限とバッチ挙動を計測するために使う．
- `OnnxEvaluator` (feature `onnx`): ONNX Runtime (ort crate) による実推論．
  CUDA EP は feature `onnx-cuda` + 実行時 opt-in の二段構え (§2.2 の可搬性 binding
  に従いデフォルトビルドは pure Rust のまま)．CUDA EP 初期化失敗は即エラーとし，
  静かな CPU フォールバックで GPU 計測を誤らせない．
  制約 (PoC): ort の `Session::run` が `&mut` 要求のため推論呼び出しを Mutex で
  直列化している．

## 5. 予算 API と停止 (実装済み)

```rust
pub struct SearchLimits { pub max_playouts: Option<u64>, pub time_ms: Option<u64> }
```

- 両方 `None` はデフォルト playout 上限 (2^20)．
- 停止理由は `StopCause` (PlayoutLimit / TimeLimit / PoolExhausted / RootTerminal)
  で結果に含める．
- playout 上限はバッチ粒度により最大 `threads × batch_size` 超過し得る (仕様)．
- 将来: depth 予算，詰将棋ソルバーと同様のノード数予算の意味論統一 (未決)．

## 6. 最終手選択 (実装済み — maou_search v0.15.0)

採用基準 (2026-07-11 確定．優先順):

1. **root-dfpn が詰みを証明** → 詰み手順の初手 (§8.1，最優先)．
2. **root の勝敗が AND-OR 伝播で確定** (勝ち/引き分け) → 確定値を達成する子 (§8.3)．
3. それ以外 → **robust child**: 負け確定 (ルート視点 proven=0) の手を除外して
   訪問回数最大 → 同数なら Q 最大 → 同率なら合法手生成順で先頭
   (`select_best_root_index`)．全手が負け確定なら除外なしで同基準 (どれも同値)．

乱択 (温度) は導入しない — 本機能は match play 相当の 1 局面探索であり，
自己対局学習の多様性付与 (AlphaZero の τ=1/30 手) や resign 判断は上位レイヤーの
責務とする (時間管理と同じ整理，§1)．

### 根拠 (2026-07-11 web 調査 + mock 検証)

- 主要エンジンの評価時 (match play) の基準は**訪問回数 argmax (robust child) が
  共通の土台**: AlphaZero (評価時 τ→0 = argmax visits)，dlshogi (proven 順序 →
  move_count 最大 → nnrate tiebreak)，lc0 (terminal rank → N → Q → P)．
- 生の max-Q (max child) は文献上最悪の基準とされ，mock でも低訪問子のノイズ Q を
  拾う挙動を機械的に再現した (41te 局面 5k playouts で訪問割合 10% の子を選択)．
- **負け確定の手の除外は dlshogi と同じ健全性規則**で，自己対局に依存せず論理的に
  正当化できる — 自己対局不可の現環境で確定できる根拠．「勝ち確定の子の優先」は
  AND-OR 即時伝播 (1 子でも負け確定なら root 勝ち確定，§8.3) により 2. に包含される．

### 自己対局導入後の再検討 (未決)

現環境では強さの比較検証ができないため，以下は自己対局フレームワーク導入後に
対局比較で再評価する:

- **LCB (secure child) 化**: leela-zero v0.17 が採用 (Student-t 分位 × 二項分散の
  下側信頼限界，実験値 3200 visits で 61.75% 勝率)．KataGo も採用 (lcbStdevs=5.0，
  minVisitPropForLCB=0.15 — 最多訪問の 15% 以上訪問した子に限り LCB 最大を採用)．
  一方 lc0 は不採用 (robust child + Q tiebreak) で強豪間でも結論が割れる．導入には
  per-child の value 分散統計 (二乗和) の追加とパラメータ (z，訪問ゲート) の対局
  チューニングが必要．
- **最短詰みの選好**: root-dfpn は find_shortest=false (§8.1) のため非最短の詰み
  手順を返し得る (41te で 45 手 line)．lc0 は moves-left で短い勝ち/長い負けを
  選好する．
- **温度乱択** (KataGo chosenMoveTemperature=0.10 相当): 対局の多様性が必要に
  なった時点で検討．

## 7. メモリ計画と GC (実装済み)

- 要件 (user): メモリは固定量とし，一定量に達したら GC する．**一回の削除で
  それなりの時間を空けて再度 GC が起きる**アルゴリズムを目指す (高頻度の
  小刻み GC は避ける)．必要性は実測済み: 4T/batch16 で 2M ノードが 2.6 秒で枯渇．
- 採用 (v0.2.0): **stop-the-world 訪問数閾値プルーニング + in-place compaction**
  (`NodePool::compact`)．
  - 同期は既存の PoolExhausted 停止機構を流用: 全スレッド join 後の quiescent
    状態でシングルスレッド実行し，`&mut` 要求で並行アクセスをコンパイル時に排除．
  - visits の単調性 (子 ≤ 親; 訪問は経路単位で加算・巻き戻し) により
    「visits ≥ T」集合は自動的にルート連結 — マーキング不要で，残存数が
    `gc_keep_ratio × 容量` 以下になる最小閾値 T をヒストグラムで選ぶ．
  - 解放された子への辺は未生成に戻り，再訪問時に再展開 (再評価) される．
    CAS 競合でリークしたノードも毎回回収される．
  - alloc は純粋 bump のまま (ホットパスコスト零)．
- 棄却した候補: 並行 GC (hazard pointer 相当が必要で race リスク)，free-list 方式
  (alloc ホットパスに競合と断片化)．
- `gc_keep_ratio` の既定は 0.5．mock 計測では 0.25 が優位 (worklog 2026-07-08)
  だが，実 NN では刈った枝の再評価コスト構造が変わるため実モデル接続後に
  再チューニングする (未決)．
- 対局レイヤー導入後は「ルート移動時の subtree 再利用 + 残りを解放」が主機構に
  なる可能性があり，単発探索の GC はその補助と位置づける．

## 8. 詰み探索統合 (8.1/8.2/8.3 実装済み，詰み探索はデフォルト有効)

詰み探索は **root-dfpn と leaf-mate をデフォルト有効** (SearchOptions::default())
とする．いずれも余剰 CPU (専用スレッド) で動き，詰みのない局面では NPS を
落とさない (Colab A100: NPS 95-99% 保持)．役割分担と PV-mate 棄却は §8.4．

### 8.1 ルート局面の詰み探索 (実装済み — v0.8.0，default-on)

- `SearchOptions::root_dfpn` (**既定 true**) で有効化．専用スレッドで
  `DfPnSolver` (find_shortest=false — 実戦用途は最初に見つかった詰みで十分)
  を root 局面に対して実行し，詰みが証明されたら root を勝ち確定 (§8.3)
  にして MCTS を停止，dfpn の詰み手順を best_move / PV / winrate=1.0 として
  返す．
- **NN 非依存で NN 盲点の詰みを補正する**のが root-dfpn の主眼．NN が誤って
  「負け」と評価する局面 (例: 41 手詰めがあるのに winrate 0.28) でも，df-pn は
  評価に関係なく root から詰みを探索するため補正できる (Colab A100 実測: 予算 2M
  で 5.3s / bestmove G\*7d / winrate 0.28→1.0)．
- 予算パラメータ: `root_dfpn_depth` (既定 2047 = 上限)，`root_dfpn_nodes`
  (**既定 2,000,000**)．両者 **CLI 露出** (`--root-dfpn-nodes` /
  `--root-dfpn-depth`)．時間予算は探索の `time_ms` に追随する．
  - **find_shortest=false ゆえ探索自体は予算に依存せず**最初の詰みを返す．
    予算を上げると所要時間が伸びるのは **TT 確保コスト** — TT は
    `(max_nodes*2).clamp(2^18, 2^23)` サイズで探索前にゼロ初期化 (memset) される
    ため予算に比例する (2M で ~256MB)．既定 2M は ~41 手級 (実測 ~1.2M ノード) を
    捕捉できる最小に近い値 (2^20 は僅かに届かない)．深追いのみ明示的に上げる．
- maou_shogi v5.5.0 で dfpn に**協調的停止フラグ** (`set_stop_flag`,
  `Arc<AtomicBool>`) を追加 — MCTS 側が先に終了したら dfpn を打ち切る
  (停止遅延 ≤ 1024 ノード，結果は timeout と同じ Unknown 扱い)．
- GC (プールの排他借用) と両立させるため dfpn スレッドは探索共有状態に
  触れず，Arc 経由の成果物 (証明フラグ + 詰み手順) だけで通信する．
- `CheckmateNoPv` (詰み証明済みだが手順復元失敗) は指し手を提示できない
  ため root-dfpn では採用しない．
- 注意: **`draw_ply` 引数は現行 dfpn API に存在しない** (初期構想の記述は誤り)．

### 8.2 葉ノードの短手詰み探索 (leaf-mate) (実装済み — 非同期，default-on)

dlshogi の葉ノード短手数詰み探索に相当．`SearchOptions::leaf_mate`
(**既定 true**) で有効化．**専用 mate スレッドによる非同期設計**で，探索
NPS を落とさない (Colab A100: 95-99% 保持)．

- **探索スレッドはブロックしない**: 新規展開する葉のうち王手手段を持つもの
  (`Board::does_have_mate_possibility` で前フィルタ，root は除外) を Arc 共有
  キューへ try-push するだけで solve せず，そのまま NN 評価へ回す．
- **専用 mate スレッド** (`leaf_mate_threads`，既定 1) が
  `DfPnSolver::new_leaf_mate` (小 TT・find_shortest=false・`leaf_mate_nodes`
  既定 50) で df-pn を回し，詰みを結果キューへ返す．
- **探索スレッド (worker) が結果を反映**: 当該葉を `try_mark_proven(WIN)` で
  勝ち確定にし AND-OR 伝播 (§8.3) する．`try_mark_proven` (CAS) ゆえ
  `finish_expansion` と競合しても proof を失わない．
- **健全性 (偽証明ゼロ)**:
  - mate スレッドは `NodePool` に触れず Arc 共有キューだけで通信する
    (GC の `compact` が `&mut NodePool` を取るため探索スレッドと同時に pool を
    借用できない; root-dfpn と同じ方針)．proof 適用は worker が inner scope 内で
    行い，`compact` (全 worker join 後に実行) と自然に排他される．
  - **GC 世代 (`generation`) ガード**: `compact` はノード index を再配置する
    ため，結果の世代が現世代と異なれば (compact 済み = index 無効) 破棄する
    (compact 後の無効 index への誤マーク = 偽証明を防ぐ)．
  - 勝ち確定にするのは df-pn の `Checkmate` / `CheckmateNoPv` のときのみ
    (root 並行 dfpn と同じ健全な証明パス)．
- **限界 (NN-coupled)**: leaf-mate は MCTS が詰み筋を降りることに依存する．
  NN 盲点 (例: 41 手詰めを winrate 0.28 と誤判定) では MCTS が詰み筋を避け
  leaf-mate が starve する (深さ 56 まで潜り leaf_mates 多数でも root 未達)．
  この種の詰みは root-dfpn (NN 非依存，§8.1) が担当する．
- 旧「dfpn API は solve 毎に TT を新規確保 (16MB) するため葉高頻度呼び出しは
  破綻」という制約は，`min_tt_entries` による小 TT (`new_leaf_mate`) + 専用
  スレッドの非同期化で解消した．統計は `SearchStats::leaf_mates`．

### 8.3 勝敗確定ノードの AND-OR 伝播 (実装済み — v0.7.0)

- いずれかの子で手番側の負けが確定 → 親は勝ち確定 (OR)．全子が確定済み
  なら親の確定値は `1 - min(子の確定値)` (AND 集約．勝ち/引き分け/負けの
  3 値 — 引き分けは千日手由来)．
- 実装: `Node` に確定状態 (`tree::proven`，手番側視点) を追加．詰み/千日手の
  終端マーク時に `propagate_proven` (search.rs) が経路上の祖先を連鎖昇格させる．
  確定ノードは以後降下せず確定値で短絡し，root が確定したら
  `StopCause::RootProven` で探索を早期停止する (best_move は確定値を達成
  する子を優先し，winrate は確定値になる)．確定ノード数は
  `SearchStats::proven_nodes`．
- 千日手由来の確定値を伝播できる根拠は経路の一意性 (§9 — 判定結果がノード
  不変のため終端と同格に扱える)．
- 詰み探索 (§8.1) の結果はこの機構に注入する (dfpn の詰み確定 → proven 化)．
- **確定値の衝突は先勝ち** (maou_search v0.15.2): 履歴非依存の詰み探索
  (root-dfpn / leaf-mate) と千日手 1 回近似 (§9) は，詰み筋が対局履歴との
  再出現を跨ぐ稀な局面で同一ノードに異なる確定値を出し得る．確定値は伝播後に
  覆せない (確定ノードは降下短絡で再伝播しない) ため `try_mark_proven` の
  CAS 先勝ちで確定し，後着の値は破棄する．
- マーク/伝播スキャンのメモリ順序は SeqCst (maou_search v0.15.1)．AcqRel
  では「自分を確定 → 兄弟を読む」の store-buffering により，最後の 2 兄弟を
  同時確定した 2 スレッドが互いを未確定と読んで AND 集約を両方中断し，親の
  確定が回復不能に失われ得る．
- 詰将棋ソルバーのノウハウ (証明駒/持ち駒優越，枝刈り) は，詰み確定を勝敗に
  変換できる場面で援用する (具体化は実装時)．
- 制限: 詰み手数 (mate distance) は保持しない — 最短詰みの選好は dfpn 統合時
  に PV から得る想定．

### 8.4 役割分担と PV-mate 棄却 (dfpn リソース戦略の結論)

Colab A100 実測 (学習済み ONNX, 29/41/63 手詰め局面) で確立した役割分担:

| 詰みの性質 | ツール |
|---|---|
| MCTS が降りる narrow mate | **leaf-mate** (小予算・全葉, §8.2) |
| root 直下 / NN 盲点の詰み (NN 非依存) | **root-dfpn** (予算 CLI 可, §8.1) |
| 受けが広い長手 tsume (63 手級) | **範囲外** — bounded mate search の実用外．NN の positional eval が答え |

- **PV-mate (PV 上の大予算 df-pn) は棄却 (REFUTED)**．実装 (a4ab7f3) して Colab で
  計測したが **leaf-mate に dominated**: 29 手 (受け narrow) を leaf-mate が 9.3s で
  割ったのに対し PV-mate 単独は 60s 未証明，全局面で leaf-mate/root-dfpn の割れない
  root を割った例はゼロ．さらに PV-mate は NPS を ~78% に落とす (大 TT ×スレッドの
  メモリ帯域 + 静かな PV 葉での大予算無駄撃ち)．**「深い詰みは `leaf_mate_nodes` を
  上げる (broad coverage) 方が PV-mate (deep single-line) より常に良い」**という
  構造的結論で撤去 (2738717)．
- **North-star の問い「MCTS がいくら探索しても詰みで評価が一変しないか」= YES**．
  41 手詰め (NN が winrate 0.28 と誤判定) を root-dfpn が 1.0 に補正した実証がその
  直接的な答え (§8.1)．

## 9. 千日手・連続王手の千日手 (実装済み — v0.6.0)

経路ハッシュ方式 (dfpn の `path_stack` 方式の援用) を採用．`Position`
(履歴付き Board) を降下に使う案は，clone コストに加え履歴管理が単一対局線
前提 (分岐探索非対応) のため棄却した．

- 降下中に経路上の各局面の `HistoryEntry { hash, in_check }` をスタックに
  積み，未展開の葉で「対局履歴 + 経路」を距離 4 から 2 手刻みで後方走査する
  (`maou_search/src/repetition.rs`)．ハッシュは maou_shogi 既存実装と同じ
  フル Zobrist (盤 + 持駒 + 手番，`Board::hash`)．
- 分類は `Position::is_perpetual_check_move` (position.rs) と同一の
  「王手フラグの手番 parity 別区間全称判定」を両側に一般化:
  千日手 = 引き分け 0.5，連続王手の千日手 = 王手をかけ続けた側の負け (0/1)．
- 実ルールの「同一局面 4 回」は待たず，経路上の最初の再出現で終端する
  (dfpn の on-path 検出と同じ探索内近似)．
- 木に合流 (transposition) が無く root への経路はノード毎に一意なため，
  判定結果をノードの終端状態に焼き付けられる — 再訪は走査なしで固定値を
  バックプロパゲーションする．
- root より前の対局履歴は `Searcher::search_with_history` で渡す
  (`HistoryEntry::from_board` で構築)．検出数は `SearchStats::repetitions`．
- コスト (DevContainer mock 相対): 降下 1 手あたりの `is_in_check` +
  葉での後方走査で 1T/2T は -4〜5%，4T は差なし (SMT の遊びに吸収)．
  GPU 律速の実 NN では不可視の見込み (未実測)．
- 未実装 (将来のレバー): 優越局面による一般化 — 盤面同一で持駒優越/劣位の
  刈り込み (dfpn の `DomPathStack` / `hand_gte` 相当)．

## 10. ベンチと計測規律

### 10.1 ベンチ (実装済み)

mock 評価の `nps_bench` と ONNX 実推論の `onnx_bench` の 2 本．
ビルド・実行・Colab GPU 計測・トラブルシューティングの詳細手順は
**[benchmarking.md](benchmarking.md)** を参照．

出力: playouts/s，衝突率，バッチ充填率，最大深さ，千日手検出数，確定ノード数，
プール使用量，GC 回数/解放量，best move / PV / 上位子．

### 10.2 計測規律 (binding)

- **NPS 計測は必ず release ビルド** (maturin デフォルトは dev profile なので注意)．
- **North-star (100万 NPS) への計上は Colab (GPU) 実測のみ**．DevContainer での
  計測は相対比較専用と明記する．
- NPS の定義は現状 **playouts/s** (= 葉評価スループット)．実 NN 接続後に
  「NN 評価局面数/s」との使い分けを確定する (未決)．

### 10.3 ベースライン (DevContainer 4C，mock 評価，相対値，2026-07-08 千日手検出込み)

| 構成 | playouts/s | 衝突率 | バッチ充填 |
|---|---|---|---|
| 1T / batch 8 | ~369K | 0% | 100% |
| 2T / batch 8 | ~597K | 0.9% | 97.2% |
| 4T / batch 16 | ~674K | 5.1% | 71.4% |

(同一セッション内の相対値．旧値 397K/624K/775K とは計測時の背景負荷が
異なるため直接比較しない — 千日手検出の増分は同時計測で 1T/2T -4〜5%，
4T 差なし)

## 11. マイルストーンと未決事項

### 実装状況

| 項目 | 状態 |
|---|---|
| MCTS コア (PUCT + virtual loss + バッチ収集) | ✅ 実装済み (maou_search v0.1.0) |
| Evaluator trait + MockEvaluator + NPS ベンチ | ✅ 実装済み |
| ノードプール GC | ✅ 実装済み (v0.2.0，§7) |
| visits u64 化 | ✅ 実装済み (v0.3.0) |
| OnnxEvaluator (ort + 特徴量/ラベル Rust 移植) | ✅ 実装済み (v0.4.0，§4) |
| AND-OR 勝敗確定伝播 | ✅ 実装済み (v0.7.0，§8.3) |
| 千日手検出 | ✅ 実装済み (v0.6.0，§9) |
| dfpn 停止フラグ + ルート並行詰み探索 | ✅ 実装済み (maou_shogi v5.5.0 + maou_search v0.8.0，§8.1) |
| leaf-mate (非同期葉詰み) + root-dfpn 予算 CLI + 詰み探索 default-on | ✅ 実装済み (maou_shogi v5.6.0 + maou_search v0.14.0 + maou v0.31.0，§8.1/8.2/8.4)．PV-mate は実装後に dominated と実証し撤去 (§8.4) |
| 最終手選択 (robust child + 負け確定除外) | ✅ 実装済み (maou_search v0.15.0，§6) |
| ルート子の確定値 (proven) の PyO3/CLI 露出 (Candidates に `proven=win\|draw\|loss` 表示) | ✅ 実装済み (maou_rust v0.16.0 + maou v0.33.0，[docs/commands/search.md](../../commands/search.md)) |
| PyO3 API / CLI (`maou search`) | ✅ 実装済み (maou_rust v0.10.0 + maou v0.23.0．[docs/commands/search.md](../../commands/search.md)) |
| Colab GPU 実測 | 配線検証済み (2026-07-08，極小モデル)．実モデルでの North-star 計測は未実施 |
| モデル×探索の強さ検証フレームワーク + パラメータチューニング | 未実装 |

### 主要な未決事項

| # | 未決 | 決め方 |
|---|---|---|
| 1 | 最終手選択の LCB (secure child) 化・最短詰み選好・温度乱択の要否 | 自己対局フレームワーク導入後に対局比較 (§6) |
| 2 | global batch collector の要否 / ort session の Mutex 直列化解消 | 実モデルの GPU 実測 (fill % と NPS) 後 |
| 4 | c_puct / fpu / batch_size / gc_keep_ratio 等の既定値 | チューニングフレームワーク |
| 5 | NPS の定義 (playouts/s vs NN eval/s) | 実 NN 接続時に確定 |

- (旧 #2「千日手検出方式」は経路ハッシュ方式の採用で解決 — §9)
- (旧 #3「葉詰み探索の有無と予算」は leaf-mate 非同期実装 + default-on で解決
  — §8.2/8.4．PV-mate は dominated と実証し棄却)

## 12. 参考

- dlshogi (探索の工夫全般，初期構想の参照元)
- AlphaZero (PUCT / policy-value MCTS)
- lc0 / KataGo / leela-zero (最終手選択 — robust child / LCB の参照実装，§6)
- [詰将棋ソルバー設計](../tsume-solver/index.md) — dfpn 統合 (§8) の詳細仕様
- [maou-shogi 設計思想](../maou-shogi-concept.md)
