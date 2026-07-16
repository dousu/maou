---
title: analyze-game コマンド新設 (棋譜 1 局の自動解析) の設計
date: 2026-07-16
status: pending
applied_in:
target:
  - docs/design/game-analysis/index.md
  - docs/commands/analyze_game.md
risk: low
reversibility: moderate
---

# 提案: `maou analyze-game` — 棋譜 1 局の自動解析コマンド

## 背景

user スケッチ原文 (2026-07-15):

> CSA や KIF ファイルから 1 局の棋譜を全部渡して，1 手ずつ 1 局面探索を
> 行う形．自動解析．時間管理は単純なもので良い — 例: 1 手 (1 局面) に
> 何秒，全局面何秒だったら 1 手は等分．成熟してきたら難しそうな局面に
> 時間配分を偏らせられると良いかもしれない (後半を重くするとか)

方向確認 (user 回答, 2026-07-16):

- コマンド名: **analyze-game**
- モデル毎回ロード問題: **最初から Rust 永続 binding を追加** (段階を踏まない)
- 出力形式: **JSON + stdout サマリ**
- user 訂正: TensorRT は `--trt-cache-dir` 指定時に初回ビルドした
  エンジンがキャッシュされ，2 回目以降のプロセスでは warmup 時間は
  ほぼ 0 になる (「毎回数十秒」は初回のみの話)

binding 制約 (VETO, user 2026-07-07): 持ち時間の消費計画は別レイヤー —
1 局面探索は与えられた予算内まで．docs/design/position-search/index.md §1
も時間管理を「上位レイヤーで設計する」と明記しており，本コマンドが
その上位レイヤーを実装する．

## 調査で確定した前提 (2026-07-16)

- 探索予算はノード数と時間の両対応: Rust `SearchLimits { max_playouts:
  Option<u64>, time_ms: Option<u64> }` (rust/maou_rust/src/maou_search.rs:330)
- 現行 `maou._rust.maou_search.search` は呼び出しごとに
  `OnnxEvaluator::from_file` でモデルをロードする (maou_search.rs:362)．
  評価器を保持する Python ハンドルは存在しない．maou_search crate は
  `OnnxEvaluator` / `Searcher` / `SearchOptions` / `SearchLimits` を
  pub export 済み (rust/maou_search/src/lib.rs:49-55) で永続化の下地はある
- 棋譜パースは Python 露出済み: `maou._rust.maou_shogi.parse_csa_str`
  (複数局対応) / `parse_kif_str` (単一局, UTF-8 前提でデコードは呼び出し側
  責務) → `GameRecord{sfen, moves(cshogi 互換 32-bit), names, ratings,
  win, endgame, times, scores, comments}` (maou_shogi.rs:520-614)．
  現状 src/maou の Python コードからは未使用
- domain `Board` に `set_sfen` / `push_move` / `get_sfen` /
  `move_to_usi` があり，各手番局面を再構築できる (src/maou/domain/board/shogi.py)
- コマンド追加は console + interface + app の 3 層 + `LAZY_COMMANDS`
  1 行という確立パターン (fetch-floodgate と同型, src/maou/infra/console/app.py:240-332)
- フォーマット自動判定は現状どこにも無い (hcpe-convert は
  `--input-format` 必須)

## 設計

### CLI

```
maou analyze-game --input-path FILE
  [--input-format csa|kif]        # 省略時は拡張子から自動判定
                                  # (.csa → csa / .kif,.kifu → kif / 不明は明示要求エラー)
  [--model-path PATH]             # 省略時 mock 評価器 (テスト用途)
  [--time-ms N]                   # 1 局面あたりの時間 (排他, 全省略時 default 1000)
  [--total-time-ms T]             # 全体 T ms を解析局面数で等分 (排他)
  [--playouts N]                  # 1 局面あたりの playout 数 (排他)
  [--num-candidates K]            # JSON に記録する候補手数 (default 5)
  [--output PATH]                 # JSON 出力先．指定時は stdout にサマリ，
                                  # 省略時は stdout に JSON のみ (pipe 用途)
  [探索 passthrough: --threads --batch-size
   --root-dfpn/--no-root-dfpn --root-dfpn-nodes --root-dfpn-depth
   --leaf-mate/--no-leaf-mate --leaf-mate-nodes --leaf-mate-threads
   --cuda/--no-cuda --tensorrt/--no-tensorrt --trt-cache-dir]
```

予算 3 オプションは相互排他 (2 つ以上指定はエラー)．探索 passthrough の
名前・デフォルトは既存 `maou search` に揃える．

### レイヤー構成 (fetch-floodgate と同型)

| 層 | ファイル | 内容 |
|---|---|---|
| infra/console | `src/maou/infra/console/analyze_game.py` | click コマンド + `LAZY_COMMANDS` 登録 |
| interface | `src/maou/interface/analyzer.py` | 入力検証・オプション組立・JSON/サマリ整形 |
| app | `src/maou/app/analysis/game_analyzer.py` | `GameAnalyzer` usecase + `BudgetAllocator` |
| rust (PyO3) | `rust/maou_rust/src/maou_search.rs` | 永続エンジン `SearchEngine` (新規 pyclass) |

app 層が `maou._rust` を直接呼ぶのは既存 `app/search/run.py` と同じ扱い．

### Rust 永続エンジン binding (`SearchEngine`)

呼び出しごとのモデルロードを排除するため，評価器を保持する pyclass を
`maou._rust.maou_search` に追加する:

```python
class SearchEngine:
    def __init__(self, *, model_path=None, threads=1, batch_size=8,
                 use_cuda=False, use_tensorrt=False,
                 trt_engine_cache_dir=None): ...
    def search(self, sfen, *, moves=None, max_playouts=None, time_ms=None,
               root_dfpn=True, root_dfpn_nodes=2_000_000,
               root_dfpn_depth=2047, leaf_mate=True, leaf_mate_nodes=50,
               leaf_mate_threads=1) -> SearchResult
```

- `__init__` で評価器 (OnnxEvaluator または mock) を 1 回だけ構築して保持．
  `search` は局面ごとに `Searcher::new(&evaluator, options)` を生成する
  (現行 `run_search` と同じ経路．ツリー/TT は局面ごとに新規)
- 探索中は GIL を解放する (`py.allow_threads`)．戻り値は既存
  `SearchResult` をそのまま使う
- 予算は毎回引数で受け取る — 時間配分の判断は一切持たない (VETO 整合)
- 既存の関数 `search` は互換のため残す (内部を `SearchEngine` 経由に
  リファクタするかは実装時判断)
- 局面ごとの `search` 呼び出しが Python 側ループに残るので，進捗表示
  (tqdm) と中断が自然に効く

### 時間配分 (BudgetAllocator — 本コマンドが担う「上位レイヤー」)

```python
@dataclass(frozen=True)
class PositionBudget:
    max_playouts: int | None
    time_ms: int | None

class BudgetAllocator(abc):  # 戦略
    def allocate(self, n_positions: int) -> list[PositionBudget]
```

- `FixedTimeAllocator(time_ms)` — 全局面同一時間
- `EqualDivisionAllocator(total_time_ms)` — `max(total // n, 1)` ms を
  全局面に配分 (床関数．端数は捨てる = 全体上限を超えない側に倒す)
- `FixedPlayoutsAllocator(playouts)` — ノード予算派

将来の傾斜配分 (難局面・終盤重視) は Allocator の戦略追加のみで実現でき，
探索側 API は変わらない．

### 解析ループ (GameAnalyzer)

1. ファイルを bytes で読み，UTF-8 先行 → 失敗時 cp932 でデコード
   (compass invariant: cp932 はほぼ任意バイト列で成功するため UTF-8 先行が本質)
2. `parse_csa_str` / `parse_kif_str` で `GameRecord` を得る．
   CSA で複数局が返った場合はエラー (1 局のみ対応と明示．将来拡張)．
   `moves` が空の棋譜もエラー
3. 解析対象 = 各指し手の直前局面 P0..P(N-1) (最終手後の局面は解析しない)．
   Allocator で N 局面分の予算を確定
4. 各 ply i について `engine.search(sfen=初期SFEN, moves=USI prefix
   m1..mi, ...)` を呼ぶ (prefix を渡すことで千日手履歴が正しく効く)．
   並行して domain `Board` を `push_move` で進め，記録用の SFEN・手番を得る．
   `GameRecord.moves` の 32-bit 値は `push_move` にそのまま渡せ，
   `move_to_usi` で USI に変換できる
5. 不正手 (push/パース失敗) は ply を明示して fail-loud

### 出力

JSON (機械可読, per-position 全記録):

```json
{
  "input": {"path": "...", "format": "csa", "names": ["...", "..."],
             "ratings": [3000.0, 2800.0], "win": 1, "endgame": "%TORYO",
             "n_moves": 118},
  "engine": {"model_path": "...", "threads": 4, "...": "..."},
  "budget": {"mode": "equal_division", "total_time_ms": 60000,
              "per_position": {"time_ms": 508}},
  "positions": [
    {"ply": 1, "side_to_move": "b", "sfen": "...",
     "played_move": "7g7f", "best_move": "2g2f", "match": false,
     "winrate": 0.53, "eval_cp": 42,
     "played_move_winrate": 0.51, "winrate_loss": 0.02,
     "pv": ["2g2f", "8c8d"],
     "candidates": [{"usi": "2g2f", "visits": 420, "winrate": 0.53,
                      "prior": 0.31, "proven": null}],
     "mate_found": false,
     "playouts": 800, "elapsed_ms": 500, "stop": "time_limit",
     "record_time_s": 12, "record_score": 55, "record_comment": null}
  ],
  "summary": {
    "match_rate": {"black": 0.62, "white": 0.58},
    "mean_winrate_loss": {"black": 0.031, "white": 0.044},
    "worst_moves": [{"ply": 88, "side": "w", "played": "5b4b",
                      "best": "3a4b", "winrate_loss": 0.34}],
    "mates_found": [{"ply": 115, "side": "b"}]
  }
}
```

- `winrate` / `eval_cp` は手番視点 (eval は既存の Ponanza 係数 600 変換)
- `played_move_winrate` は root_children から実戦の手を引いた値．
  未訪問なら null，その場合 `winrate_loss` (= best − played, 同一局面内
  比較) も null
- `mate_found` は `stop == "root_proven"` (PV が詰み手順)
- `record_time_s` / `record_score` / `record_comment` は棋譜側の記録の
  echo (`times` は moves と長さ不一致があり得るため範囲外は null)
- stdout サマリ (`--output` 指定時): 対局者・結果，先手/後手の一致率と
  平均 winrate 損失，worst_moves 一覧，詰み発見，総所要時間
- 進捗は tqdm (stderr) で局面単位に表示

### テスト計画

- allocator 単体 (等分・端数・最小値 1ms 保証)
- デコード (UTF-8 / cp932 fallback / 不正トレイル)
- `GameAnalyzer` golden: 数手の小 CSA/KIF fixture + mock 評価器で
  JSON スキーマと ply 進行を検証
- CLI: CliRunner + mock 評価器 (`--output` 経由で JSON を読む —
  tqdm と stdout の連結問題を回避)
- `SearchEngine` 再利用: mock 評価器で同一エンジンから複数回 search し
  結果整合を確認 (Python 側テスト)
- CSA 複数局・空棋譜・不正手の fail-loud

### 版数 (実装時)

- maou 0.42.0 → 0.43.0 (feat: 新コマンド)
- maou_rust 0.21.1 → 0.22.0 (feat: SearchEngine binding)
- maou_search crate は既存 export のみで足りる見込み — 変更が生じた
  場合のみ bump

## 提案する docs 変更 (本 review の approve 対象)

1. **docs/design/game-analysis/index.md (新規)**: 本設計を設計
   ドキュメントとして常設 (章立て: 目的とスコープ / CLI / レイヤー構成 /
   SearchEngine binding / 時間配分戦略 / 出力スキーマ / テスト)．
   position-search 設計との責務境界 (時間管理 = 本機能，1 局面探索 =
   position-search) を明記
2. **docs/commands/analyze_game.md (新規)**: CLAUDE.md MUST
   (新 CLI コマンド追加時のドキュメント作成) に従い，実装 PR 内で
   既存フォーマット (Overview / CLI options / Example) で作成

## 根拠

- user スケッチ + 方向確認 3 点 (上記) に基づく
- VETO「持ち時間の消費計画は別レイヤー」と position-search 設計 §1 の
  スコープ外宣言に整合 — 時間配分は app 層 Allocator，探索は与えられた
  予算のみ
- モデル 1 回ロード + N 局面探索は Rust crate の既存 pub export で
  実装可能 (調査済み)．既存 `search` 関数は不変更で後方互換
