# 棋譜解析コマンド (analyze-game) 設計ドキュメント

> **状態: 設計確定・実装中 (living document)**．実装の進行に合わせて本ドキュメントを
> 更新する．各節に「実装済み」「設計方針 (未実装)」「未決」のいずれかを明記する．
> 実装済み記述の正は常にコード (`src/maou/app/analysis/` ほか) であり，乖離を
> 見つけたら本ドキュメント側を直す．
> 提案・承認の経緯: reviews/2026-07-16-analyze-game-command-design.md

## 1. 目的とスコープ (設計方針)

CSA / KIF ファイルの **1 局の棋譜を丸ごと受け取り，1 手ずつ 1 局面探索を行う
自動解析コマンド** `maou analyze-game`．各局面のエンジン最善手・勝率・PV と
実戦の手との比較 (一致率，勝率損失) を機械可読 JSON として出力する．

- **時間管理は本機能の責務**: 1 局面探索エンジン (docs/design/position-search/)
  は「与えられた予算内で 1 局面を探索する」までが責務であり (binding，user 決定
  2026-07-07)，持ち時間・全体時間の配分計画はその上位レイヤー = 本機能が担う．
- 時間配分はまず単純に (1 局面 N ms 固定 / 全体 T ms の等分)．成熟後に
  難局面・終盤への傾斜配分を戦略追加で導入できる構造にする (user スケッチ
  2026-07-15)．
- **スコープ外**: 複数局を含む CSA の一括解析 (当面はエラー，将来拡張)，
  最終手後の局面の解析，対局機能．

## 2. CLI (設計方針)

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

- 予算 3 オプション (`--time-ms` / `--total-time-ms` / `--playouts`) は相互排他．
  2 つ以上の指定はエラー．
- 探索 passthrough の名前・デフォルトは既存 `maou search` に揃える．
- **実行プロバイダの選択肢は `maou search` と同一に維持する** (user 指示
  2026-07-16): デフォルト = CPU，`--cuda` = CUDA EP，`--tensorrt` = TensorRT EP
  (+ `--trt-cache-dir`)．analyze-game 側で特定 EP を前提・固定しない．
  TensorRT はエンジンキャッシュ利用時，初回ビルド以降の warmup がほぼ 0 になる．

## 3. レイヤー構成 (設計方針)

fetch-floodgate と同型の Clean Architecture 構成:

| 層 | ファイル | 責務 |
|---|---|---|
| infra/console | `src/maou/infra/console/analyze_game.py` | click コマンド + `LAZY_COMMANDS` 登録 |
| interface | `src/maou/interface/analyzer.py` | 入力検証・オプション組立・JSON/サマリ整形 |
| app | `src/maou/app/analysis/game_analyzer.py` | `GameAnalyzer` usecase + `BudgetAllocator` |
| rust (PyO3) | `rust/maou_rust/src/maou_search.rs` | 永続エンジン `SearchEngine` (pyclass) |

- app 層が `maou._rust` を直接呼ぶのは既存 `app/search/run.py` と同じ扱い．
- 棋譜パースは既存露出の `maou._rust.maou_shogi.parse_csa_str` (複数局対応) /
  `parse_kif_str` (単一局) を用いる．`GameRecord.moves` (cshogi 互換 32-bit) は
  domain `Board.push_move` にそのまま渡せ，`move_to_usi` で USI に変換できる．

## 4. 永続エンジン binding `SearchEngine` (設計方針)

現行 `maou._rust.maou_search.search` は呼び出しごとに ONNX モデルをロードする
(1 局面 1 回)．N 局面の連続解析でこれを排除するため，評価器を保持する pyclass を
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

- `__init__` で評価器 (OnnxEvaluator または mock) を 1 回だけ構築して保持する．
  `search` は局面ごとに `Searcher::new(&evaluator, options)` を生成する
  (現行 `run_search` と同じ経路．ツリー / TT は局面ごとに新規)．
- 探索中は GIL を解放する (`py.allow_threads`)．戻り値は既存 `SearchResult`．
- 予算は毎回引数で受け取る — **時間配分の判断を一切持たない** (§1 の責務分離)．
- 実行プロバイダは `__init__` 引数で選択し，既存 `search` 関数と同じ cargo
  feature gate (`onnx` / `onnx-cuda` / `onnx-tensorrt`) に従う — デフォルト
  wheel の可搬性を維持 (VETO「wheel 可搬性維持」)．
- 既存の関数 `search` は互換のため残す．
- バッチ関数 (1 call で全局面) ではなくハンドル型にした理由: 時間配分・進捗表示
  (tqdm)・中断を Python 側ループに残せるため．

## 5. 時間配分戦略 `BudgetAllocator` (設計方針)

```python
@dataclass(frozen=True)
class PositionBudget:
    max_playouts: int | None
    time_ms: int | None

class BudgetAllocator(abc):  # 戦略
    def allocate(self, n_positions: int) -> list[PositionBudget]
```

- `FixedTimeAllocator(time_ms)` — 全局面同一時間．
- `EqualDivisionAllocator(total_time_ms)` — `max(total // n, 1)` ms を全局面に
  配分 (床関数．端数は捨てる = 全体上限を超えない側に倒す)．
- `FixedPlayoutsAllocator(playouts)` — ノード予算派．
- 将来の傾斜配分 (難局面・終盤重視) は Allocator の戦略追加のみで実現し，
  探索側 API は変えない．

## 6. 解析ループ (設計方針)

1. ファイルを bytes で読み，UTF-8 先行 → 失敗時 cp932 でデコード
   (cp932 はほぼ任意バイト列で成功するため UTF-8 先行判定が本質)．
2. `parse_csa_str` / `parse_kif_str` で `GameRecord` を得る．CSA で複数局が
   返った場合はエラー (1 局のみ対応と明示)．`moves` が空の棋譜もエラー．
3. 解析対象 = 各指し手の直前局面 P0..P(N-1)．Allocator で N 局面分の予算を確定．
4. 各 ply i について `engine.search(sfen=初期SFEN, moves=USI prefix m1..mi, ...)`
   を呼ぶ (prefix を渡すことで千日手履歴が正しく効く)．並行して domain `Board`
   を `push_move` で進め，記録用の SFEN・手番を得る．
5. 不正手 (push / パース失敗) は ply を明示して fail-loud．

## 7. 出力スキーマ (設計方針)

JSON (機械可読, per-position 全記録):

```json
{
  "input": {"path": "...", "format": "csa", "names": ["...", "..."],
             "ratings": [3000.0, 2800.0], "win": 1, "endgame": "%TORYO",
             "n_moves": 118},
  "engine": {"model_path": "...", "threads": 4},
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

- `winrate` / `eval_cp` は手番視点 (eval は既存の Ponanza 係数 600 変換)．
- `played_move_winrate` は root_children から実戦の手を引いた値．未訪問なら
  null，その場合 `winrate_loss` (= best − played, 同一局面内比較) も null．
- `mate_found` は `stop == "root_proven"` (PV が詰み手順)．
- `record_time_s` / `record_score` / `record_comment` は棋譜側記録の echo
  (`times` は moves と長さ不一致があり得るため範囲外は null)．
- stdout サマリ (`--output` 指定時): 対局者・結果，先手/後手の一致率と平均
  winrate 損失，worst_moves 一覧，詰み発見，総所要時間．
- 進捗は tqdm (stderr) で局面単位に表示．

## 8. テスト (設計方針)

- allocator 単体 (等分・端数・最小値 1ms 保証)
- デコード (UTF-8 / cp932 fallback / 不正トレイル)
- `GameAnalyzer` golden: 数手の小 CSA/KIF fixture + mock 評価器で JSON
  スキーマと ply 進行を検証
- CLI: CliRunner + mock 評価器 (`--output` 経由で JSON を読む — tqdm と
  stdout の連結問題を回避)
- `SearchEngine` 再利用: mock 評価器で同一エンジンから複数回 search し結果
  整合を確認 (Python 側テスト)
- CSA 複数局・空棋譜・不正手の fail-loud

## 9. 未決事項

- 傾斜配分戦略の具体形 (難易度指標・終盤重み) — 単純等分の運用実績を見て設計
- 複数局 CSA / ディレクトリ一括解析の要否
- 解析結果の可視化 (勝率グラフ等) との接続
