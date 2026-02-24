# Stage 3 ハング調査報告

## 1. 問題の概要

- Stage 3（preprocessing データ，~40M レコード）の学習が "Training 0%" でハングする
- Stage 1，Stage 2 は正常に完了する
- ストリーミングモード（StreamingKifDataset + DataPrefetcher）で発生

## 2. 実行環境

- Google Colab A100 ハイメモリ（167GB RAM）
- PyTorch（spawn multiprocessing context）
- persistent_workers=True
- DataPrefetcher（バックグラウンドスレッド + CUDA stream）

## 3. 確定した事実

- 全7ワーカーが約30秒以内にバッチ生成を完了
- "Worker X: first batch produced" ログが全ワーカーで出力される
- Training 進捗は 0% のまま進まない
- セマフォリーク警告が29件，一貫して発生
- ブロック箇所はワーカー→メインスレッド間のパイプライン内

## 4. 調査で排除した仮説

| # | 仮説 | 対策 | 結果 |
|---|------|------|------|
| 1 | 合計ワーカー数超過 | `_cap_total_workers` で合計8に制限 (`3fe3869`) | ハング継続 |
| 2 | データ読込の遅延（ハングではない） | 診断ログをINFOに昇格 (`5b35f9c`) | ワーカーは30秒で完了，遅延ではない |
| 3 | pin_memory_thread デッドロック | streaming DataLoader で pin_memory=False (`cfdab0e`) | ハング継続 |
| 4 | DataPrefetcher が pin_memory を True に上書き | pin_memory_override のデフォルトを None に変更 (`b216e28`) | ハング継続 |
| 5 | torch.compile + CUDA stream 競合 | 分析で排除 | lazy compilation のため Training 0% 時点では未実行 |
| 6 | バックグラウンドスレッドでの CUDA 操作 | GPU 転送をメインスレッドに移動 (`905ba83`) | ハング継続 |

## 5. 実施した変更（コミット一覧）

| # | SHA | メッセージ | 変更内容 |
|---|-----|----------|---------|
| 1 | `3fe3869` | `fix(learning): cap total spawn workers to prevent Stage 3 hang` | training + validation の合計ワーカー数を8に制限 |
| 2 | `5b35f9c` | `fix(learning): improve Stage 3 diagnostics with INFO-level progress logs` | "first batch produced" ログを INFO に昇格，ファイル読込ログ改善，FIRST_BATCH_TIMEOUT 300→180s |
| 3 | `cfdab0e` | `fix(learning): disable DataLoader pin_memory for streaming mode to prevent deadlock` | streaming DataLoader の pin_memory を False に固定 |
| 4 | `b216e28` | `fix(learning): stop DataPrefetcher from overriding DataLoader pin_memory setting` | pin_memory_override のデフォルトを True→None，TrainingLoop の呼び出しも None に |
| 5 | `54ebd92` | `fix(learning): add DataPrefetcher diagnostics and env var bypass for Stage 3 debug` | _loader_thread 診断ログ + MAOU_DISABLE_GPU_PREFETCH バイパス |
| 6 | `6480630` | `Revert "fix(learning): add DataPrefetcher diagnostics..."` | 54ebd92 の revert（TDD アプローチに切り替えたため） |
| 7 | `905ba83` | `fix(learning): move GPU transfer from loader thread to main thread to prevent deadlock` | GPU 転送を _loader_thread からメインスレッド(__iter__)に移動 |

## 6. 未調査の次のステップ候補

以下の切り分けテストが未実施:

1. **`--dataloader-workers 0` テスト**: spawn ワーカーを完全に無効化し，メインプロセスでデータを読み込む．これで動作すれば spawn ワーカー + DataPrefetcher の組み合わせが原因
2. **DataPrefetcher 完全バイパス**: `MAOU_DISABLE_GPU_PREFETCH=1` 環境変数で DataPrefetcher をスキップ（要再実装: コミット 54ebd92 は revert 済み）．DataLoader を直接イテレーションして動作すれば DataPrefetcher が原因
3. **`persistent_workers=False` テスト**: persistent_workers を無効にして，エポックごとにワーカーを再生成．ワーカー管理の問題を切り分け
4. **`_loader_thread` 内部の詳細ログ**: `for batch in self.loader:` の前後にログを追加し，DataLoader イテレーション自体がブロックしているか確認
5. **DataLoader の `timeout` パラメータ変更**: 現在 `timeout=0`（無制限待機）を適切な値に変更し，DataLoader 内部のタイムアウト検出を有効化

## 7. 現在のアーキテクチャ

データフロー図（修正後の状態）:

```
spawn workers (7 processes)
  ↓ multiprocessing.Queue (worker_result_queue)
DataLoader.__next__() ← _loader_thread (バックグラウンドスレッド)
  ↓ threading.Queue (prefetch queue, CPU バッチ)
__iter__() ← メインスレッド
  ↓ GPU 転送 (CUDA stream)
Training loop (forward + backward)
```

ブロック箇所は `_loader_thread` の `for batch in self.loader:`
（DataLoader.__next__() の呼び出し）が最有力．
