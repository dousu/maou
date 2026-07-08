---
status: applied
applied_in: d832507
date: 2026-07-08
target: [docs/design/position-search/benchmarking.md, docs/design/position-search/index.md]
risk: low
reversibility: trivial
---

# NPS ベンチマーク手順書の新設と index.md の実装状況追随

## Trigger
user 指示 (2026-07-08): 「ひとまず動いて以下の結果が得られたのでベンチマーク
手順をドキュメントに残しておいてください」— Colab GPU での onnx_bench 初回
動作確認 (極小モデル，T4 相当) の直後の明示指示のため approve 済み扱いで
起票する．手順の出典は同日の Colab 検証セッション (LD_LIBRARY_PATH /
cuDNN 問題の解決を含む) と worklog 予定の実測ログ．

## Proposed change
- `docs/design/position-search/benchmarking.md` を新規作成:
  nps_bench (mock) / onnx_bench (ONNX) の使い分け，極小テストモデル生成
  (`tests/make_tiny_onnx.py`)，ローカル (相対比較専用) と Colab GPU
  (North-star 計測) の手順セル列，dlopen 起因の `libonnxruntime_providers_shared.so`
  と cuDNN 系ロードエラーのトラブルシューティング，統計の読み方
  (fill % が GPU 効率の主指標) とパラメータ掃引指針，数値は worklog に置く
  記録規律．
- `docs/design/position-search/index.md` を実装に追随 (living document 運用):
  §3.5/§7 GC 実装済み化 (採用: stop-the-world 閾値プルーニング + compaction，
  棄却: 並行 GC / free-list)，§4 OnnxEvaluator 実装済み化 (golden parity 検証・
  Mutex 直列化の PoC 制約)，visits u64，§10.1 から benchmarking.md へリンク，
  実装状況表と未決事項表の更新 (GC/visits u64 の未決を解決済みとして削除)．

## Motivation
Colab 検証手順が会話にしか存在せず，再計測 (実モデルでの North-star 計測，
パラメータ掃引) のたびに再構築が必要だった．また index.md の実装状況が
v0.1.0 時点のまま stale で，「実装済み記述の正はコード」の運用に反していた．

## Alternatives considered
1. 手順を index.md §10 に追記 — セル列 + トラブルシューティングで分量が
   大きく，設計文書の可読性を損なうため別ページに分離 (index からリンク)．
2. 計測値も docs に記録 — 陳腐化が速いため worklog/compass に置く規律を
   明文化して不採用．

## What this enables
- 次回以降の Colab 計測 (実モデル North-star，batch/threads 掃引) が
  コピペで再現できる．
- ベンチ統計の解釈基準 (fill %・衝突率・gc_runs) が共有される．

## What this constrains
- ベンチ CLI のオプション変更時は benchmarking.md の同期義務が生じる．

## Rollback plan
benchmarking.md を削除し index.md の変更を revert するだけ (コード非依存)．
