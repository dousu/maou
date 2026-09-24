---
status: approved         # pending | approved | applied | rejected
applied_in:
date: 2026-09-24
target: [docs/colab-cli-notes.md]
risk: low
reversibility: trivial
---

# §10.1 のチェックリストに「wheel 導入の再試行と排他」と「1 セッションを予算内に収める」を足す

## Trigger

2026-09-24，L4 セッションでセル B (`colab_search_values_job.py --check`) の
1 回目が wheel の導入で落ちた (2 回目は成功)．VM 上の `job.log` から:

```
━━━━━━━━━━━━╸  14.7/46.4 MB 2.4 MB/s
pip._vendor.urllib3.exceptions.ReadTimeoutError:
  HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out.
rc=2
```

1. **pip 既定の受信待ち 15 秒**で転送途中に落ちた．`maou[tensorrt-infer]` は
   TensorRT / onnxruntime-gpu / CUDA ライブラリで GB 級あり，一瞬の停止で当たる．
   pip の `--retries` は接続段階にしか効かず，転送途中の停止では回復しない．
2. 同じログで **点検 (セル B) とワーカー (セル C) の pip が同時に走っていた**
   (11:09:49 と 11:10:52 に開始)．同じ `dist-packages` に 2 本が書き込み，
   ワーカー側の `ldconfig` が
   `libnvinfer_builder_resource_win_sm75.so.10.16.1 is truncated` を出した．
   今回は探索開始が両方の完了後で実害は無かったが，順序が違えば書きかけの
   `.so` を読む．
3. 1 セッション目は `--max-positions 900000` が 20 時間の予算に収まらず
   SIGTERM で畳まれた (実測 10.7 局面/秒，見積 45,800 局面/時は外挿)．
   畳まれると CLI の要約が出ず，flush 前の局面を捨て，`searched < 上限` による
   完了判定 (#532) も使えない．

driver 側は PR #533 で直した．§10.1 はこの 3 つに触れていないので，
3 本目の driver が同じ穴を踏まないよう一般形で足す．

## 変更 (§10.1 のリストへ 2 項目追加)

「環境構築は §6 と ... の手順をそのまま踏む」の項目の直後へ:

```
- [ ] **wheel の導入はネットワークの一時停止で落ちる前提で書く**．
      pip 既定の受信待ち 15 秒では GB 級の `[tensorrt-infer]` / `[cuda]` が
      転送途中で `ReadTimeoutError` になる (pip の `--retries` は転送途中の
      停止には効かない)．`--timeout 60 --retries 10` を付け，失敗したら
      `pip install` を丸ごと数回やり直す (落としきったパッケージはキャッシュに
      残るので，やり直すたびに残りだけを取りに行く)．**点検とワーカーの両方が
      導入するなら排他にする** (`fcntl.flock`)．2 本の pip が同じ
      `dist-packages` に書き込むと，片方の `ldconfig` が書きかけの `.so` を
      読む．`ldconfig` もロックの内側で行う
- [ ] 時間予算で畳む driver は，**1 セッションの仕事量を実測スループットから
      予算内に収まるよう決める**．予算切れは保険であって通常の終わり方に
      しない．SIGTERM で畳むと CLI の要約が出ず，flush 前の成果を捨て，
      要約に頼る完了判定も効かなくなる．外挿した見積で決めた値は初回の
      実測で見直す
```

## 影響

規範の追加のみ．既存の記述は変えない．`colab_search_values_job.py` は
PR #533 で両方を満たす．`colab_arm0_job.py` は wheel を自分で導入せず，
時間予算で畳む設計でもないので追随不要．
