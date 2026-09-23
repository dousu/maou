---
status: pending          # pending | approved | applied | rejected
date: 2026-09-23
target: [docs/colab-cli-notes.md]
risk: low
reversibility: trivial
---

# §10.1 のチェックリストに「環境構築手順を踏む」と「失敗も終了として扱う」を足す

## Trigger

`b673675e` で §10.1「新しい long-running driver を書くときのチェックリスト」を
新設した直後，同じ driver (`scripts/colab_search_values_job.py`) が**環境構築で**
落ちた (L4 セッション，2026-09-23 12:48 JST，user 報告):

```
$ /usr/bin/python3 -m pip install --upgrade \
  .../maou-0.104.0-cp312-cp312-manylinux_2_28_x86_64.whl \
  .../maou-0.104.0-cp313-cp313-manylinux_2_28_x86_64.whl
  rc=1
[FAIL] wheel install — RuntimeError: pip install failed rc=1
```

正の手順は **§6** と **`docs/design/position-search/benchmarking.md` § "Colab (GPU)"**
に既にあったが，driver はそこから 4 箇所ずれていた (PR #530 `1b05ab31` で修正):

1. Release の全 `.whl` を 1 回の `pip install` に渡す (cp312 と cp313 が並ぶので
   非互換側でコマンドごと落ちる)
2. `releases/latest` を引く (正はタグ固定の `releases/tags/latest`)
3. extras が無い (`search-values` は ONNX GPU 推論なので `maou[tensorrt-infer]`)
4. `ldconfig` の手順が無い (**踏まないと `--tensorrt` / `--cuda` を渡しても
   EP が解決できない**)

**1 だけを直すと，セル B の点検は PASS したうえで投入後 20 時間を失う**種類の
欠陥だった (4 が残るため)．§10.1 は退避・診断・命名は列挙しているが
**環境構築には触れていない**．これが穴だった．

あわせて同 PR のレビューで，人間が見るモニタが `done`/`grace` だけを終了条件に
していて **`failed` で永久ループする**バグも見つけた．これも「driver を書くときに
満たすもの」の一般形なのでチェックリストに入れる．

## 変更 (§10.1 のリストへ 2 項目追加)

現行のリスト末尾:

```
- [ ] 投入前の点検手段を持つ (例: `colab_search_values_job.py --check`)．
      数十時間を投じる前に，入力・モデルの同一性・wheel の版・GPU を確かめる
```

この直後へ:

```
- [ ] **環境構築は §6 と `docs/design/position-search/benchmarking.md`
      § "Colab (GPU)" の手順をそのまま踏む**．自前で書き直さない:
      **タグ固定の `releases/tags/latest`** を引き，**`cp{major}{minor}` で
      wheel を 1 枚に絞り** (Release には cp312 と cp313 が並ぶ — 全部を 1 回の
      `pip install` に渡すと非互換側でコマンドごと落ちる)，用途に応じた extras
      (`learn-model` は `[cuda]`，ONNX GPU 推論は `[tensorrt-infer]`) を付け，
      GPU 推論なら `/etc/ld.so.conf.d/maou.conf` + `ldconfig` まで行って
      **`libonnxruntime_providers_shared` と `libnvinfer.so.10` の解決を確認する**．
      導入は**同期的に**行い rc を見てから次へ進む．
      **`ldconfig` を踏み忘れても wheel の導入自体は成功するので，点検は通って
      しまい，投入してから数十時間を失う**
- [ ] 人間が見るモニタを持つなら，**失敗も終了として扱う**．成功側の phase
      だけを終了条件にすると，死んだジョブを相手に待ち続ける．終了時は
      エラー本文と `diag/` の場所を出す
```

## 影響

規範の追加のみ．既存の記述は変えない．`colab_arm0_job.py` は §6 の手順を
踏んでいるので追随不要．
