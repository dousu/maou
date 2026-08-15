---
status: applied
applied_in: PENDING_SHA
date: 2026-08-15
target: [docs/commands/learn_model.md]
risk: low
reversibility: trivial
---

# `--adaptive-batch-measurement-interval` の既定が `1` から `5` に変わる

## Trigger

`/audit-backlog` (2026-08-15)．backlog の **Deferred 7** 行
([2026-08-08 app/learning](../audits/2026-08-08-src-maou-app-learning.md))
の緩和策を実装した結果として生じる doc drift．

2026-08-15 の前 run で，ユーザが step 3d の設計判断として
**「`measurement_interval` の既定を上げる」**を選んだ
([記録](../audits/2026-08-15-backlog-writeable-contract-and-decisions.md))．
その run は「1 行で contained」という誤った scope 説明のもとで回答を得て
しまったため実装を見送っており，本 run が正しい scope で実装した．

## この提案が P2 の恒久承認でカバーされる理由

CLAUDE.md § "Standing approval — drift corrections only" の判定基準は
**訂正後の本文が現行コードから一意に決まるか**の 1 点である．

この表のセルは `learn_model.py` の `@click.option` の `default=` を
そのまま写したものであり，コードが `default=5` になった以上，
セルの値は `5` 以外にあり得ない．**新しい指針でも節の再構成でもない**
ので P2 の drift correction に当たる．

なお**既定値を上げるという決定そのもの**は P2 由来ではなく，
2026-08-15 にユーザが 3d で選んだ判断帯の回答である．P2 が覆うのは，
その決定を実装した後に doc の 1 セルを追随させる部分だけである．

## 検証結果 (HEAD `377e1e3` + 本 run の変更)

`measurement_interval` の既定は **4 箇所**にある．

| # | 場所 | 変更前 | 変更後 |
|---|---|---|---|
| (i) | `src/maou/app/learning/gradient_noise_scale.py:85` | `measurement_interval: int = 1` | `= 5` |
| (ii) | `src/maou/app/learning/adaptive_batch.py:68` | `measurement_interval: int = 1` | `= 5` |
| (iii) | `src/maou/infra/console/learn_model.py:217` | `default=1` | `default=5` |
| (iv) | `src/maou/infra/console/utility.py:688` | `default=1` | `default=5` |

**本番経路を決めているのは (iii)** である — `learn_model.py:984` が
`measurement_interval=adaptive_batch_measurement_interval` と常に明示的に
渡すため，(i)(ii) だけを変えても挙動は変わらない．(iv) は
`benchmark-training` の同名オプションで，`utility.py:1365` が同じく常に
明示的に渡す．

doc 側で**既定値を明示しているのは 1 箇所だけ**である．
`docs/commands/utility_benchmark_training.md:48` はこのオプションを
他の adaptive batch オプションとまとめた 1 行で扱っており既定値を
書いていないので，訂正の対象にならない (再確認済み)．

## 提案する変更

`docs/commands/learn_model.md:32`

**Before**

```markdown
| `--adaptive-batch-measurement-interval INT` | `1` | GNS 計測の optimizer step 間隔．計測中は勾配スナップショット分の追加メモリを使用するため，大規模モデル(100M+ params)では 5-10 を推奨． |
```

**After**

```markdown
| `--adaptive-batch-measurement-interval INT` | `5` | GNS 計測の optimizer step 間隔．計測 1 回につきパラメータテンソルごとに `.item()` による host-device 同期が走り，勾配スナップショット分の追加メモリも使用する．大規模モデル(100M+ params)では 5-10 を推奨． |
```

既定値のセルを `1` → `5` に直し，説明にコストのもう一方の軸
(host-device 同期) を書き足す．同期のコストは
`gradient_noise_scale.py:150`/`:189`/`:192`/`:247` の `.item()` から，
メモリのコストは `:151`/`:193` の `g.clone()` から，いずれもコードから
一意に読み取れる．

## 影響

`--adaptive-batch` を有効にし，かつ
`--adaptive-batch-measurement-interval` を明示していなかった実行は，
GNS の計測頻度が 1/5 になる．GNS 値は
`training_loop.py:1031-1038` で `gradient_accumulation_steps` の更新に
使われるため，**adaptive batch の挙動が変わる**．データの互換性には
影響しない．
