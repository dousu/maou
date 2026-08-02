---
title: 合法手マスキングの doc 訂正とデッド CLI オプションの削除
date: 2026-08-03
status: applied
applied_in: PENDING_SHA
target:
  - docs/loss-functions.md
  - docs/commands/learn_model.md
risk: low
reversibility: easy
---

# 提案: 合法手マスキングの実態を doc に反映し，未配線の CLI オプションを削除する

## 背景

合法手情報が学習プロセスで使われているかを調査した結果，**Stage 3 では事実上
どこでも使われていない**ことが判明した．

- Stage 3 に供給される `legal_move_mask` は `torch.ones_like` のダミー
  (`dataset.py` / `streaming_dataset.py`)．前処理出力スキーマに合法手の列が無い
- 全要素 1 のため `masked_fill(~all_True, -inf)` は恒等変換になり，
  **`legal_move_mask=None` を渡した場合と勾配まで一致する**
- `callbacks.py` の policy 系メトリクスはマスクを一切参照せず raw 1496 次元で計算
- Stage 2 の legal-moves head は保存されるだけで Stage 3 は backbone しか読まない
- ONNX 出力は `["policy", "value"]` の 2 つのみ

一方 **推論時は完全にマスクされている**．`rust/maou_search/src/onnx.rs` が
`generate_legal_moves` の結果に対応する logits だけを gather して softmax を取るため，
非合法手が指されることはない．

## 実測 (発火量プローブ)

compass の TRIPWIRE「GPU 時間を使うレバー A/B の前に → 実装を直接叩いて発火量を
予測」に従い，学習済みモデルが非合法手へ漏らしている確率質量を測定した．
これはマスクを実効化したときに softmax の分母が変わる量の上限にあたる．

対象: floodgate golden 棋譜 342 局面 / `model_20260725_044443_vit-19.8m` / CPU EP

| 指標 | 値 |
|---|---|
| 非合法手への確率質量 (平均) | 0.0158 |
| 同 p50 / p90 / p99 / max | 0.0063 / 0.0385 / 0.1525 / 0.3293 |
| argmax が非合法手 | 0 / 342 局面 |
| top-5 に非合法手が混入 | 79 / 1710 枠 (4.6%) |

漏れの平均が 1.6% であり，マスクを実効化しても合法手の確率は平均 1.016 倍に
スケールするだけ．損失値の変化は 1% 前後で，n=40 の A/B が検出できる ~130 Elo
には遠く及ばない (TRIPWIRE の言う「発火したが検出限界未満」に着地することが
実装前に判明した)．

**結論**: 前処理出力への合法手列追加とメトリクスの合法手化は**見送る**
(user 判断, 2026-08-03)．doc の記述だけを実態に合わせる．

## 変更内容

### 1. `docs/loss-functions.md` — Stage 3 の「合法手マスキング」節

現状の記述は「Policy損失の計算時，`legal_move_mask` を用いて…確率質量が
合法手(平均~20手)のみに分配される」「学習効率の向上: softmaxの有効次元が
1496→~20に縮小」と**実装済み機能として断定**していたが，一度も発動していない．

見出しを「機構はあるが現在は発動していない」に改め，以下を追記した:

- ダミーマスクにより `masked_fill` が恒等変換になること
- 推論側 (`onnx.rs`) が合法手限定 softmax を行うため実害が限定的であること
- 上記の実測値と，合法手列追加を見送った判断
- `policy_top5_accuracy` / `policy_f1` / `policy_top1_win_rate` が
  raw 1496 次元で計算されており非合法手が混入し得ること

「ターゲット形式」節の `normalize_policy_targets` の記述にも注記を追加．

### 2. `docs/commands/learn_model.md` — 削除したオプションの行を除去

`--resume-reachable-head-from` / `--resume-legal-moves-head-from` の 2 行を削除．

## 併せて実施したコード変更 (src/)

`--resume-reachable-head-from` / `--resume-legal-moves-head-from` を削除した．

これらは **click が受け取り `interface/learn.py` まで転送されるが，そこから先で
一度も参照されない**デッドオプションだった．Stage 3 の `Network` は policy/value の
2 head しか持たず (`network.py`)，対応する head が存在しないため配線先が無い．
`ModelIO.split_state_dict` も legal-moves head の state dict を
アンダースコア変数で明示的に破棄している．

指定してもエラーにならず黙って無視されるため，「再開したつもりで再開していない」
事故を招く．compass VETO「些末な warning・lint・pre-existing バグは defer せず
即修正」に該当すると判断した．

- `src/maou/infra/console/learn_model.py`: `@click.option` 2 件と関数引数・転送を削除
- `src/maou/interface/learn.py`: 引数と docstring を削除
- `tests/maou/infra/console/test_cli_option_compatibility.py`: 除外リストから 2 件削除

将来 Stage 3 に補助 head を足す場合は，head の実装と同時にオプションを
入れ直すのが筋 (現状は配線先の無いオプションだけが残っている状態だった)．

## リスク

- **低**: 削除した 2 オプションは指定しても何も起きなかったため，動作の変化は
  「指定するとエラーになる」ことのみ．CLI 表面の破壊的変更にあたるため
  minor bump とした (`maou 0.74.1` → `0.75.0`．0.x では破壊的変更を minor で
  扱う既存慣行に従う — 例 `68c4c68` `feat(search)!:` が 0.71.0 → 0.72.0)．
- doc 変更は記述のみで挙動に影響しない．
- 逆行は容易．
