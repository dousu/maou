# 損失関数設計ドキュメント

## 概要

Maouのマルチステージ学習では，各ステージの学習目標に応じて異なる損失関数を使用する．
ステージ間の依存関係は「Stage 1 → Stage 2 → Stage 3」の順序であり，
前段ステージでバックボーンに基礎的な盤面理解を獲得させた上で，
後段ステージで高度な意思決定を学習する設計である．

| Stage | 目的 | 出力形式 | 損失関数 | メトリクス |
|-------|------|---------|---------|-----------|
| 1 | 到達可能升の学習 | 81次元二値ベクトル | ReachableSquaresLoss (BCE) | Accuracy |
| 2 | 合法手の学習 | 1496次元二値ベクトル | LegalMovesLoss (ASL / BCE) | F1 Score |
| 3 | 評価値と指し手の学習 | Policy確率分布 + Value勝率 | KLDivLoss + BCEWithLogitsLoss | Policy Accuracy + Value Loss |

---

## Stage 1: 到達可能升の学習

### 損失関数: ReachableSquaresLoss

`BCEWithLogitsLoss` をラップした損失関数．9×9盤面の各升が駒の到達可能範囲に含まれるかどうかを
独立した二値分類として学習する．

### 選択理由

- **タスク特性**: 81升それぞれが独立した二値分類(到達可能/不可能)
- **BCEが最適**: 各ラベルが独立であり，multi-label binary classification の標準的な損失関数
- **数値安定性**: `BCEWithLogitsLoss` は内部で log-sum-exp トリックを使用し，sigmoid + BCE の数値的不安定性を回避

### pos_weight による不均衡調整

到達可能な升は通常少数(駒種と配置に依存)であるため，正例と負例の不均衡が存在する．
`--stage1-pos-weight` CLIオプションで正例の重みを調整可能．

- `pos_weight=1.0` (デフォルト): 正例と負例を等しく扱う
- `pos_weight > 1.0`: 正例(到達可能升)の損失を増幅し，recall向上
- 推奨: Stage 1 はタスクが単純で高精度(99%+)に到達するため，デフォルト値で通常十分

---

## Stage 2: 合法手の学習

### 損失関数: LegalMovesLoss

`AsymmetricLoss` (ASL) または `BCEWithLogitsLoss` を内部で使用する損失関数．
1496ラベルの合法手マスクをmulti-label binary classificationとして学習する．

### ASL 選択理由

#### 極端なクラス不均衡

将棋の各局面における合法手数は平均~20手であり，1496ラベル中わずか1.3%が正例である．
標準BCEでは大量の負例(非合法手)からの損失が支配的になり，
正例(合法手)の学習シグナルが希釈される．

#### Near-miss negatives の重要性

将棋の合法手判定では「惜しい負例」が学習上特に重要である:

- **駒がブロックする1マス先**: 飛車が他の駒でブロックされている場合，ブロック位置の1マス先は非合法だが，特徴空間上では合法手に非常に近い
- **成/不成のペア**: 成れない条件での成手バージョンは非合法だが，同じ移動先への不成は合法
- **王手放置の禁止**: 合法な移動先に見えるが，王手を放置するため非合法となる手

ASLの負例focusing(γ-)により，これらの「困難な負例」の損失を保持しつつ，
明らかに非合法な手(全く無関係な升への移動)の損失を抑制できる．

#### 正例/負例の独立制御

ASLの最大の利点は，正例(γ+)と負例(γ-)に独立したfocusing parameterを持つことである:

- **γ+=0 (推奨)**: 合法手を「漏れなく」学習する要件に最適．正例の損失を一切軽視しない
- **γ->0**: 容易な負例の損失を抑制し，困難な負例に集中

Focal Lossではγが正例/負例共通であるため，この独立制御ができない．

### 将棋ドメイン推奨パラメータ

| パラメータ | CLIオプション | 推奨値 | 探索範囲 | 説明 |
|-----------|-------------|-------|---------|------|
| γ+ | `--stage2-gamma-pos` | 0.0 | 固定推奨 | 正例の損失を一切軽視しない |
| γ- | `--stage2-gamma-neg` | 2.0 | 1.0-4.0 | 容易な負例の抑制度 |
| clip | `--stage2-clip` | 0.02 | 0.0-0.05 | 負例のハードクリッピング |

### パラメータ探索の推奨手順

1. **ベースライン確立**: デフォルト(ASL無効，標準BCE)でF1スコアを測定
2. **γ-の探索**: `--stage2-gamma-neg` を 1.0, 2.0, 3.0, 4.0 で比較(clip=0.0固定)
3. **clipの微調整**: 最良のγ-で `--stage2-clip` を 0.0, 0.01, 0.02, 0.05 で比較
4. **hidden_dim + Dropout**: 必要に応じて `--stage2-hidden-dim 512 --stage2-head-dropout 0.1` を追加

### Reference

Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021

---

## Stage 3: 評価値と指し手の学習

### Policy損失: KLDivLoss(reduction="batchmean")

#### ターゲット形式

前処理パイプラインで棋譜から出現率マップ(ソフトターゲット)を計算し，
float16の確率分布として保存する．`normalize_policy_targets` で
`legal_move_mask` を適用した上で確率分布に正規化する．

#### 選択理由

- **ターゲットが確率分布**: ソフトターゲット(出現率マップ)はラベル空間上の確率分布であり，
  分布間の距離を最小化するKL divergenceが理論的に最適
- **Cross Entropyとの関係**: ターゲット分布が固定の場合，KL divergenceの最小化は
  cross entropyの最小化と等価であるが，`KLDivLoss` はターゲットのエントロピー項を含むため，
  損失値の解釈性が向上する
- **reduction="batchmean"**: バッチ平均で正規化し，バッチサイズに依存しない安定した勾配を得る

### Value損失: BCEWithLogitsLoss

#### ターゲット形式

勝率 = win_count / total_count (0.0-1.0の連続値)．

#### 選択理由

- **二値確率の予測タスク**: 勝率は[0,1]区間の確率値であり，BCEが理論的に最適
- **logitsベースの数値安定性**: Value headはlogitsを出力し，
  損失関数内部でsigmoidが適用される(log-sum-exp trick)
- **autocast対応**: `BCEWithLogitsLoss` はPyTorchのAMP(Automatic Mixed Precision)と互換

---

## 設計原則

### 後方互換性

全ての損失関数パラメータはデフォルト値で既存動作と完全に一致する:

- Stage 1: `ReachableSquaresLoss(pos_weight=1.0)` = 従来BCE
- Stage 2: `LegalMovesLoss(gamma_pos=0.0, gamma_neg=0.0, clip=0.0)` = 従来BCE
- Stage 3: 変更なし

### Clean Architecture

損失関数はすべてdomain層(`src/maou/domain/loss/loss_fn.py`)に配置する．
損失関数の選択とパラメータ設定はinterface層(`src/maou/interface/learn.py`)で行い，
CLIオプションはinfra層(`src/maou/infra/console/learn_model.py`)で定義する．

依存フロー: `infra → interface → app → domain`

### AMP互換性

- **AsymmetricLoss**: forward内で `logits.float()` / `targets.float()` により
  FP16 → FP32キャストを明示的に実行(数値安定性確保)
- **BCEWithLogitsLoss**: PyTorch内部でAMP対応済み
- **KLDivLoss**: PyTorch内部でAMP対応済み
