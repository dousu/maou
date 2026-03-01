# ADR-005: 指し手別勝率(Move Win Rates)の導入

## ステータス

✅ **Accepted** - 2026-02-28議論開始，2026-03-01 Phase 1承認

## コンテキスト

### 現状の問題

Stage 3のpreprocessデータは，同一局面を集約した上で以下の情報を保持している:

| フィールド | 内容 | 型 |
|---|---|---|
| `moveLabel[i]` | 指し手 i の選択率 (`move_count[i] / total_count`) | float32[1496] |
| `resultValue` | 局面全体の勝率 (`win_count / total_count`) | float32 |

この設計では**指し手と勝率の関係が失われている**．具体的には:

- 「頻繁に選ばれるが実は勝率が低い悪手」を識別できない
- 「少数派だが勝率が高い好手」を見逃す
- モデルは「棋譜で人気の手 = 良い手」という仮定のもとで学習している

### サンプル数の問題

指し手別の勝率を導入する際，サンプル数が少ない局面では勝率が0%または100%に偏る．
これは学習にノイズを導入し，過学習の原因となる．

---

## 決定事項

### 1. 指し手別勝率(`moveWinRate`)をpreprocess出力に追加

中間ストア(DuckDB)に指し手ごとの勝ち数を追加し，
最終出力に指し手別勝率を含める．

#### 中間ストアの変更

```
既存:
  move_label_indices:  [10, 20, 55]      # 指された手のインデックス
  move_label_values:   [8,  5,  2]       # 各手の出現回数
  win_count:           10.0              # 局面全体の勝ち数
  count:               15               # 局面全体の出現数

追加:
  move_win_indices:    [10, 20, 55]      # (move_label_indicesと同一)
  move_win_values:     [7.0, 2.0, 1.0]  # 各手を選んだときの勝ち数(float)
```

`move_win_indices` は `move_label_indices` と常に同一であるため，
実装上はインデックス配列を共有し，値配列(`move_win_values`)のみを追加する．

#### 最終出力スキーマの変更

```
既存 (変更なし):
  id:                  uint64
  boardIdPositions:    List[List[uint8]]   # 9×9盤面
  piecesInHand:        List[uint8]         # 持ち駒(14要素)
  moveLabel:           List[float32]       # 指し手選択率(1496要素)
  resultValue:         float32             # 局面全体の勝率

追加:
  moveWinRate:         List[float32]       # 指し手別勝率(1496要素，フォールバック適用済み)
  bestMoveWinRate:     float32             # 最善手の勝率(max(moveWinRate)，フォールバック時は0.5)
```

### 2. フォールバック戦略: 合法手への均等配分(方式B)

指し手のサンプル数が閾値未満の場合，その局面の `moveWinRate` 全体を
合法手(=棋譜中で指された手)への均等配分に置き換える．

```
サンプル数 ≥ 閾値:
  moveWinRate[i] = move_win_count[i] / move_count[i]   (各手の実勝率)

サンプル数 < 閾値 (フォールバック):
  moveWinRate[i] = 1/N   (N = 棋譜中で指された手の種類数，全合法手に均等配分)
```

#### フォールバック値の選択理由

| 候補 | 採否 | 理由 |
|---|---|---|
| 均等配分 (`1/N`) | **採用** | 「この局面の手の優劣は不明」を正直に表現 |
| 局面勝率 (`resultValue`) | 却下 | 勝勢の局面で全手が高勝率と誤認させる |
| 固定値 `0.5` | 却下 | 局面の状況を無視した中立値 |
| NaN + マスク | 却下 | 推論時(maou evaluate)にモデルへ無意味な値を渡すことになる |

#### 閾値の設定

- **閾値はpreprocess側で固定**: CLIパラメータとして指定
- 閾値を変更する場合はpreprocessのデータ生成自体をやり直す
- **推奨範囲**: 2〜4(実データの分布に基づく直感値)
- 閾値の判定は**局面の出現回数(`count`)**に対して行う

### 3. Policy教師信号の設計: 2つの方向性

`moveWinRate` を用いたpolicy教師信号の構成について，
2つの方向性を検討する．いずれも学習時のCLI選択で切り替え可能とする．

#### 方向1: 加重方式

```python
raw[i] = moveLabel[i] × moveWinRate[i]
policy_target[i] = raw[i] / Σ raw[j]   # 正規化
```

- **特徴**: 棋譜のコンセンサス(選択率)を活かしつつ，悪手を抑制
- **メリット**: 選択率の情報を保持し，安定した学習が期待できる
- **デメリット**: 「頻繁に指されるがやや悪い手」が残りやすい

#### 方向2: 勝率のみ

```python
policy_target[i] = moveWinRate[i] / Σ moveWinRate[j]   # 合法手で正規化
```

- **特徴**: 選択率に関係なく「勝てる手」のみを学習
- **メリット**: 既存の棋譜データのバイアスに惑わされない
- **デメリット**: 少数派の手の勝率ノイズに敏感(フォールバックで緩和)

#### フォールバック時の教師信号

いずれの方向でも，フォールバック適用局面では:

```python
policy_target[i] = 1/N   # 全合法手に均等配分
```

これは「手の優劣が不明なため，モデルに特定の手を優先させない」ことを意味する．

また，フォールバック適用時の `bestMoveWinRate` は固定値 `0.5` とする．
これは「勝敗不明 = 中立値」を表し，Value Headの教師信号として
局面の優劣判断にバイアスを与えない．

### 4. 損失関数: GCELossを継続使用

現状のGCELoss(Generalized Cross Entropy)はノイズに強い特性を持ち，
moveWinRateベースの教師信号でも有効と判断する．

| 項目 | 判断 | 理由 |
|---|---|---|
| GCEの継続使用 | **可** | ノイズ耐性はmoveWinRateの勝率ノイズに対しても有効 |
| パラメータ `q` | **要再調整** | 現状の `q=0.7` はmoveLabelに対して最適化された値 |
| 合法手マスキング | **継続** | 変更なし |

フォールバック(均等配分)の局面に対するGCEの挙動:
- 平坦な分布をターゲットとするため，モデルへの勾配が小さくなる
- `q` が小さいほどこの傾向が強まり，フォールバック局面の学習への影響が自然に抑制される

### 5. Value Head: 現状維持(resultValue)

Policy教師信号の変更とValue教師信号の変更を同時に行うと，
効果の切り分けが困難になるため，まずpolicy側のみ変更する．

| Value target候補 | 採否 | 理由 |
|---|---|---|
| `resultValue`(現状維持) | **採用** | policy変更の効果を先に検証 |
| `max(moveWinRate)` | 将来検討 | 最善手の勝率を予測する方が推論時と整合的 |
| 加重平均 | 見送り | resultValueと近い値になり変更の意味が薄い |

### 6. メトリクスの見直し

#### 既存メトリクスの扱い

| メトリクス | 現状の意味 | moveWinRate導入後 | 対応 |
|---|---|---|---|
| `policy_cross_entropy` | moveLabelとの乖離 | moveWinRate分布との乖離 | ターゲット変更に伴い自動的に意味が変わる |
| `policy_top5_accuracy` | 実際に指された手がtop5にあるか | 意味が変わる | 要見直し |
| `policy_f1_score` | 指された手の再現率 | 意味が変わる | 要見直し |
| `value_brier_score` | 勝率予測のMSE | 変更なし | 維持 |

#### 追加すべきメトリクス

1. **`policy_top1_win_rate`**:
   モデルのtop1予測の手に対応する実際の勝率．
   これが上昇していれば「強い手を選べるようになっている」ことを示す．

2. **`policy_move_label_ce`** (参考値):
   moveWinRateで学習しつつ，従来のmoveLabelとのCross Entropyも記録する．
   「人間の棋譜からどれだけ離れているか」のモニタリングに使用し，
   極端な乖離は学習の暴走を示唆する．

3. **`policy_expected_win_rate`**:
   `Σ softmax(policy_logits)[i] × moveWinRate[i]` で算出．
   モデルの予測分布から期待される勝率．

---

## 理由

### なぜ指し手別勝率が必要か

現状のmoveLabelは「棋譜における選択頻度」であり，
「良い手」と「よく指される手」を区別できない．
moveWinRateの導入により，モデルは「勝ちにつながる手」を直接学習できるようになる．

### なぜ均等配分フォールバックか

- 推論時(`maou evaluate`)にモデルへ渡す値として意味がある
- NaNやマスクと異なり，特別な処理が不要
- 「情報がない = 全手が等価」という最もバイアスの少ない仮定

### なぜpreprocess側で閾値固定か

- **出力データサイズの抑制**: 閾値を学習側で適用する場合，指し手ごとのカウント数を
  出力に含める必要があり，データサイズが増大する．preprocess側でフォールバック適用済みの
  `moveWinRate` を出力することで，カウント情報を出力から除外できる
- 推論時との一貫性: preprocessで確定した値がそのままモデルの入出力になる
- 閾値の変更は実験の再現性の観点からもデータ生成からやり直す方が健全

---

## 実装方針(概要)

### preprocess側の変更

1. `_process_single_array()` で指し手ごとの勝ち数を集計
2. 中間ストア(DuckDB)に `move_win_values` カラムを追加
3. スパースマージ用Rust UDFを拡張(`add_sparse_arrays_rust` を流用)
4. finalize時にフォールバック適用 + `moveWinRate` 計算
5. Arrow IPCスキーマに `moveWinRate` カラム追加

### 学習側の変更

1. DatasetクラスでmoveWinRateを読み込み
2. policy教師信号の構成をCLIで切り替え可能に(方向1 / 方向2 / 従来moveLabel)
3. GCELossの `q` パラメータ再調整
4. 新規メトリクス追加(top1 win rate，moveLabel CE，expected win rate)

### CLI変更

- `maou pre-process`: `--win-rate-threshold` パラメータ追加
- `maou learn-model`: `--policy-target-mode {move-label, win-rate, weighted}` パラメータ追加

---

## 未決定事項

- [ ] GCE `q` パラメータの最適値(moveWinRateに対する再調整)
- [ ] フォールバック閾値の最終決定(2, 3, or 4)
- [ ] フォールバック率の実データでの確認(閾値ごとの影響度)
- [ ] Value headの教師信号見直し(policy変更の効果検証後)
- [ ] `policy_top5_accuracy` / `policy_f1_score` の再定義

---

## 参考: データフロー(変更後)

```
HCPE入力
  │
  ▼
_process_single_array()
  ├── hash計算
  ├── move_label_count集計 (既存)
  ├── move_win_count集計   (★新規: 各手の勝ち数)
  └── result_value集計     (既存)
  │
  ▼
中間ストア(DuckDB)
  ├── move_label (sparse indices + values)  (既存)
  ├── move_win   (sparse values)            (★新規: indicesは共有)
  ├── win_count                             (既存)
  └── count                                 (既存)
  │
  ▼
finalize
  ├── moveLabel = move_label / count        (既存)
  ├── moveWinRate = move_win / move_label   (★新規)
  │   └── count < 閾値 → 1/N に置換        (★フォールバック)
  └── resultValue = win_count / count       (既存)
  │
  ▼
Arrow IPC出力 (.feather)
  ├── boardIdPositions, piecesInHand        (既存)
  ├── moveLabel                             (既存)
  ├── moveWinRate                           (★新規)
  ├── bestMoveWinRate                       (★新規: フォールバック時は0.5)
  └── resultValue                           (既存)
```
