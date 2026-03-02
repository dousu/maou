# torch.compile と state_dict の互換性

## 概要

`torch.compile()` でモデルをコンパイルすると，`state_dict()` のキーに `_orig_mod.` プレフィックスが付与される．
本ドキュメントでは，現在の対応状況を整理する．

## 背景: `_orig_mod.` プレフィックス

```python
model = Network(...)
compiled = torch.compile(model)

model.state_dict()     # {"backbone.layer.weight": ...}
compiled.state_dict()  # {"_orig_mod.backbone.layer.weight": ...}
```

`torch.compile()` は `OptimizedModule` でラップし，元モジュールを `_orig_mod` 属性として保持する．
パラメータ自体は同一オブジェクトを共有しており，学習時の勾配更新は元モジュールにも反映される．

## 現在の対応状況 (v0.5.5)

### ユーティリティ

`ModelIO._strip_orig_mod_prefix()` を追加し，プレフィックス除去を一元化した．

### 対応済み箇所

| 箇所 | ファイル | 対応内容 |
|------|---------|---------|
| `split_state_dict()` | `model_io.py` | プレフィックス除去後に分類・保存 |
| `load_backbone()` | `model_io.py` | ロード後にプレフィックス除去 |
| `load_policy_head()` | `model_io.py` | ロード後にプレフィックス除去 |
| `load_value_head()` | `model_io.py` | ロード後にプレフィックス除去 |
| `load_reachable_head()` | `model_io.py` | ロード後にプレフィックス除去 |
| `load_legal_moves_head()` | `model_io.py` | ロード後にプレフィックス除去 |
| `save_model()` | `model_io.py` | ユーティリティで除去を簡素化 |
| `_load_component_state_dict()` | `dl.py` | 双方向対応(追加/除去) + unexpected_keys 検知 |
| `HeadlessNetwork.load_state_dict()` | `network.py` | フィルタリング前にプレフィックス除去 |

### Stage 別の状況

| Stage | compile 対象 | 保存方法 | プレフィックス問題 |
|-------|-------------|---------|------------------|
| Stage 1/2 | `Stage1/2ModelAdapter` | `self.backbone.state_dict()` | なし(backbone は非コンパイル) |
| Stage 3 | `Stage3ModelAdapter` | `self.model.state_dict()` | なし(Network は非コンパイル) |

### Stage 3 アダプターパターン

Stage 1/2 と同様に，Stage 3 でもアダプターパターンを採用している．
`Network` 本体ではなく `Stage3ModelAdapter` を `torch.compile()` の対象とすることで，
`Network.state_dict()` にプレフィックスが付かず，保存時の除去処理が不要になる．

```python
# Stage 3 アダプターパターン
adapter = Stage3ModelAdapter(network)
compiled = torch.compile(adapter)  # アダプターをコンパイル
network.state_dict()  # プレフィックスなし → 除去不要
```

### `_load_component_state_dict()` の安全性

`_load_component_state_dict()` は `strict=False` で部分読み込みを行う．
これはコンポーネント単位(backbone のみ / head のみ)のロードに必要な設定である．

安全性を確保するため以下の対策を実施している:

- **unexpected_keys の検知**: ロードした state_dict にモデルに存在しないキーがある場合，
  `RuntimeError` を送出する．アーキテクチャ不一致や誤ったチェックポイントを早期に検知できる．
- **component パラメータ**: ロード対象のコンポーネント名をログに出力し，
  問題発生時のデバッグを容易にする．
- **missing_keys のログ**: 部分ロードで期待される missing_keys を INFO レベルで記録する．
