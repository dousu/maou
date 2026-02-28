# torch.compile と state_dict の互換性

## 概要

`torch.compile()` でモデルをコンパイルすると，`state_dict()` のキーに `_orig_mod.` プレフィックスが付与される．
本ドキュメントでは，現在の対応状況と残っている改善タスクを整理する．

## 背景: `_orig_mod.` プレフィックス

```python
model = Network(...)
compiled = torch.compile(model)

model.state_dict()     # {"backbone.layer.weight": ...}
compiled.state_dict()  # {"_orig_mod.backbone.layer.weight": ...}
```

`torch.compile()` は `OptimizedModule` でラップし，元モジュールを `_orig_mod` 属性として保持する．
パラメータ自体は同一オブジェクトを共有しており，学習時の勾配更新は元モジュールにも反映される．

## 現在の対応状況 (v0.2.7)

### ユーティリティ

`ModelIO._strip_orig_mod_prefix()` を追加し，プレフィックス除去を一元化した．

### 対応済み箇所

| 箇所 | ファイル | 対応内容 |
|------|---------|---------|
| `split_state_dict()` | `model_io.py` | プレフィックス除去後に分類・保存 |
| `load_backbone()` | `model_io.py` | ロード後にプレフィックス除去 |
| `save_model()` | `model_io.py` | ユーティリティで除去を簡素化 |
| `_load_component_state_dict()` | `dl.py` | 双方向対応(追加/除去) |
| `HeadlessNetwork.load_state_dict()` | `network.py` | フィルタリング前にプレフィックス除去 |

### Stage 別の状況

| Stage | compile 対象 | 保存方法 | プレフィックス問題 |
|-------|-------------|---------|------------------|
| Stage 1/2 | `Stage1/2ModelAdapter` | `self.backbone.state_dict()` | なし(backbone は非コンパイル) |
| Stage 3 | `Network` 本体 | `trained_model.state_dict()` | `_strip_orig_mod_prefix()` で除去 |

## 残タスク

### 1. Stage 3 アダプターパターン導入 (リファクタリング)

**現状**: Stage 3 では `Network` 本体を直接 `torch.compile()` する．
そのため `state_dict()` に `_orig_mod.` プレフィックスが付き，保存時に除去が必要になる．

**改善案**: Stage 1/2 と同様にアダプターパターンを導入する．

```python
# 現在の Stage 3
compiled_model = torch.compile(network)  # Network 本体をコンパイル
compiled_model.state_dict()  # _orig_mod. 付き → 除去が必要

# 改善後
class Stage3ModelAdapter(torch.nn.Module):
    def __init__(self, network: Network):
        super().__init__()
        self.network = network

    def forward(self, inputs):
        return self.network(inputs)

adapter = Stage3ModelAdapter(network)
compiled = torch.compile(adapter)  # アダプターをコンパイル
network.state_dict()  # プレフィックスなし → 除去不要
```

**メリット**:
- `_strip_orig_mod_prefix()` による保存時の除去が不要になる
- Stage 1/2/3 で一貫したパターンになる
- 保存されたチェックポイントが常にクリーンなキーを持つ

**影響範囲**:
- `src/maou/app/learning/dl.py` の `Learning` クラス
- `src/maou/interface/learn.py` の Stage 3 学習フロー
- `src/maou/app/learning/model_io.py` の `save_model()`

**優先度**: 低 (現在の `_strip_orig_mod_prefix()` で実害なし)

### 2. head ロード関数のプレフィックス除去

**現状**: `load_policy_head()`，`load_value_head()`，`load_reachable_head()`，`load_legal_moves_head()` は
プレフィックス除去を行わず `torch.load()` の結果をそのまま返す．

**理由**: 現在の保存フローでは head の `state_dict()` にプレフィックスが付かないため問題は顕在化しない．
ただし，外部で保存されたコンパイル済みモデルのチェックポイントからヘッドを分離して保存した場合は，
プレフィックス付きのキーが保存される可能性がある．

**改善案**: `load_backbone()` と同様に `_strip_orig_mod_prefix()` を適用する．

**優先度**: 低 (通常フローでは発生しない)

### 3. `_load_component_state_dict()` の `strict=False` 検討

**現状**: `dl.py` の `_load_component_state_dict()` は常に `strict=False` で部分読み込みを行う．
これにより，キーの不一致がサイレントに無視される可能性がある．

**検討事項**:
- backbone 全体をロードする場合は `strict=True` の方が安全
- 一方，head のみのロードでは backbone キーが missing になるため `strict=False` が必要
- コンポーネント種別に応じて strict モードを切り替えるのが理想

**優先度**: 中 (パラメータのロード漏れに気づきにくい)
