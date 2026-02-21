# `--input-path` CLIオプション 徹底調査結果

## 調査日時
2026-02-21

## 1. `learn-model` コマンドでの `--input-path` 使用状況

### 1.1 CLI定義（L59-63）
```python
@click.option(
    "--input-path",
    help="Input file or directory path.",
    type=click.Path(exists=True, path_type=Path),
    required=False,
)
```

### 1.2 各ステージでの利用

#### **Stage 3（デフォルト）での使用**
- **定義**: L699で関数パラメータ化
- **用途**: 前処理済みデータ（preprocessing）の入力
- **コードパス**:
  - L889-890: `input_path is not None` チェック → `FileSystem.collect_files(input_path)` で全ファイル収集
  - L891-905: ストリーミングモード判定（複数ファイルの場合ストリーミング使用）
  - **非マルチステージ時の使用** (L1313-1396):
    - L1319-1350: `file_paths_for_streaming` 変数で管理、StreamingFileSource生成
    - L1339-1346: 訓練/検証に分割、ストリーミングモード有効化
    - L1353-1396: `learn.learn()` に `datasource=datasource` として渡す

#### **Stage 1での使用**
- **非マルチステージ時のみ**（Stage 1専用データなしの場合）
- Stage 1は `--stage1-data-path` で指定するため、`input_path` は使われない
- マルチステージMode（`--stage 1` 指定）で `--stage1-data-path` がない場合、`input_path` は利用されない

#### **Stage 2での使用**
- 同様に `--stage2-data-path` で指定
- `input_path` は使われない

#### **Stage 3（マルチステージ時）での使用**
- **定義**: L1138-1189
- **重要**: L1190-1244の FALLBACK ロジック
  ```python
  elif input_path is not None:
      # Fallback: use --input-path for Stage 3 data
      _s3_paths = (
          collected_paths
          if collected_paths is not None
          else FileSystem.collect_files(input_path)
      )
  ```
- **意味**: 
  - マルチステージ時に `--stage3-data-path` が指定されていない場合
  - `--input-path` を **Stage 3データとしてのみ** 使用
  - つまり、マルチステージ時の `--input-path` は Stage 3 専用データパスとして機能

#### **`all` ステージでの使用**
- L873-878: `is_multi_stage` フラグで判定
- `--stage all` の場合もマルチステージ扱い
- Stage 1/2 では `--stage1-data-path` / `--stage2-data-path` 必須
- Stage 3 では `--input-path` または `--stage3-data-path` を使用

### 1.3 `--stage3-data-path` との関係

| 条件 | 使用するデータパス | コード位置 |
|------|------------------|---------|
| `--stage3-data-path` が指定されている | `--stage3-data-path` | L1138-1189 |
| `--stage3-data-path` がない + `--input-path` 指定 | `--input-path` （fallback） | L1190-1244 |
| `--input-path` のみ指定 + 非マルチステージ | `--input-path` （非マルチステージ） | L1313-1396 |

**結論**: 
- `--stage3-data-path` が優先度が高い（明示的）
- `--input-path` は fallback として Stage 3 に使用
- 異なる用途ではなく、**`--stage3-data-path` の後方互換性オプション**

### 1.4 `collected_paths` 構築ロジック（L889-890）

```python
if input_path is not None:
    collected_paths = FileSystem.collect_files(input_path)
```

- **単一ファイル**: `collected_paths` は1要素リスト
- **ディレクトリ**: 再帰的に全ファイル収集
- **ストリーミング判定** (L891):
  ```python
  if no_streaming or len(collected_paths) < 2:
      # Map-style (全データをメモリに一括読み込み)
      datasource = FileDataSource.FileDataSourceSpliter(...)
  else:
      # Streaming (ファイルごとに遅延読み込み)
      file_paths_for_streaming = collected_paths
  ```

---

## 2. 他のコマンドでの `--input-path` 使用

### 2.1 `utility.benchmark-dataloader` (L24-29)
- **用途**: ベンチマーク測定用の入力データ
- **使用**: L248-258
  ```python
  if input_path is not None:
      datasource = FileDataSource.FileDataSourceSpliter(
          file_paths=FileSystem.collect_files(input_path),
          ...
      )
  ```
- **特徴**: 学習と同じ入力形式をサポート

### 2.2 `utility.benchmark-training` (L416-420)
- **用途**: ベンチマーク測定用の入力データ
- **使用**: L907-917
  ```python
  if input_path is not None:
      datasource = FileDataSource.FileDataSourceSpliter(
          file_paths=FileSystem.collect_files(input_path),
          ...
      )
  ```
- **特徴**: 学習と同じロジック

### 2.3 `utility.generate-stage2-data` (L1169-1172)
- **用途**: HCPE ファイル入力
- **オプション名**: `--input-path`
- **使用**: HCPE → 前処理データ変換用
- **注記**: 学習コマンドの `--input-path` とは異なるセマンティクス

---

## 3. `--input-path` を廃止した場合の影響範囲

### 3.1 コマンド別影響

| コマンド | 使用箇所 | 影響度 | 代替案 |
|---------|--------|------|------|
| `learn-model` (Stage 3非マルチ) | 必須 | **HIGH** | なし（必須） |
| `learn-model` (Stage 3マルチ) | Optional (fallback) | MEDIUM | `--stage3-data-path` に統一可 |
| `learn-model` (Stage 1/2) | 不使用 | NONE | N/A |
| `benchmark-dataloader` | 必須 | **HIGH** | 完全に削除不可 |
| `benchmark-training` | 必須 | **HIGH** | 完全に削除不可 |
| `generate-stage2-data` | オプション | LOW | そのまま保持可 |

### 3.2 ユーザーへの影響

#### **廃止不可の理由**
1. **非マルチステージ学習 (デフォルト)**: `--input-path` が必須
   - `maou learn-model --input-path data/` が標準使用パターン
2. **ベンチマークコマンド**: `--input-path` が入力の主要手段
3. **テスト互換性**: test_learn_model_cli.py でテスト依存

#### **廃止可能な範囲**
- マルチステージ Stage 3 の fallback メカニズムのみ
- 代わりに `--stage3-data-path` を明示要求可

---

## 4. 実装上の設計意図

### 4.1 `--input-path` の二重役割

```
┌─────────────────────────────────────┐
│    CLI: --input-path               │
└─────────────────────────────────────┘
         ↓
    ┌────────────────────────────────────────┐
    │ 非マルチステージ                      │
    │ └→ Stage 3 データ直接使用              │
    │ マルチステージ                        │
    │ └→ --stage3-data-path 指定時: 無視    │
    │ └→ --stage3-data-path 未指定: fallback │
    └────────────────────────────────────────┘
```

### 4.2 なぜこの設計か？

1. **後方互換性**: 従来の `--input-path` での Stage 3 学習をサポート
2. **段階的移行**: Stage1/2/3 用に個別オプション (`--stage1-data-path` 等) を導入
3. **Flexibility**: マルチステージユーザーが明示的に `--stage3-data-path` を指定可

---

## 5. テスト影響分析

### 5.1 CLI互換性テスト

**test_cli_option_compatibility.py**:
- L140: `"input-path"` が `training_params` に含まれている
- 意味: `learn-model` と `benchmark-training` の両方に必須
- 削除不可

### 5.2 学習モデルテスト

**test_learn_model_cli.py**:
- L33-45: `--input-path` でサンプルファイル指定（skipped）
- L91-111: `--stage1-data-path` 使用時の テスト

---

## 6. ドキュメンテーション状況

**docs/commands/learn_model.md**:
- L21: `--input-path PATH` 説明あり
- L114: `--stage3-data-path` との関係 **明示されていない**
- **問題**: ユーザーに `--input-path` vs `--stage3-data-path` の違いが不明確

---

## 7. 廃止判定基準

### `--input-path` は以下の理由で **廃止不可**

1. ✓ **Stage 3 非マルチステージ**: 唯一の入力方法
2. ✓ **ベンチマークコマンド**: `--input-path` が主要入力
3. ✓ **下位互換性**: 既存スクリプトが依存
4. ✓ **テスト依存**: CLI互換性テストで要求

### `--input-path` の廃止可能シナリオ

- **Stage 3 マルチステージで `--stage3-data-path` 必須化**: 可（従来版で使用例少）
- **非マルチステージ廃止**: 不可（デフォルト動作）
- **`benchmark-*` コマンドでの廃止**: 不可（代替なし）

---

## 8. 推奨アクション

### 短期（ドキュメント改善）
- [ ] docs/commands/learn_model.md に明確な `--input-path` vs `--stage3-data-path` 説明追加
- [ ] 使用例で非マルチステージと マルチステージを区別

### 中期（Deprecation対応）
- [ ] `--input-path` の動作を明示的に（warning で説明）
- [ ] マルチステージ時に `--stage3-data-path` を推奨

### 長期（バージョン2.0）
- [ ] マルチステージで `--stage3-data-path` 必須化検討
- [ ] 非マルチステージで `--input-path` は継続（後方互換性）
