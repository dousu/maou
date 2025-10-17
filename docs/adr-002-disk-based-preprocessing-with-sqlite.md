# ADR-002: ディスクベース前処理とSQLite技術選定

## ステータス

✅ **Accepted** - 2025-10-13実装完了

## コンテキスト

Maou将棋AIプロジェクトの前処理（`pre-process`）において，大規模データセット処理時にメモリ不足が発生していた．

### 発生していた問題

1. **メモリ不足エラー**: 2020年のfloodgateデータ（13,393,044局面，10,271,289ユニーク局面）を処理する際，推定211GBのメモリが必要
2. **実行環境の制約**: 利用可能なメモリが約50GBの環境で実行不可能
3. **スケーラビリティ**: データ量の増加に伴い，メモリ要件が線形に増加

### メモリ使用量の内訳

前処理では，ユニーク局面ごとに以下のデータを`intermediate_dict`（Pythonの辞書）で保持：

```python
intermediate_dict[hash_id] = {
    "count": int32,                           # 4 bytes
    "winCount": float32,                      # 4 bytes
    "moveLabelCount": np.int64[1496],        # 11,968 bytes
    "features": np.uint8[104, 9, 9],         # 8,424 bytes
    "legalMoveMask": np.uint8[1496],         # 1,496 bytes
}
# + hash_id (uint64): 8 bytes
# + Python辞書オーバーヘッド: 約200 bytes
# = 合計約22,104 bytes/エントリ
```

10,271,289ユニーク局面 × 22,104 bytes ≈ **211GB**

## 決定事項

### 1. ディスクベース中間データストアの導入

**決定**: メモリベースの`intermediate_dict`をディスクベースの永続化ストアに置き換える

**理由**:
- メモリ使用量を劇的に削減（211GB → 1-5GB，約97%削減）
- 処理中断時のリカバリ機能
- より大規模なデータセットへの対応

### 2. SQLiteの技術選定

**決定**: ディスクベースストアの実装にSQLiteを採用

**理由**:

#### SQLiteの利点

1. **重複データの効率的な更新**
   - ユニーク局面（`hash_id`）に対して`count`, `winCount`, `moveLabelCount`を累積的に更新
   - SQLiteの`UPDATE`文により既存レコードを効率的に更新
   - トランザクションによるバッチ書き込みで高速化

2. **インデックスによる高速検索**
   - `PRIMARY KEY (hash_id)`によりO(log N)での重複チェック
   - 10百万レコードでも高速なアクセス

3. **トランザクションによるデータ整合性**
   - バッチ書き込み時の原子性保証
   - 処理中断時のロールバック機能

4. **標準ライブラリ**
   - Pythonの標準ライブラリ（`sqlite3`）で利用可能
   - 追加の依存関係が不要

5. **メモリ効率**
   - データはディスク上に保存
   - 必要な分だけメモリにロード
   - キャッシュサイズを制御可能（`PRAGMA cache_size`）

#### 代替案の検討

**NumPy memmap**:
- ✅ 高速なI/O
- ❌ 重複チェックと更新処理が複雑（線形探索または別途インデックス管理が必要）
- ❌ 可変長データの扱いが困難

**HDF5**:
- ✅ 階層的データ構造
- ✅ 圧縮機能
- ❌ 追加の依存関係（`h5py`）
- ❌ 頻繁な更新操作には不向き

**Parquet**:
- ✅ カラム指向ストレージで圧縮効率が高い
- ❌ 追加の依存関係（`pyarrow`）
- ❌ ランダムアクセスと更新が非効率

**LevelDB/RocksDB**:
- ✅ 高速なキーバリューストア
- ❌ 追加の依存関係とバイナリ配布の複雑さ
- ❌ オーバースペック

**選定理由**: SQLiteは標準ライブラリで利用可能で，重複チェックと更新処理に最適なバランスを提供

## 実装詳細

### アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│ PreProcess                                                  │
│  ├─ transform()                                             │
│  │   ├─ IntermediateDataStore初期化                        │
│  │   ├─ バッチ処理ループ                                   │
│  │   │   └─ merge_intermediate_data()                      │
│  │   │       └─ IntermediateDataStore.add_or_update_batch()│
│  │   └─ aggregate_intermediate_data()                      │
│  │       └─ IntermediateDataStore.finalize_to_array()      │
│  └─ cleanup (自動削除)                                      │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ IntermediateDataStore (domain/data/intermediate_store.py)  │
│  ├─ SQLiteデータベース                                      │
│  │   └─ intermediate_data テーブル                         │
│  │       ├─ hash_id (TEXT PRIMARY KEY)                     │
│  │       ├─ count (INTEGER)                                │
│  │       ├─ win_count (BLOB, pickled)                      │
│  │       ├─ move_label_count (BLOB, pickled numpy array)  │
│  │       ├─ features (BLOB, pickled numpy array)          │
│  │       └─ legal_move_mask (BLOB, pickled numpy array)   │
│  ├─ バッチバッファ（メモリ）                               │
│  │   └─ 1000レコード単位でフラッシュ                      │
│  └─ 最適化                                                  │
│      ├─ PRAGMA journal_mode=WAL                            │
│      ├─ PRAGMA synchronous=NORMAL                          │
│      └─ PRAGMA cache_size=-64000 (64MB)                    │
└─────────────────────────────────────────────────────────────┘
```

### データフロー

1. **バッチ処理**: HCPEデータを読み込み，ユニーク局面を計算
2. **バッファリング**: 中間データをメモリバッファに蓄積（デフォルト1000エントリ）
3. **フラッシュ**: バッファが閾値に達したらSQLiteに書き込み
   - 既存レコードの場合: `UPDATE`で累積
   - 新規レコードの場合: `INSERT`
4. **最終化**: 全データ処理後，SQLiteから読み出して最終配列を生成
5. **クリーンアップ**: 一時データベースファイルを自動削除

### SQLiteスキーマ設計

```sql
CREATE TABLE IF NOT EXISTS intermediate_data (
    hash_id TEXT PRIMARY KEY,              -- uint64をTEXTで保存
    count INTEGER NOT NULL,                -- 出現回数
    win_count BLOB NOT NULL,               -- 勝利数（pickled float32）
    move_label_count BLOB NOT NULL,        -- 手のラベル数（pickled np.int64[1496]）
    features BLOB NOT NULL,                -- 特徴量（pickled np.uint8[104,9,9]）
    legal_move_mask BLOB NOT NULL          -- 合法手マスク（pickled np.uint8[1496]）
);
```

**設計上の注意点**:

1. **hash_idをTEXTで保存**: SQLiteのINTEGERはuint64の大きな値を扱えないため
2. **BLOBでnumpy配列を保存**: pickleを使用して効率的にシリアライズ
3. **win_countもBLOBで保存**: float32値がREAL型で正しく扱われない問題を回避

### パフォーマンス最適化

```python
self._conn.execute("PRAGMA journal_mode=WAL")      # Write-Ahead Logging
self._conn.execute("PRAGMA synchronous=NORMAL")    # fsync削減
self._conn.execute("PRAGMA cache_size=-64000")     # 64MBキャッシュ
```

- **WALモード**: 読み書き並行性の向上
- **NORMAL同期**: パフォーマンスとデータ安全性のバランス
- **大きなキャッシュ**: メモリ内でのバッファリング

### CLIインターフェース

```bash
poetry run maou pre-process \
  --input-path hcpe/floodgate/2020 \
  --output-dir preprocess/floodgate/2020 \
  --process-max-workers 1 \
  --intermediate-cache-dir /tmp/cache \      # オプション
  --intermediate-batch-size 1000             # オプション
```

## 性能評価

### 小規模テスト（171,361局面，154,442ユニーク）

- **処理時間**: 約2分
- **メモリ使用量**: 約1-2GB（バッファサイズに依存）
- **ディスク使用量**: 約1GB（一時データベース）

### 推定性能（10,271,289ユニーク局面）

- **処理時間**: 約2-3時間（線形スケール想定）
- **メモリ使用量**: 約1-5GB（バッファサイズに依存）
- **ディスク使用量**: 約30-50GB（一時データベース + 最終出力）

### メモリ削減効果

| 項目 | 改善前 | 改善後 | 削減率 |
|------|--------|--------|--------|
| メモリ使用量 | 211GB | 1-5GB | 97% |
| 最小メモリ要件 | 256GB | 8GB | 97% |

## 影響範囲

### 変更されたファイル

1. **新規**: `src/maou/domain/data/intermediate_store.py` (314行)
   - `IntermediateDataStore`クラスの実装

2. **修正**: `src/maou/app/pre_process/hcpe_transform.py`
   - `intermediate_dict`を`IntermediateDataStore`に置き換え
   - 一時ディレクトリの自動管理
   - クリーンアップ処理の追加

3. **修正**: `src/maou/interface/preprocess.py`
   - 新パラメータ`intermediate_cache_dir`, `intermediate_batch_size`の追加

4. **修正**: `src/maou/infra/console/pre_process.py`
   - CLIオプションの追加

### 後方互換性

✅ **互換性あり**: 既存のCLIインターフェースは変更なしで動作

- 新しいオプションはオプショナル
- デフォルト動作は自動的に一時ディレクトリを使用

## 今後の改善案

### 短期的な改善

1. **並列処理の最適化**
   - 現在は`--process-max-workers 1`のみ対応
   - マルチワーカーでの中間データマージの最適化

2. **プログレス表示の改善**
   - SQLiteへの書き込み進捗の可視化
   - 推定残り時間の表示

### 長期的な改善

1. **代替バックエンドの検討**
   - より高速なキーバリューストア（RocksDB等）の評価
   - 大規模データセット（100億局面以上）での性能測定

2. **圧縮の導入**
   - BLOBデータのZstd圧縮
   - ディスク使用量のさらなる削減

3. **分散処理**
   - 複数マシンでの並列前処理
   - 中間データのマージ戦略

## 参考資料

- [SQLite Performance Tuning](https://www.sqlite.org/pragma.html)
- [Pickle Protocol Documentation](https://docs.python.org/3/library/pickle.html)
- [NumPy Structured Arrays](https://numpy.org/doc/stable/user/basics.rec.html)

## レビューコメント

実装レビュー時に以下の点が議論された：

1. **Q**: NumPyのmemmapの方が効率的では？
   **A**: 重複チェックと更新処理の複雑さから，SQLiteの方が実装とメンテナンスが容易．小規模テストでも実用的な性能を確認．

2. **Q**: uint64のハッシュ値をTEXTで保存する理由は？
   **A**: SQLiteのINTEGER型はuint64の大きな値（特に上位ビットが1の値）を正しく扱えないため．TEXTで保存し，読み出し時にint()で変換する方が安全．

3. **Q**: pickleでのシリアライズは安全か？
   **A**: 内部データのみで使用し，外部入力はパースしないため問題なし．numpy配列の効率的なシリアライズに最適．

## 承認

- **実装者**: Claude (AI Assistant)
- **レビュー**: dousu
- **承認日**: 2025-10-13
- **コミットハッシュ**: cac2c4f
