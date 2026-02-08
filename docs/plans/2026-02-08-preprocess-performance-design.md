# Preprocess パイプライン パフォーマンス改善設計

## 背景

151ファイル・約6,000万レコードのHCPEデータをpreprocessに変換する処理で，intermediate store構築が6ファイルの処理に2時間以上かかる．ワーカー数は4，メモリは100GB．intermediate storeのデータ量は経験的に120GBを超える．

## ボトルネック分析

### P0: DuckDB行単位UPSERT（全体の70-80%）

`_flush_buffer()` が1レコードごとにSELECT→INSERT/UPDATEを実行．6ファイルで推定480万SQL実行．

- 場所: `src/maou/domain/data/intermediate_store.py` `_flush_buffer()`
- 規模: 6ファイル × ~400Kユニーク局面 = ~2.4Mレコード → ~4.8M SQL実行

### P0: メインプロセスでのシリアルmerge

ワーカー結果のmergeがメインプロセスでシリアル実行．merge中は他のワーカー結果を回収しない．

- 場所: `src/maou/app/pre_process/hcpe_transform.py` L610-690

### P1: 冗長な `set_hcp()` 呼び出し（4N+U回）

`Transform.board_move_label()` と `board_game_result()` が各自 `Board()` を生成し `set_hcp()` を呼ぶ．60Mレコード × 3回余分 = 180M回の不要なHuffmanデコード．

- 場所: `src/maou/app/pre_process/hcpe_transform.py` `_process_single_array()`
- 場所: `src/maou/app/pre_process/transform.py` 各staticメソッド

### P1: `np.vectorize` による偽ベクトル化

`make_board_id_positions()` で `np.vectorize` を使用（実質Pythonループ）．81マスのdict lookup + Polars DataFrame往復変換．

- 場所: `src/maou/app/pre_process/feature.py` `make_board_id_positions()`

## 設計

### 1. DuckDB Intermediate Store の最適化

#### 変更対象

`src/maou/domain/data/intermediate_store.py`

#### 1.1 Rust UDFの登録

`IntermediateDataStore.__init__` でDuckDB接続作成時に，`add_sparse_arrays_rust` をラップした2つのスカラーUDFを登録する:

- `merge_sparse_indices(existing_indices, existing_values, new_indices, new_values)` → マージ後のindices
- `merge_sparse_values(existing_indices, existing_values, new_indices, new_values)` → マージ後のvalues

```python
conn.create_function("merge_sparse_indices",
    lambda ei, ev, ni, nv: add_sparse_arrays_rust(ei, ev, ni, nv)[0],
    [LIST(USMALLINT), LIST(INT), LIST(USMALLINT), LIST(INT)], LIST(USMALLINT))

conn.create_function("merge_sparse_values",
    lambda ei, ev, ni, nv: add_sparse_arrays_rust(ei, ev, ni, nv)[1],
    [LIST(USMALLINT), LIST(INT), LIST(USMALLINT), LIST(INT)], LIST(INT))
```

#### 1.2 `bulk_upsert()` の新設

`add_dataframe_batch()` の行単位 `iter_rows()` → `_batch_buffer` → `_flush_buffer()` パターンを廃止．ワーカーから受け取ったPolars DataFrameをDuckDBに一時テーブルとして登録し，1回の `INSERT ... ON CONFLICT DO UPDATE` で全レコードをUPSERTする:

```sql
INSERT INTO intermediate_data
SELECT hash_id, count, win_count,
       compress_sparse_indices(move_label_count),
       compress_sparse_values(move_label_count),
       board_id_positions, pieces_in_hand
FROM batch_df
ON CONFLICT (hash_id) DO UPDATE SET
    count = intermediate_data.count + excluded.count,
    win_count = intermediate_data.win_count + excluded.win_count,
    move_label_indices = merge_sparse_indices(
        intermediate_data.move_label_indices, intermediate_data.move_label_values,
        excluded.move_label_indices, excluded.move_label_values),
    move_label_values = merge_sparse_values(
        intermediate_data.move_label_indices, intermediate_data.move_label_values,
        excluded.move_label_indices, excluded.move_label_values)
```

#### 1.3 `_batch_buffer` と `_flush_buffer()` の廃止

バッファリングロジックはDuckDB側に委譲されるため，Pythonレベルのバッファは不要になる．

### 2. Transform処理の1パス化

#### 変更対象

- `src/maou/app/pre_process/transform.py` — 新staticメソッド追加
- `src/maou/app/pre_process/hcpe_transform.py` — `_process_single_array()` のループ統合
- `src/maou/app/pre_process/feature.py` — `make_board_id_positions()` のlookup table化

#### 2.1 `Transform` にBoard受け取り版メソッドを追加

既存のstaticメソッドは維持し，構築済みBoardインスタンスを受け取る新しいメソッドを追加:

```python
@staticmethod
def board_move_label_from_board(board: shogi.Board, move16: int) -> int:
    """構築済みBoardからmove labelを計算（set_hcpスキップ）．"""
    return make_move_label(board.get_turn(), move16)

@staticmethod
def board_game_result_from_board(board: shogi.Board, game_result: int) -> float:
    """構築済みBoardからgame resultを計算（set_hcpスキップ）．"""
    return make_result_value(board.get_turn(), game_result)

@staticmethod
def board_feature_from_board(board: shogi.Board) -> Tuple[List, List]:
    """構築済みBoardからfeatureを計算（set_hcpスキップ）．"""
    board_id_positions = make_board_id_positions_fast(board)
    pieces_in_hand = make_pieces_in_hand(board)
    return board_id_positions, pieces_in_hand
```

#### 2.2 `_process_single_array()` を1+1ループに統合

3つのループ(4N+U回のset_hcp)を2つのループ(N+U回)に統合:

```python
# 統合ループ: hash + move_label + game_result を1パスで (N回のset_hcp)
for i in range(n):
    board.set_hcp(data["hcp"][i])
    hashs[i] = board.hash()
    move_labels[i] = Transform.board_move_label_from_board(board, data["bestMove16"][i])
    wins[i] = Transform.board_game_result_from_board(board, data["gameResult"][i])

# ユニーク局面ループ: board_feature のみ (U回のset_hcp)
for u_idx, hash_val in enumerate(uniq_hash):
    board.set_hcp(data["hcp"][idx[start_idx]])
    board_id_positions, pieces_in_hand = Transform.board_feature_from_board(board)
```

#### 2.3 `make_board_id_positions_fast()` — numpy lookup table版

`np.vectorize` + dict lookup + Polars DataFrame往復を廃止．モジュールレベルでlookup tableを事前生成:

```python
_PIECE_ID_TABLE = np.zeros(MAX_CSHOGI_PIECE_ID + 1, dtype=np.uint8)
for cshogi_id, maou_id in Board._CSHOGI_TO_PIECE_ID.items():
    _PIECE_ID_TABLE[cshogi_id] = maou_id

def make_board_id_positions_fast(board: shogi.Board) -> List[List[int]]:
    """numpy lookup tableで81マスを一括変換．"""
    raw_board = np.array(board.pieces, dtype=np.int32)
    converted = _PIECE_ID_TABLE[raw_board]
    return converted.reshape(9, 9).tolist()
```

### 3. ワーカーオーケストレーションの非同期化

#### 変更対象

`src/maou/app/pre_process/hcpe_transform.py` — `transform()` のワーカー管理部分

#### 3.1 merge専用スレッドとキューの導入

```python
merge_queue: queue.Queue = queue.Queue(maxsize=max_workers)
merge_error: List[Exception] = []

def merge_worker(q, store):
    while True:
        item = q.get()
        if item is None:
            break
        try:
            store.bulk_upsert(item)
        except Exception as e:
            merge_error.append(e)
        finally:
            q.task_done()

merge_thread = threading.Thread(target=merge_worker, args=(merge_queue, self._intermediate_store))
merge_thread.start()
```

`maxsize=max_workers` でバックプレッシャーを実現．キューには最大4件(ワーカー数分)しか溜まらない．

#### 3.2 `as_completed()` ループの修正

現在の `break` による1件ずつ処理を廃止．完了したfutureの結果をキューに投入し，即座に次のワーカー結果を回収:

```python
for future in as_completed(futures):
    dataname = futures.pop(future)
    try:
        batch_result = future.result()
        merge_queue.put(batch_result)
    except Exception as exc:
        logger.error(f"{dataname} failed: {exc}")

    try:
        new_dataname, new_data = next(data_iterator)
        new_future = executor.submit(self._process_single_array, new_data)
        futures[new_future] = new_dataname
    except StopIteration:
        pass

    if merge_error:
        raise merge_error[0]

    pbar.update(1)
```

#### 3.3 終了処理

```python
merge_queue.put(None)  # 終了シグナル
merge_thread.join()
```

### 4. Finalization処理の最適化

#### 変更対象

`src/maou/domain/data/intermediate_store.py` — `finalize_to_dataframe()` / `iter_finalize_chunks_df()`

#### 4.1 DuckDBバルク読み出し + Polars直接変換

行単位の `fetchall()` → Pythonループを廃止．DuckDBの `.pl()` でPolars DataFrameを直接取得:

```python
def iter_finalize_chunks_df(self, chunk_size=1_000_000):
    offset = 0
    while True:
        df = self._conn.execute(
            "SELECT * FROM intermediate_data ORDER BY hash_id LIMIT ? OFFSET ?",
            (chunk_size, offset)
        ).pl()

        if df.is_empty():
            break

        df = df.with_columns([
            expand_and_normalize_move_labels(
                pl.col("move_label_indices"),
                pl.col("move_label_values"),
                pl.col("count")
            ).alias("moveLabel"),
            (pl.col("win_count") / pl.col("count")).alias("resultValue"),
        ])

        yield df.select(["hash_id", "boardIdPositions", "piecesInHand", "moveLabel", "resultValue"])
        offset += chunk_size
```

#### 4.2 sparse展開のバッチ化

per-rowの `expand_sparse_array_rust()` をカラム単位で処理する Rust関数を新設，または `pl.Expr.map_batches` で既存関数をバッチ適用．

## 推定改善効果

| 施策 | 対象部分の短縮率 | 難易度 |
|------|----------------|--------|
| DuckDB行単位UPSERT → バルクUPSERT | 70-80% | 中 |
| set_hcp 4N+U → N+U（1パス化） | 60-75% (transform部分) | 低 |
| merge専用スレッド | ワーカー遊休排除 | 低 |
| np.vectorize → numpy lookup table | 5-10% | 低 |
| finalization最適化 | finalization時間の大幅短縮 | 中 |

全体の推定効果: 6ファイル2時間 → 15-20分．151ファイル全体で ~1-2時間．
