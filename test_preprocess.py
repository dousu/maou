from pathlib import Path

import numpy as np

# ==== 設定 ====
NUM_FILES = 10
DATA_PER_FILE = 1000
MAX_MOVE_IDS = 100

# ==== サンプルデータ作成 ====
source_dtype = np.dtype(
    [("hash", "U20"), ("win", np.bool_), ("move_id", np.int16)]
)
intermediate_dtype = np.dtype(
    [
        ("hash", "U20"),
        ("count", np.int16),
        ("win_count", np.int16),
        ("move_labels_count", np.int16, (MAX_MOVE_IDS,)),
    ]
)
target_dtype = np.dtype(
    [
        ("hash", "U20"),
        ("winrate", np.float16),
        ("move_labels", np.float16, (MAX_MOVE_IDS,)),
    ]
)
data_dir = Path("test_data")
np.random.seed(0)
data_dir.mkdir(exist_ok=True)


def random_hash(n, n_unique_hashes=10):
    return np.array(
        [
            f"hash_{i}"
            for i in np.random.randint(
                low=0, high=n_unique_hashes, size=n
            )
        ],
        dtype="U20",
    )


for i in range(NUM_FILES):
    pos_hashes = random_hash(DATA_PER_FILE)
    wins = np.random.choice([True, False], size=DATA_PER_FILE)
    move_ids = np.random.randint(
        0, MAX_MOVE_IDS, size=DATA_PER_FILE, dtype=np.int16
    )
    data = np.zeros(DATA_PER_FILE, dtype=source_dtype)
    data["hash"] = pos_hashes
    data["win"] = wins
    data["move_id"] = move_ids
    data.tofile(data_dir / f"file_{i}.npy")

intermediate_data = np.zeros(0, dtype=intermediate_dtype)


def process_file(file_path):
    data = np.memmap(file_path, dtype=source_dtype, mode="r")
    idx = np.argsort(data["hash"], kind="mergesort")
    sorted_hash = data["hash"][idx]
    # これをつかって1ファイルの局面数はだせているので利用する
    uniq, counts = np.unique(sorted_hash, return_counts=True)
    moves = []
    wins = []
    i = 0
    for c in counts:
        moves.append(
            np.bincount(
                data["move_id"][idx[i : i + c]],
                minlength=MAX_MOVE_IDS,
            )
        )
        wins.append(np.sum(data["win"][idx[i : i + c]]))
        i += c
    return uniq, counts, moves, wins


for i in range(NUM_FILES):
    file_path = data_dir / f"file_{i}.npy"
    uniq, counts, moves, wins = process_file(file_path)
    # hash値を持っているかを調べてなかったらレコード追加，あればマイグレーション
    for h, c, m, w in zip(uniq, counts, moves, wins):
        idx_array = np.where(intermediate_data["hash"] == h)[0]
        if len(idx_array) == 0:
            new_record = np.zeros(1, dtype=intermediate_dtype)
            new_record["hash"] = h
            new_record["count"] = c
            new_record["win_count"] = w
            new_record["move_labels_count"] = m
            intermediate_data = np.append(
                intermediate_data, new_record
            )
        else:
            idx = idx_array[0]

            intermediate_data["count"][idx] += c
            intermediate_data["win_count"][idx] += w
            intermediate_data["move_labels_count"][idx] += m

print(intermediate_data)

target_data = np.zeros(
    len(intermediate_data), dtype=target_dtype
)
target_data["hash"] = intermediate_data["hash"]
target_data["winrate"] = (
    intermediate_data["win_count"] / intermediate_data["count"]
)
target_data["move_labels"] = (
    intermediate_data["move_labels_count"]
    / intermediate_data["count"][:, np.newaxis]
)

print(target_data)
