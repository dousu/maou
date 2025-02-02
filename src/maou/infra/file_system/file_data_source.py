from pathlib import Path
from typing import Any

import numpy as np

from maou.interface import learn


class FileDataSource(learn.DataSource):
    def __init__(self, *, file_paths: list[Path], schema: dict[str, str]) -> None:
        """ファイルシステムから複数のファイルに入っているデータを取り出す.

        Args:
            file_paths (list[Path]): npyファイルのリスト
            schema (dict[str, str]): 各フィールドのマッピング（例: {"hcp": "hcp", "eval": "eval"}）
        """
        self.file_paths = file_paths
        self.schema = schema

        self.file_row_offsets = []
        total_rows = 0
        for file in self.file_paths:
            data = np.load(file, mmap_mode="r")
            num_rows = data.shape[0]
            self.file_row_offsets.append((file, total_rows, num_rows))
            total_rows += num_rows

        self.total_rows = total_rows

    def __getitem__(self, idx: int) -> dict[str, Any]:
        for file, start_idx, num_rows in self.file_row_offsets:
            if start_idx <= idx < start_idx + num_rows:
                relative_idx = idx - start_idx

                # ここもメモリマップ使っているがファイルサイズはそれほどでもないので
                # パフォーマンス上のデメリットがあるならなくしてもいい
                data = np.load(file, mmap_mode="r")

                return {
                    key: data[column][relative_idx]
                    for key, column in self.schema.items()
                }

        raise IndexError(f"Index {idx} out of range.")

    def __len__(self) -> int:
        return self.total_rows
