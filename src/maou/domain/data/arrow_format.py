"""Arrow IPC File/Stream 形式の判定．

`.feather` として書かれるファイルには Arrow IPC の 2 形式がある．

- **File 形式**: 先頭 8 バイトが ``ARROW1\\x00\\x00``．末尾に footer を
  持つのでメタデータだけを読んで行数を得られる．本リポジトリが書くのは
  常にこちら．
- **Stream 形式**: footer が無く，行数はデータを読まないと分からない．
  過去に書かれたファイルや外部ツール由来の入力で現れる．

判定と行数取得は `infra` (ローカルファイル) と `interface` (入力ファイルの
サイズ調整) の双方から要る．`interface` は `infra` に依存できないため，
両者が依存できる最下層の `domain` に置く．
"""

import logging
from collections.abc import Iterable
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


# Arrow IPC File形式のマジックバイト (先頭8バイト)
ARROW_FILE_MAGIC = b"ARROW1\x00\x00"


def is_arrow_ipc_file_bytes(data: bytes) -> bool:
    """バイト列がArrow IPC File形式かどうかを判定する．

    Args:
        data: 判定するバイト列 (先頭8バイトのみ参照する)

    Returns:
        Arrow IPC File形式ならTrue，Stream形式ならFalse
    """
    return data[:8] == ARROW_FILE_MAGIC


def is_arrow_ipc_file_format(file_path: Path) -> bool:
    """ファイルがArrow IPC File形式かどうかを判定する．

    先頭8バイトのマジックバイトで判定する．
    Stream形式の場合はFalseを返す．

    Args:
        file_path: 判定するファイルのパス

    Returns:
        Arrow IPC File形式ならTrue，Stream形式ならFalse
    """
    with open(file_path, "rb") as f:
        header = f.read(8)
    return is_arrow_ipc_file_bytes(header)


def scan_row_count(file_path: Path) -> int:
    """featherファイルの行数のみを取得する．

    Arrow IPC File形式の場合はメタデータから高速に取得する．
    Stream形式の場合は ``pl.read_ipc_stream`` でDataFrameの行数を取得する．

    Args:
        file_path: featherファイルのパス

    Returns:
        ファイル内の行数
    """
    if is_arrow_ipc_file_format(file_path):
        # File形式: メタデータのみ読み(高速)
        lf = pl.scan_ipc(file_path)
        return lf.select(pl.len()).collect().item()
    else:
        # Stream形式: DataFrameの高さを取得
        # Note: Stream形式ではメタデータのみの読み出しが不可能なため，
        # 全データを読む必要がある．大規模ファイルではメモリ使用量に注意．
        logger.info(
            "File %s is Arrow IPC Stream format. "
            "Reading full data for row count "
            "(consider converting to File format).",
            file_path,
        )
        df = pl.read_ipc_stream(file_path)
        row_count = df.height
        del df
        return row_count


def scan_row_counts(
    file_paths: Iterable[Path],
) -> list[int]:
    """複数のfeatherファイルの行数をまとめて取得する．

    :func:`scan_row_count` をファイル順に適用するだけだが，
    **途中で例外が出たときに部分結果を返さない**ことを保証する．
    呼び出し側がこのループを自前で書くと，「例外で打ち切られた
    カウントを memo として保持してしまい，以降 total_rows が
    過少申告される」という壊れ方を各々が作り込むことになる
    (実際 ``StreamingFileSource`` と ``StreamingHcpeDataSource`` は
    別実装で，安全なのは構造が違う結果の偶然にすぎなかった)．

    Args:
        file_paths: featherファイルパスの列

    Returns:
        入力と同じ順のファイルごとの行数リスト

    Raises:
        OSError: ファイルを読めないとき
        polars.exceptions.PolarsError: featherとして解釈できないとき
    """
    return [scan_row_count(fp) for fp in file_paths]
