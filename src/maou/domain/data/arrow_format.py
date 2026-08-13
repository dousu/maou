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
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


# Arrow IPC File形式のマジックバイト (先頭8バイト)
ARROW_FILE_MAGIC = b"ARROW1\x00\x00"

# 行数スキャンを並列化する下限ファイル数．これ未満はスレッド生成のほうが
# 高くつく．
PARALLEL_SCAN_MIN_FILES = 8

# 行数スキャンの既定スレッド数．メタデータ読みは I/O 待ちが支配的なので
# CPU 数ではなく同時ラウンドトリップ数として選ぶ．
DEFAULT_SCAN_WORKERS = 8


@contextmanager
def feather_read_errors_name_the_file(
    file_path: Path,
) -> Iterator[None]:
    """`.feather` 読み込みの例外に，どのファイルかと推定原因を書き足す．

    **中途書き込みの `.feather` は拡張子で捕捉できない．**
    中断した ``save_*_df`` や in-place な ``rsync`` / ``gsutil cp`` が
    残す不完全なファイルは拡張子が `.feather` のままなので，
    :func:`~maou.infra.file_system.path_utils._is_temp_artifact` の
    ような末尾拡張子フィルタは原理的に素通しする．

    ただし**素通しした先で黙って壊れたデータを読むわけではない** —
    実際には読み込み時点で確実に落ちる．問題は落ち方で，polars も
    Rust バックエンドも**ファイル名を含まない**メッセージを出す:

    - 途中書き (File 形式のまま footer が欠ける):
      ``ComputeError: out-of-spec: InvalidFooter``
    - 0 バイト: ``OSError: failed to fill whole buffer``

    数百ファイルを渡す運用ではこれだけでは犯人を特定できないので，
    例外に :meth:`BaseException.add_note` で注記を足す．

    ``add_note`` を使うのは **例外の型も同一性も変えない**ため．
    ``OSError`` を捕まえている呼び出し側 (:func:`scan_row_counts` の
    「部分結果を返さない」保証に依存するテストなど) はそのまま動く．

    Args:
        file_path: 読もうとしている `.feather` のパス

    Yields:
        なし (with 節の中で読み込みを行う)
    """
    try:
        yield
    except Exception as e:
        note = f"reading feather file: {file_path}"
        try:
            size = file_path.stat().st_size
        except OSError:
            pass
        else:
            note += f" ({size} bytes)"
            if size == 0:
                note += (
                    " — the file is empty; it is most likely an "
                    "interrupted write, not valid Arrow IPC"
                )
            else:
                note += (
                    " — if this is a footer/spec error the file is "
                    "most likely incomplete (an interrupted "
                    "save_*_df / rsync / gsutil cp leaves a "
                    "truncated .feather that no extension filter "
                    "can catch)"
                )
        e.add_note(note)
        raise


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
    count = _scan_row_count_if_file_format(file_path)
    if count is not None:
        return count
    return _scan_row_count_stream(file_path)


def _scan_row_count_if_file_format(
    file_path: Path,
) -> int | None:
    """File 形式ならメタデータから行数を返し，Stream 形式なら ``None``．

    メタデータ読みだけで完結するので安全に並列化できる (:func:`scan_row_counts`)．

    Args:
        file_path: featherファイルのパス

    Returns:
        File形式なら行数，Stream形式なら ``None``
    """
    with feather_read_errors_name_the_file(file_path):
        if not is_arrow_ipc_file_format(file_path):
            return None
        lf = pl.scan_ipc(file_path)
        return int(lf.select(pl.len()).collect().item())


def _scan_row_count_stream(file_path: Path) -> int:
    """Stream 形式のファイルの行数を全読みで得る．

    Stream形式ではメタデータのみの読み出しが不可能なため，全データを
    読む必要がある．**この関数は並列に呼ばないこと** — ピークメモリが
    同時実行数倍になる．

    Args:
        file_path: featherファイルのパス

    Returns:
        ファイル内の行数
    """
    logger.info(
        "File %s is Arrow IPC Stream format. "
        "Reading full data for row count "
        "(consider converting to File format).",
        file_path,
    )
    with feather_read_errors_name_the_file(file_path):
        df = pl.read_ipc_stream(file_path)
        row_count = df.height
        del df
        return row_count


def scan_row_counts(
    file_paths: Iterable[Path],
    *,
    max_workers: int | None = None,
) -> list[int]:
    """複数のfeatherファイルの行数をまとめて取得する．

    :func:`scan_row_count` をファイル順に適用するだけだが，
    **途中で例外が出たときに部分結果を返さない**ことを保証する．
    呼び出し側がこのループを自前で書くと，「例外で打ち切られた
    カウントを memo として保持してしまい，以降 total_rows が
    過少申告される」という壊れ方を各々が作り込むことになる
    (実際 ``StreamingFileSource`` と ``StreamingHcpeDataSource`` は
    別実装で，安全なのは構造が違う結果の偶然にすぎなかった)．

    File 形式のファイルはメタデータだけ読めば行数が分かるので，
    **その部分だけをスレッドプールで並列に**引く．ネットワーク越しの
    ストレージでは 1 ファイルあたりのラウンドトリップが支配的で，
    数百ファイルの逐次スキャンが起動レイテンシの支配項になる．

    Stream 形式のファイルは行数を得るのに**全データを読む**必要がある
    ため，並列化するとピークメモリがワーカー数倍になる．そちらは
    逐次のまま残す (元々「File 形式に変換せよ」と警告する経路である)．

    Args:
        file_paths: featherファイルパスの列
        max_workers: File 形式のメタデータ読みに使うスレッド数．
            ``None`` なら ``DEFAULT_SCAN_WORKERS`` とファイル数の小さい方．

    Returns:
        入力と同じ順のファイルごとの行数リスト

    Raises:
        OSError: ファイルを読めないとき
        polars.exceptions.PolarsError: featherとして解釈できないとき
    """
    paths = list(file_paths)
    if len(paths) < PARALLEL_SCAN_MIN_FILES:
        # スレッド生成のほうが高くつく規模
        return [scan_row_count(fp) for fp in paths]

    workers = max_workers or min(
        DEFAULT_SCAN_WORKERS, len(paths)
    )

    # 例外は「部分結果を返さない」ため，全件そろってから返す．
    # ``ThreadPoolExecutor.map`` は消費時に最初の例外を再送出するので，
    # with 節の中で list 化しきる．
    with ThreadPoolExecutor(max_workers=workers) as executor:
        probed = list(
            executor.map(_scan_row_count_if_file_format, paths)
        )

    # Stream 形式 (``None``) だけを逐次で埋める
    return [
        count
        if count is not None
        else _scan_row_count_stream(path)
        for count, path in zip(probed, paths)
    ]
