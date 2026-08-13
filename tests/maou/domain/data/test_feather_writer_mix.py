"""writer の違う ``.feather`` を混ぜてもマージできることの回帰テスト．

`/audit-backlog` 2026-08-13 / backlog 行 N-1．

polars 1.38 は ``Binary`` 列を **BinaryView** で書き，``maou_io`` の
``save_feather`` (arrow-rs) は **LargeBinary** で書く．中身は同じでも
Arrow の型としては別物なので，混ざると ``concat_batches`` が

    It is not possible to concatenate arrays of different data types
    (BinaryView, LargeBinary)

で落ちていた．``pre-process --input-split-rows`` は入力を chunk する
ので，writer の違う ``.feather`` が同じ入力ディレクトリに混在すると
**マージが停止する**．File / Stream 形式の違いとは無関係で，polars 書きの
File 形式同士でも Rust 書きと混ぜれば落ちることを実測で確認してある．
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow.feather as pf
import pytest

from maou.domain.data.rust_io import (
    merge_hcpe_feather_files,
    save_hcpe_df,
)
from maou.domain.data.schema import create_empty_hcpe_df


def _polars_written(path: Path, rows: int) -> Path:
    """polars が書いた ``.feather`` (Binary 列は BinaryView)."""
    create_empty_hcpe_df(rows).write_ipc(
        path, compression="lz4"
    )
    return path


def _rust_written(path: Path, rows: int) -> Path:
    """maou_io が書いた ``.feather`` (Binary 列は LargeBinary)."""
    save_hcpe_df(create_empty_hcpe_df(rows), path)
    return path


def _hcp_arrow_type(path: Path) -> str:
    return str(pf.read_table(path).schema.field("hcp").type)


def test_the_two_writers_really_differ(tmp_path: Path) -> None:
    """前提の確認: 2 つの writer が別の Arrow 型を書くこと.

    ここが一致するようになったら (polars 側の既定が変わるなど)，
    このファイルのテストは前提を失う．
    """
    polars_path = _polars_written(
        tmp_path / "polars.feather", 2
    )
    rust_path = _rust_written(tmp_path / "rust.feather", 2)

    assert _hcp_arrow_type(polars_path) == "binary_view"
    assert _hcp_arrow_type(rust_path) == "large_binary"


def test_merge_accepts_mixed_writers(tmp_path: Path) -> None:
    """polars 書きと Rust 書きを混ぜてもマージできること."""
    paths = [
        str(_polars_written(tmp_path / "polars.feather", 3)),
        str(_rust_written(tmp_path / "rust.feather", 2)),
    ]

    outputs = merge_hcpe_feather_files(
        paths, str(tmp_path / "out"), 100, "merged"
    )

    assert len(outputs) == 1
    assert pl.read_ipc(outputs[0]).height == 5
    assert (
        _hcp_arrow_type(Path(outputs[0])) == "large_binary"
    ), "寄せ先は maou_io の save_feather が書く非 view 側"


def test_merge_order_does_not_change_the_output_type(
    tmp_path: Path,
) -> None:
    """どちらを先に渡しても出力の型が同じこと."""
    polars_path = str(
        _polars_written(tmp_path / "polars.feather", 2)
    )
    rust_path = str(_rust_written(tmp_path / "rust.feather", 2))

    first = merge_hcpe_feather_files(
        [polars_path, rust_path],
        str(tmp_path / "out_a"),
        100,
        "a",
    )
    second = merge_hcpe_feather_files(
        [rust_path, polars_path],
        str(tmp_path / "out_b"),
        100,
        "b",
    )

    assert _hcp_arrow_type(Path(first[0])) == _hcp_arrow_type(
        Path(second[0])
    )


@pytest.mark.parametrize("writer", ["polars", "rust"])
def test_single_writer_output_type_is_unchanged(
    tmp_path: Path, writer: str
) -> None:
    """同じ writer だけの入力は従来どおりの型で出ること.

    **trap**: 常に正規化すると，これまで BinaryView のまま出ていた
    polars 書きだけの入力の出力が LargeBinary に変わってしまう．
    正規化は**スキーマが食い違うときだけ**走る．
    """
    write = (
        _polars_written if writer == "polars" else _rust_written
    )
    paths = [
        str(write(tmp_path / f"{writer}_{i}.feather", 2))
        for i in range(2)
    ]
    expected = _hcp_arrow_type(Path(paths[0]))

    outputs = merge_hcpe_feather_files(
        paths, str(tmp_path / "out"), 100, "same"
    )

    assert _hcp_arrow_type(Path(outputs[0])) == expected
