"""``benchmark_polars_io`` が最後まで走ることの回帰テスト．

`/audit-backlog` 2026-08-13 / backlog 行 N-2．

``docs/performance.md`` が案内する
``python -m maou.infra.utility.benchmark_polars_io`` は，
``benchmark_datasource_iteration`` が ``.npy`` パスを ``FileDataSource``
に渡していたため ``ValueError: Only .feather files are supported`` で
**必ず落ちていた** (``FileManager.__init__`` の拡張子ガード)．
``.npy`` 比較を削除して経路を 1 本にしたので，
案内どおりのコマンドが完走することを固定する．
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from maou.infra.utility import benchmark_polars_io
from maou.interface.data_schema import (
    get_hcpe_dtype,
    get_hcpe_polars_schema,
    get_preprocessing_dtype,
    get_preprocessing_polars_schema,
)


def test_main_runs_to_completion(
    tmp_path: Path, capsys
) -> None:
    """documented command 相当が例外なく完走すること."""
    benchmark_polars_io.main(output_dir=tmp_path, num_records=8)

    captured = capsys.readouterr()
    assert "PERFORMANCE BENCHMARK SUMMARY" in captured.out
    assert "DataSource Iteration" in captured.out


def test_no_npy_artifacts_are_written(tmp_path: Path) -> None:
    """``.npy`` を一切書かないこと (削除の実効性)."""
    benchmark_polars_io.main(output_dir=tmp_path, num_records=8)

    assert list(tmp_path.glob("*.npy")) == []
    assert list(tmp_path.glob("*.feather")) != []


def test_npy_helpers_are_gone() -> None:
    """``.npy`` I/O ヘルパが残っていないこと.

    残っていると「使う予定のない形式をコードだけが知っている」状態に
    戻る (2026-08-12 の判断: `.npy` はもう使わない)．
    """
    for name in (
        "save_hcpe_array",
        "load_hcpe_array",
        "save_preprocessing_array",
        "load_preprocessing_array",
    ):
        assert not hasattr(benchmark_polars_io, name), name


# ============================================================================
# /audit-backlog 2026-08-13 — backlog 行 N5-1 の回帰テスト
# ============================================================================


class TestSchemaDerivedTestData:
    """ベンチのテストデータをスキーマから導出した変更の回帰テスト．

    以前は列名も長さ (9x9 / 14 / MOVE_LABELS_NUM / 32 バイト) も
    ハードコードした dict で，スキーマが変わると polars 内部の
    ``KeyError`` になるだけだった (backlog 行 N5-1)．
    """

    def test_hcpe_data_matches_schema_exactly(self) -> None:
        """生成した DataFrame が polars スキーマと完全一致すること."""
        bench = benchmark_polars_io.PerformanceBenchmark(
            num_records=4
        )
        df = bench._create_hcpe_test_data_polars()

        assert dict(df.schema) == get_hcpe_polars_schema()
        assert df.height == 4

    def test_preprocessing_data_matches_schema_exactly(
        self,
    ) -> None:
        bench = benchmark_polars_io.PerformanceBenchmark(
            num_records=4
        )
        df = bench._create_preprocessing_test_data_polars()

        assert (
            dict(df.schema) == get_preprocessing_polars_schema()
        )
        assert df.height == 4

    def test_list_lengths_come_from_structured_dtype(
        self,
    ) -> None:
        """List 列の要素数が structured dtype の宣言と一致すること.

        これが「スキーマ追従」の実体．長さのハードコードが復活すると
        dtype 側を変えてもここが追従せず落ちる．
        """
        bench = benchmark_polars_io.PerformanceBenchmark(
            num_records=3
        )
        df = bench._create_preprocessing_test_data_polars()
        fields = get_preprocessing_dtype().fields
        assert fields is not None

        row = df.row(0, named=True)
        assert (
            len(row["piecesInHand"])
            == (fields["piecesInHand"][0].shape[0])
        )
        assert (
            len(row["moveLabel"])
            == (fields["moveLabel"][0].shape[0])
        )
        assert (
            len(row["moveWinRate"])
            == (fields["moveWinRate"][0].shape[0])
        )
        board_shape = fields["boardIdPositions"][0].shape
        assert len(row["boardIdPositions"]) == board_shape[0]
        assert len(row["boardIdPositions"][0]) == board_shape[1]

        hcpe_fields = get_hcpe_dtype().fields
        assert hcpe_fields is not None
        hcpe_row = bench._create_hcpe_test_data_polars().row(
            0, named=True
        )
        assert (
            len(hcpe_row["hcp"])
            == hcpe_fields["hcp"][0].shape[0]
        )
        assert (
            len(hcpe_row["ratings"])
            == hcpe_fields["ratings"][0].shape[0]
        )

    def test_strings_fit_the_declared_width(self) -> None:
        """文字列列が structured dtype の宣言幅を超えないこと.

        超えると structured array へ戻すときに黙って切り詰められる．
        """
        bench = benchmark_polars_io.PerformanceBenchmark(
            num_records=4
        )
        df = bench._create_hcpe_test_data_polars()
        fields = get_hcpe_dtype().fields
        assert fields is not None

        for column in ("id", "endgameStatus"):
            width = fields[column][0].itemsize // 4
            assert all(
                len(v) <= width for v in df[column].to_list()
            ), column

    def test_new_schema_column_is_named_in_the_error(
        self,
    ) -> None:
        """スキーマに長さ不明の列が増えたら列名を名指しして落ちること.

        trap: 素朴な実装は polars の ``KeyError`` や無言の欠落になり，
        どこを直せばよいかが分からない — それが N5-1 の中身だった．
        """
        schema = dict(get_preprocessing_polars_schema())
        schema["newVectorColumn"] = pl.List(pl.Float32)

        with pytest.raises(ValueError, match="newVectorColumn"):
            benchmark_polars_io._create_test_data(
                schema,
                get_preprocessing_dtype(),
                2,
                "preprocessing",
            )

    def test_unsupported_dtype_is_named_in_the_error(
        self,
    ) -> None:
        schema = {"weird": pl.Object()}

        with pytest.raises(ValueError, match="weird"):
            benchmark_polars_io._create_test_data(
                schema,
                get_preprocessing_dtype(),
                2,
                "preprocessing",
            )
