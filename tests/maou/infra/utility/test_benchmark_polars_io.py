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

from maou.infra.utility import benchmark_polars_io


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
