"""``BigQueryDataSource`` のローカルキャッシュ検証に関する回帰テスト．

`/audit-backlog` 2026-08-12 / backlog 行 O6．

`__download_all_to_local` の完了検証が `glob("*.npy")` だったのに対し，
`__save_to_local` は `.feather` を書くため，キャッシュが正常に作られても
常に "Created 0 local cache files" と警告が出ていた．

GCP 資格情報なしで走らせるため，``PageManager`` は ``object.__new__``
で生成し，検証が読む属性だけを差し込む (``__init__`` は BigQuery
クライアントを要求する)．
"""

import logging
from pathlib import Path
from typing import Any

import pytest

from maou.infra.bigquery.bq_data_source import (
    BigQueryDataSource,
)


def _make_page_manager(
    cache_dir: Path, *, total_pages: int
) -> Any:
    """BigQuery に触れずに ``PageManager`` の検証部分だけを組み立てる．"""
    pm = object.__new__(BigQueryDataSource.PageManager)
    pm.local_cache_dir = cache_dir
    pm.total_pages = total_pages
    pm.dataset_fqn = "proj.dataset"
    pm.table_name = "table"
    # __get_local_cache_path が読む name-mangled 属性
    pm._PageManager__pruning_info = []
    return pm


def test_local_cache_validation_counts_feather_files(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``__save_to_local`` が書く拡張子をキャッシュ検証が数えること．

    検証は「全ページのキャッシュが既にある」場合は早期 return で
    スキップされるので，1 ページ分をわざと欠けさせてダウンロード経路を
    通す．``__save_to_local`` は実際の書き込みだけスタブに差し替える．
    """
    pm = _make_page_manager(tmp_path, total_pages=2)

    paths = [
        pm._PageManager__get_local_cache_path(page_num)
        for page_num in range(2)
    ]
    assert all(path.suffix == ".feather" for path in paths)

    # page 0 だけ既に存在させ，page 1 をダウンロード経路に通す
    paths[0].write_bytes(b"")

    pm._PageManager__fetch_from_bigquery = lambda page_num: (
        page_num
    )
    pm._PageManager__save_to_local = lambda page_num, data: (
        paths[page_num].write_bytes(b"")
    )

    maou_logger = logging.getLogger("maou")
    original_propagate = maou_logger.propagate
    maou_logger.propagate = True
    try:
        with caplog.at_level(logging.INFO):
            pm._PageManager__download_all_to_local()
    finally:
        maou_logger.propagate = original_propagate

    assert "Created 2 local cache files" in caplog.text
    assert (
        "No local cache files were created" not in caplog.text
    )
