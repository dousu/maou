"""ローカルキャッシュの有効化判定がディレクトリの有無だけであることの回帰テスト．

`/audit-backlog` 2026-08-14 / backlog 行 O5(a)．

かつては ``--input-local-cache`` (bool flag) と ``--input-local-cache-dir``
(str) の 2 つがあり，**BigQuery だけが flag を見て S3/GCS は dir を見る**
という層をまたいだ食い違いがあった．``--input-s3`` に flag だけを渡すと
S3 の elif を全て外れ，入力ソース未指定の誤誘導エラーで停止していた．

2026-08-14 のユーザ判断で「dir に一本化」と決まったので，判定を
``local_cache_dir is not None`` の 1 箇所に寄せた．ここではその不変条件を
固定する — bool flag が復活したり，``use_local_cache`` が再び外部から
渡せるようになったら落ちる．

GCP 資格情報なしで走らせるため，``__init__`` は呼ばず署名だけを検査する．
"""

import inspect

from maou.infra.bigquery.bq_data_source import (
    BigQueryDataSource,
)

_CONSTRUCTORS = {
    "BigQueryDataSource": BigQueryDataSource,
    "PageManager": BigQueryDataSource.PageManager,
    "BigQueryDataSourceSpliter": (
        BigQueryDataSource.BigQueryDataSourceSpliter
    ),
}


def test_no_constructor_accepts_use_local_cache() -> None:
    """``use_local_cache`` を外から渡せないこと．

    渡せる限り「dir はあるが flag が False」という状態が作れてしまい，
    判定が 2 箇所に戻る．
    """
    for name, cls in _CONSTRUCTORS.items():
        params = inspect.signature(cls.__init__).parameters
        assert "use_local_cache" not in params, (
            f"{name}.__init__ が use_local_cache を受け付けています"
        )


def test_every_constructor_accepts_local_cache_dir() -> None:
    """``local_cache_dir`` は 3 つの構築経路すべてに残っていること．"""
    for name, cls in _CONSTRUCTORS.items():
        params = inspect.signature(cls.__init__).parameters
        assert "local_cache_dir" in params, (
            f"{name}.__init__ に local_cache_dir がありません"
        )


def test_flag_is_derived_from_the_directory() -> None:
    """``use_local_cache`` が dir の有無から導出されていること．

    属性そのものは ``PageManager`` の内部で今も使われている
    (LRU の要否，``get_page`` の分岐)．消したのは**外から与える手段**
    なので，導出が 1 文で行われていることを固定する．
    """
    source = inspect.getsource(BigQueryDataSource.PageManager)
    assert (
        "self.use_local_cache = local_cache_dir is not None"
        in source
    ), (
        "use_local_cache が local_cache_dir から導出されていません"
    )
