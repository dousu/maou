"""``enable_bundling`` / ``bundle_size_gb`` が不活性であることを固定する．

`audits/coverage.md` § Out-of-scope backlog の **O5(c)** 行に対する
characterization test．

この 2 つのノブは CLI から受理され，`DataSourceSpliter` → `PageManager`
→ `ObjectStorageDataSource` と受け渡されるだけで，値を判定に使う場所が
リポジトリ全体に存在しない (`.feather` は常に個別保存される)．

そのため **既定値を変えても挙動は変わらない**．本テストはその前提
そのものを固定する:

1. 3 層の既定値が一致していること (2026-08-13 以前は `DataSourceSpliter`
   だけ ``True`` で他の 2 つと食い違っていた)．
2. ノブが実際に読まれていないこと — キーワード引数としての素通し以外に
   ``enable_bundling`` を Load するコードが無いこと．

2 が破れたとき (= 誰かが bundling を実装したとき) は，1 の「既定値を
揃えても安全」という論拠が消える．そのときは docs 3 本 +
`docs/rust-backend.md` の「受理するが効果なし」という記述と，O5 行の
判断を作り直すこと．
"""

import ast
import inspect
from pathlib import Path

import pytest

from maou.infra.object_storage.data_source import (
    ObjectStorageDataSource,
)

_KNOBS = ("enable_bundling", "bundle_size_gb")

_LAYERS = {
    "DataSourceSpliter": (
        ObjectStorageDataSource.DataSourceSpliter.__init__
    ),
    "PageManager": ObjectStorageDataSource.PageManager.__init__,
    "ObjectStorageDataSource": ObjectStorageDataSource.__init__,
}


@pytest.mark.parametrize("knob", _KNOBS)
def test_defaults_agree_across_layers(knob: str) -> None:
    """3 層の既定値が一致する."""
    defaults = {
        layer: inspect.signature(func).parameters[knob].default
        for layer, func in _LAYERS.items()
    }
    distinct = set(defaults.values())
    assert len(distinct) == 1, (
        f"{knob} の既定値が層をまたいで食い違っている: {defaults}．"
        "受理するが効果のないノブなので，どの層から構築しても"
        "同じ値が既定になるべき．"
    )


def test_enable_bundling_default_is_false() -> None:
    """不活性なノブの既定は CLI と同じ ``False`` に揃える."""
    for layer, func in _LAYERS.items():
        default = (
            inspect.signature(func)
            .parameters["enable_bundling"]
            .default
        )
        assert default is False, (
            f"{layer}.__init__ の enable_bundling 既定が {default}．"
            "CLI (`--input-enable-bundling`) の既定は False であり，"
            "この機能は未実装なので False に揃える．"
        )


def _loads_outside_keyword_passthrough(
    source: str, name: str
) -> list[int]:
    """``name`` を Load する箇所のうち，素通し以外の行番号を返す.

    ``f(enable_bundling=enable_bundling)`` のようなキーワード引数への
    受け渡しは「読んでいる」とはみなさない — 値を判定にも保存にも
    使っていないため．
    """
    tree = ast.parse(source)

    passthrough: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            value = keyword.value
            if isinstance(value, ast.Name) and value.id == name:
                passthrough.add(id(value))

    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == name
        and isinstance(node.ctx, ast.Load)
        and id(node) not in passthrough
    ]


@pytest.mark.parametrize("knob", _KNOBS)
def test_knob_is_never_read(knob: str) -> None:
    """ノブは素通しされるだけで，判定にも保存にも使われない."""
    module_path = Path(inspect.getfile(ObjectStorageDataSource))
    source = module_path.read_text(encoding="utf-8")

    offending = _loads_outside_keyword_passthrough(source, knob)

    assert offending == [], (
        f"{module_path.name} の行 {offending} で {knob} が"
        "読まれている．このノブが実際に効くようになったのなら，"
        "docs/commands/pre_process.md ほか 3 本と "
        "docs/rust-backend.md の「受理するが効果なし」という記述，"
        "および audits/coverage.md の O5 行を作り直すこと．"
    )
