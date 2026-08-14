"""``enable_bundling`` / ``bundle_size_gb`` が存在しないことを固定する．

`audits/coverage.md` § Out-of-scope backlog の **O5(c)** 行に対する
回帰テスト．

この 2 つのノブは CLI から受理され，`DataSourceSpliter` → `PageManager`
→ `ObjectStorageDataSource` と受け渡されるだけで，値を判定に使う場所が
リポジトリ全体に存在しなかった (`.feather` は常に個別保存される)．

2026-08-14 のユーザ判断により **ノブごと削除**した (実装する / deprecation
警告を出して残す，はいずれも却下された)．前身の
`test_bundling_knobs_inert.py` は「受理はするが読まれない」ことを固定して
いたが，本テストはより強く **どの層にも存在しない**ことを固定する．

復活させる場合 (= 誰かが bundling を実装する場合) は，このテストを消す
だけでなく，`docs/commands/` 3 本の「このオプションは存在しない」という
記述と，O5 行の判断を作り直すこと．
"""

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

_CLI_OPTIONS = (
    "--input-enable-bundling",
    "--input-bundle-size-gb",
)

_CONSOLE_MODULES = ("pre_process.py", "utility.py")


@pytest.mark.parametrize("knob", _KNOBS)
def test_knob_is_absent_from_every_constructor(
    knob: str,
) -> None:
    """3 層のどのコンストラクタも受理しない."""
    accepting = [
        layer
        for layer, func in _LAYERS.items()
        if knob in inspect.signature(func).parameters
    ]

    assert accepting == [], (
        f"{accepting} が {knob} をまだ受理している．"
        "このノブは 2026-08-14 に削除されたので，復活させるなら"
        "実際に bundling を実装し，docs/commands/ 3 本と "
        "audits/coverage.md の O5 行を作り直すこと．"
    )


@pytest.mark.parametrize("knob", _KNOBS)
def test_knob_is_absent_from_the_module(knob: str) -> None:
    """モジュール本文にも痕跡が残っていない."""
    module_path = Path(inspect.getfile(ObjectStorageDataSource))
    source = module_path.read_text(encoding="utf-8")

    assert knob not in source, (
        f"{module_path.name} に {knob} が残っている．"
        "受け渡しだけの死んだノブが再び生えている可能性がある．"
    )


@pytest.mark.parametrize("option", _CLI_OPTIONS)
def test_cli_option_is_absent(option: str) -> None:
    """CLI からも公開されていない."""
    console_dir = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "maou"
        / "infra"
        / "console"
    )

    offending = [
        name
        for name in _CONSOLE_MODULES
        if option
        in (console_dir / name).read_text(encoding="utf-8")
    ]

    assert offending == [], (
        f"{offending} がまだ {option} を公開している．"
        "このオプションは 2026-08-14 に "
        "pre-process / benchmark-dataloader / benchmark-training の "
        "3 コマンドから削除された．"
    )
