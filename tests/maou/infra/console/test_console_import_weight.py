"""軽量コマンドが重い依存を import しないことの回帰テスト．

USI エンジンは GUI に登録されて**対局のたびに起動**されるため，起動時の
import コストがそのまま体感になる (実測: 学習/クラウド依存を巻き込むと
usiok まで数秒〜十数秒)．また base install (torch なし) でも動く必要がある
コマンドが，infra 経由で torch を module-level に引き込む回帰は過去にも
起きている．

判定は subprocess で行う (pytest 本体は既に torch を読んでいるため)．
"""

import subprocess
import sys

_PROBE = (
    "import sys; import {module}; "
    "print(','.join(n for n in ('torch', 'google.cloud.bigquery', "
    "'boto3') if n in sys.modules))"
)


def _heavy_modules_after_import(module: str) -> list[str]:
    """`module` を import した直後に読み込まれている重い依存を返す．"""
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    return [
        name for name in proc.stdout.strip().split(",") if name
    ]


def test_usi_console_does_not_import_training_deps() -> None:
    """`maou usi` / `maou-usi` の起動経路が torch/クラウド SDK を読まない．"""
    assert (
        _heavy_modules_after_import("maou.infra.console.usi")
        == []
    )


def test_selfplay_console_does_not_import_training_deps() -> (
    None
):
    """自己対局コマンドも同様 (評価は Rust 側で完結する)．"""
    assert (
        _heavy_modules_after_import(
            "maou.infra.console.selfplay"
        )
        == []
    )


def test_cloud_probes_resolve_on_access() -> None:
    """遅延化した後もクラウド probe は参照時に解決される．

    `HAS_*` を先に読まずクラス名を直接 import しても実体が返ること
    (module global を残すと `__getattr__` が呼ばれず None が漏れる)．
    """
    from maou.infra.console import common

    # 依存が入っていない環境では False/None が正 (どちらでも矛盾しないこと)
    has_bq = common.HAS_BIGQUERY
    assert isinstance(has_bq, bool)
    assert (common.BigQueryDataSource is not None) == has_bq
