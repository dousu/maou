from pathlib import Path

from click.testing import CliRunner

from maou.infra.console.pretrain_cli import pretrain


def test_pretrain_cli(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        pretrain,
        [
            "--input-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    # App layer implementation is still a stub, so the CLI should surface the placeholder message.
    assert "not implemented" in result.output.lower()
