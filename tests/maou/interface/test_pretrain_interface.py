from pathlib import Path

from maou.interface.pretrain import pretrain


def test_pretrain_returns_placeholder(
    tmp_path: Path,
) -> None:
    result = pretrain(
        input_dir=tmp_path,
        config_path=None,
    )

    assert isinstance(result, str)
    # App layer pretraining logic remains unimplemented for now, so the stub should emit this marker.
    assert "not implemented" in result.lower()
