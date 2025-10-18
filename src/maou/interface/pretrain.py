from pathlib import Path
from typing import Optional

from maou.app.learning.masked_autoencoder import (
    MaskedAutoencoderPretraining,
)

__all__ = ["pretrain"]


def pretrain(
    *,
    input_dir: Optional[Path],
    config_path: Optional[Path],
) -> str:
    """Execute the masked autoencoder pretraining workflow.

    Args:
        input_dir: Optional path to the training data source.
        config_path: Optional configuration file path.

    Returns:
        Placeholder message describing the stub status.
    """
    options = MaskedAutoencoderPretraining.Options(
        input_dir=input_dir,
        config_path=config_path,
    )
    return MaskedAutoencoderPretraining().run(options)
