from pathlib import Path
from typing import Optional

from maou.app.learning.dl import LearningDataSource
from maou.app.learning.masked_autoencoder import (
    MaskedAutoencoderPretraining,
)

__all__ = ["pretrain"]


def pretrain(
    *,
    datasource: LearningDataSource.DataSourceSpliter,
    config_path: Optional[Path],
    output_path: Optional[Path] = None,
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    mask_ratio: float = 0.75,
    device: Optional[str] = None,
    num_workers: int = 0,
    hidden_dim: int = 512,
) -> str:
    """Execute the masked autoencoder pretraining workflow.

    Args:
        datasource: Data source factory providing access to training data.
        config_path: Optional configuration file path.
        output_path: Optional location to store the resulting state_dict.
        epochs: Number of optimisation epochs.
        batch_size: Mini-batch size for training.
        learning_rate: Optimiser learning rate.
        mask_ratio: Fraction of feature elements that will be masked.
        device: Optional explicit device identifier (e.g. "cpu" or "cuda").
        num_workers: Number of DataLoader workers.
        hidden_dim: Hidden dimension size of the autoencoder MLP.

    Returns:
        A message describing where the trained state_dict was saved.
    """
    options = MaskedAutoencoderPretraining.Options(
        datasource=datasource,
        config_path=config_path,
        output_path=output_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mask_ratio=mask_ratio,
        device=device,
        num_workers=num_workers,
        hidden_dim=hidden_dim,
    )
    return MaskedAutoencoderPretraining().run(options)
