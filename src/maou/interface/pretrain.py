from pathlib import Path
from typing import Literal, Optional

from maou.app.learning.dl import LearningDataSource
from maou.app.learning.masked_autoencoder import (
    MaskedAutoencoderPretraining,
)

__all__ = ["pretrain"]


def pretrain(
    *,
    datasource: LearningDataSource.DataSourceSpliter,
    datasource_type: Literal["preprocess", "hcpe"],
    config_path: Optional[Path],
    output_path: Optional[Path] = None,
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    mask_ratio: float = 0.75,
    device: Optional[str] = None,
    compilation: bool = False,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    hidden_dim: int = 512,
    forward_chunk_size: Optional[int] = None,
    cache_transforms: Optional[bool] = None,
) -> str:
    """Execute the masked autoencoder pretraining workflow.

    Args:
        datasource: Data source factory providing access to training data.
        datasource_type: Format of the datasource ('preprocess' or 'hcpe').
        config_path: Optional configuration file path.
        output_path: Optional location to store the resulting state_dict.
        epochs: Number of optimisation epochs.
        batch_size: Mini-batch size for training.
        learning_rate: Optimiser learning rate.
        mask_ratio: Fraction of feature elements that will be masked.
        device: Optional explicit device identifier (e.g. "cpu" or "cuda").
        compilation: Enable PyTorch compilation for the autoencoder model.
        num_workers: Number of DataLoader workers. Defaults to 0 when omitted.
        pin_memory: Explicit pin_memory setting. Defaults based on device when None.
        prefetch_factor: Prefetch batches per worker. Defaults to 2 when omitted.
        hidden_dim: Hidden dimension size of the autoencoder MLP.
        forward_chunk_size: Optional limit for forward micro-batch size.
        cache_transforms: Enable in-memory caching of extracted feature tensors.

    Returns:
        A message describing where the trained state_dict was saved.
    """
    resolved_num_workers = (
        num_workers if num_workers is not None else 0
    )
    normalized_type = datasource_type.lower()
    if normalized_type not in {"preprocess", "hcpe"}:
        msg = (
            "datasource_type must be either 'preprocess' or 'hcpe', "
            f"got {datasource_type}"
        )
        raise ValueError(msg)
    if cache_transforms is None:
        cache_transforms_enabled = normalized_type == "hcpe"
    else:
        cache_transforms_enabled = cache_transforms
    resolved_prefetch_factor = (
        prefetch_factor if prefetch_factor is not None else 2
    )
    option_kwargs = dict(
        datasource=datasource,
        cache_transforms=cache_transforms_enabled,
        config_path=config_path,
        output_path=output_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mask_ratio=mask_ratio,
        device=device,
        compilation=compilation,
        num_workers=resolved_num_workers,
        pin_memory=pin_memory,
        prefetch_factor=resolved_prefetch_factor,
        hidden_dim=hidden_dim,
    )
    if forward_chunk_size is not None:
        option_kwargs["forward_chunk_size"] = forward_chunk_size
    options = MaskedAutoencoderPretraining.Options(
        **option_kwargs
    )
    return MaskedAutoencoderPretraining().run(options)
