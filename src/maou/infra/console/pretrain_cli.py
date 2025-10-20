from pathlib import Path
from typing import Optional

import click

from maou.infra.console.common import (
    app_logger,
    FileDataSource,
    FileSystem,
    handle_exception,
)
from maou.interface.pretrain import (
    pretrain as pretrain_interface,
)


@click.command("pretrain")
@click.option(
    "--input-dir",
    type=click.Path(path_type=Path, exists=True),
    required=False,
    help="Input directory containing training data.",
)
@click.option(
    "--input-format",
    type=click.Choice(["preprocess", "hcpe"]),
    default="preprocess",
    show_default=True,
    help="Format of the input data (preprocess or hcpe).",
)
@click.option(
    "--input-file-packed/--no-input-file-packed",
    default=False,
    show_default=True,
    help="Indicates whether the local files are bit-packed.",
)
@click.option(
    "--config-path",
    type=click.Path(path_type=Path, exists=True),
    required=False,
    help="Optional configuration file for pretraining.",
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    required=False,
    help="Location to store the pretrained state_dict.",
)
@click.option(
    "--epochs",
    type=int,
    default=5,
    show_default=True,
    help="Number of training epochs.",
)
@click.option(
    "--batch-size",
    type=int,
    default=64,
    show_default=True,
    help="Mini-batch size used during optimisation.",
)
@click.option(
    "--learning-rate",
    type=float,
    default=1e-3,
    show_default=True,
    help="Learning rate for Adam optimiser.",
)
@click.option(
    "--mask-ratio",
    type=float,
    default=0.75,
    show_default=True,
    help="Fraction of elements masked prior to reconstruction.",
)
@click.option(
    "--device",
    type=str,
    required=False,
    help="Explicit device identifier (e.g. 'cpu' or 'cuda').",
)
@click.option(
    "--compilation/--no-compilation",
    default=False,
    show_default=True,
    help="Enable PyTorch compilation of the masked autoencoder.",
)
@click.option(
    "--dataloader-workers",
    "--num-workers",
    type=int,
    default=None,
    help="Number of worker processes for the DataLoader (default: 0).",
)
@click.option(
    "--prefetch-factor",
    type=int,
    default=None,
    help=(
        "Number of batches prefetched by each worker when workers are enabled "
        "(default: 2)."
    ),
)
@click.option(
    "--pin-memory/--no-pin-memory",
    default=None,
    help=(
        "Enable or disable pinned memory for DataLoader tensors. By default, "
        "it follows the selected device."
    ),
)
@click.option(
    "--hidden-dim",
    type=int,
    default=512,
    show_default=True,
    help="Hidden dimension of the autoencoder MLP.",
)
@click.option(
    "--forward-chunk-size",
    type=int,
    default=None,
    help=(
        "Maximum number of samples processed per forward pass. "
        "Defaults to an adaptive value."
    ),
)
@handle_exception
def pretrain(
    input_dir: Optional[Path],
    input_format: str,
    input_file_packed: bool,
    config_path: Optional[Path],
    output_path: Optional[Path],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    mask_ratio: float,
    device: Optional[str],
    compilation: bool,
    dataloader_workers: Optional[int],
    prefetch_factor: Optional[int],
    pin_memory: Optional[bool],
    hidden_dim: int,
    forward_chunk_size: Optional[int],
) -> None:
    """CLI entry point for masked autoencoder pretraining."""
    app_logger.info(
        "Invoking masked autoencoder pretraining command."
    )
    if input_dir is None:
        raise click.BadOptionUsage(
            option_name="--input-dir",
            message="--input-dir must be specified for pretraining.",
        )

    array_type = (
        "preprocessing" if input_format == "preprocess" else "hcpe"
    )
    file_paths = FileSystem.collect_files(input_dir)
    if not file_paths:
        raise click.ClickException(
            f"No input files found under {input_dir}."
        )

    datasource = FileDataSource.FileDataSourceSpliter(
        file_paths=file_paths,
        array_type=array_type,
        bit_pack=input_file_packed,
    )

    resolved_workers = dataloader_workers if dataloader_workers is not None else 0
    resolved_prefetch = (
        prefetch_factor if prefetch_factor is not None else 2
    )

    message = pretrain_interface(
        datasource=datasource,
        config_path=config_path,
        output_path=output_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mask_ratio=mask_ratio,
        device=device,
        compilation=compilation,
        num_workers=resolved_workers,
        pin_memory=pin_memory,
        prefetch_factor=resolved_prefetch,
        hidden_dim=hidden_dim,
        forward_chunk_size=forward_chunk_size,
    )
    click.echo(message)
