from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import click

import maou.infra.console.common as common
import maou.interface.learn as learn
from maou.interface.learn import StageDataConfig

if TYPE_CHECKING:
    from maou.infra.console.common import GCS as GCSType
    from maou.infra.console.common import S3 as S3Type
    from maou.infra.console.common import (
        BigQueryDataSource as BigQueryDataSourceType,
    )
    from maou.infra.console.common import (
        FileDataSource as FileDataSourceType,
    )
    from maou.infra.console.common import (
        FileSystem as FileSystemType,
    )
    from maou.infra.console.common import (
        GCSDataSource as GCSDataSourceType,
    )
    from maou.infra.console.common import (
        S3DataSource as S3DataSourceType,
    )
else:
    BigQueryDataSourceType = Any
    FileDataSourceType = Any
    FileSystemType = Any
    GCSType = Any
    GCSDataSourceType = Any
    S3Type = Any
    S3DataSourceType = Any


app_logger = common.app_logger
FileDataSource = common.FileDataSource
FileSystem = common.FileSystem
HAS_BIGQUERY = common.HAS_BIGQUERY
HAS_GCS = common.HAS_GCS
HAS_AWS = common.HAS_AWS
BigQueryDataSource: BigQueryDataSourceType | None = getattr(
    common, "BigQueryDataSource", None
)
GCS: GCSType | None = getattr(common, "GCS", None)
GCSDataSource: GCSDataSourceType | None = getattr(
    common, "GCSDataSource", None
)
S3: S3Type | None = getattr(common, "S3", None)
S3DataSource: S3DataSourceType | None = getattr(
    common, "S3DataSource", None
)


@click.command("learn-model")
@click.option(
    "--input-path",
    help="Input file or directory path.",
    type=click.Path(exists=True, path_type=Path),
    required=False,
)
@click.option(
    "--input-file-packed",
    type=bool,
    is_flag=True,
    help="Enable unpacking local numpy file.",
    default=False,
    required=False,
)
@click.option(
    "--input-dataset-id",
    help="BigQuery dataset ID for input.",
    type=str,
    required=False,
)
@click.option(
    "--input-table-name",
    help="BigQuery table name for input.",
    type=str,
    required=False,
)
@click.option(
    "--input-format",
    help="Input format: 'hcpe' or 'preprocess'.",
    type=str,
    default="hcpe",
    required=False,
)
@click.option(
    "--input-batch-size",
    help="Batch size for reading from BigQuery.",
    type=int,
    default=10000,
    required=False,
)
@click.option(
    "--input-max-cached-bytes",
    help="Max cache size in bytes for input (default: 500MB).",
    type=int,
    default=500 * 1024 * 1024,
    required=False,
)
@click.option(
    "--input-cache-mode",
    type=click.Choice(
        ["file", "memory", "mmap"], case_sensitive=False
    ),
    help="Cache strategy for local inputs (default: file). 'mmap' is deprecated, use 'file' instead.",
    default="file",
    show_default=True,
    required=False,
)
@click.option(
    "--input-clustering-key",
    help="BigQuery clustering key.",
    type=str,
    required=False,
)
@click.option(
    "--input-partitioning-key-date",
    help="BigQuery date partitioning key.",
    type=str,
    required=False,
)
@click.option(
    "--input-local-cache",
    type=bool,
    is_flag=True,
    help="Enable local caching of cloud data.",
    default=False,
    required=False,
)
@click.option(
    "--input-local-cache-dir",
    type=str,
    help="Directory path for storing the local cache of cloud data.",
    required=False,
)
@click.option(
    "--input-enable-bundling",
    type=bool,
    is_flag=True,
    help="Enable bundling of arrays for efficient local caching (1GB chunks).",
    default=False,
    required=False,
)
@click.option(
    "--input-bundle-size-gb",
    type=float,
    help="Target bundle size in GB for array bundling (default: 1.0).",
    default=1.0,
    required=False,
)
@click.option(
    "--input-gcs",
    type=bool,
    is_flag=True,
    help="Use GCS as input data source.",
    required=False,
)
@click.option(
    "--input-s3",
    type=bool,
    is_flag=True,
    help="Use S3 as input data source.",
    required=False,
)
@click.option(
    "--input-bucket-name",
    help="S3/GCS bucket name for input.",
    type=str,
    required=False,
)
@click.option(
    "--input-prefix",
    help="S3/GCS prefix path for input.",
    type=str,
    required=False,
)
@click.option(
    "--input-data-name",
    help="Name to identify the data in S3/GCS for input.",
    type=str,
    required=False,
)
@click.option(
    "--input-max-workers",
    help="Number of parallel download threads for S3/GCS input (default: 8).",
    type=int,
    required=False,
    default=8,
)
@click.option(
    "--gpu",
    type=str,
    help="PyTorch device (e.g., 'cuda:0' or 'cpu').",
    required=False,
)
@click.option(
    "--model-architecture",
    type=click.Choice(
        list(learn.SUPPORTED_MODEL_ARCHITECTURES),
        case_sensitive=False,
    ),
    help=(
        "Backbone architecture to use. Supported values: "
        + ", ".join(learn.SUPPORTED_MODEL_ARCHITECTURES)
    ),
    required=False,
    default="resnet",
    show_default=True,
)
@click.option(
    "--vit-embed-dim",
    type=int,
    default=None,
    help="ViT: embedding dimension (default: 512).",
)
@click.option(
    "--vit-num-layers",
    type=int,
    default=None,
    help="ViT: number of encoder layers (default: 6).",
)
@click.option(
    "--vit-num-heads",
    type=int,
    default=None,
    help="ViT: number of attention heads (default: 8).",
)
@click.option(
    "--vit-mlp-ratio",
    type=float,
    default=None,
    help="ViT: MLP hidden dimension ratio (default: 4.0).",
)
@click.option(
    "--vit-dropout",
    type=float,
    default=None,
    help="ViT: dropout rate (default: 0.1).",
)
@click.option(
    "--compilation",
    type=bool,
    help="Enable PyTorch compilation.",
    required=False,
    default=False,
)
@click.option(
    "--detect-anomaly",
    type=bool,
    is_flag=True,
    help="Enable torch.autograd anomaly detection (default: disabled).",
    required=False,
    default=False,
)
@click.option(
    "--test-ratio",
    type=float,
    help="Test set ratio.",
    required=False,
)
@click.option(
    "--epoch",
    type=int,
    help="Number of training epochs.",
    required=False,
)
@click.option(
    "--batch-size",
    type=int,
    help=(
        "Training batch size."
        " Recommended by GPU memory: 512 (8GB), 1024 (16GB),"
        " 2048 (24GB), 4096 (40-80GB)."
        " Use --gradient-accumulation-steps to simulate larger batches."
    ),
    required=False,
)
@click.option(
    "--dataloader-workers",
    type=int,
    help="Number of DataLoader workers.",
    required=False,
)
@click.option(
    "--pin-memory",
    type=bool,
    is_flag=True,
    help="Enable pinned memory for faster GPU transfers.",
    required=False,
)
@click.option(
    "--prefetch-factor",
    type=int,
    help="Number of batches loaded in advance by each worker (default: 4).",
    required=False,
)
@click.option(
    "--tensorboard-histogram-frequency",
    type=int,
    help="Log parameter histograms every N epochs (default: disabled).",
    default=0,
    show_default=True,
    required=False,
)
@click.option(
    "--tensorboard-histogram-module",
    "tensorboard_histogram_modules",
    multiple=True,
    type=str,
    help=(
        "Only log histograms for parameter names matching this glob pattern."
        " Provide multiple times to add more filters."
    ),
)
@click.option(
    "--cache-transforms/--no-cache-transforms",
    default=None,
    help=(
        "Enable in-memory caching of dataset transforms when supported by the "
        "input pipeline."
    ),
    required=False,
)
@click.option(
    "--gce-parameter",
    type=float,
    help="GCE loss hyperparameter (default: 0.1).",
    required=False,
    default=0.1,
)
@click.option(
    "--policy-loss-ratio",
    type=float,
    help="Policy loss weight.",
    required=False,
    default=1.0,
)
@click.option(
    "--value-loss-ratio",
    type=float,
    help="Value loss weight.",
    required=False,
    default=1.0,
)
@click.option(
    "--learning-ratio",
    type=float,
    help="Learning rate.",
    required=False,
    default=0.01,
)
@click.option(
    "--lr-scheduler",
    type=click.Choice(
        list(learn.SUPPORTED_LR_SCHEDULERS.values()),
        case_sensitive=False,
    ),
    help="Learning rate scheduler to apply.",
    required=False,
    default=learn.SUPPORTED_LR_SCHEDULERS[
        "warmup_cosine_decay"
    ],
    show_default=True,
)
@click.option(
    "--momentum",
    type=float,
    help="Optimizer momentum.",
    required=False,
)
@click.option(
    "--optimizer",
    type=click.Choice(["adamw", "sgd"], case_sensitive=False),
    help="Optimizer to use (default: adamw).",
    required=False,
    default="adamw",
    show_default=True,
)
@click.option(
    "--optimizer-beta1",
    type=float,
    help="AdamW beta1 parameter.",
    required=False,
    default=0.9,
    show_default=True,
)
@click.option(
    "--optimizer-beta2",
    type=float,
    help="AdamW beta2 parameter.",
    required=False,
    default=0.999,
    show_default=True,
)
@click.option(
    "--optimizer-eps",
    type=float,
    help="AdamW epsilon parameter.",
    required=False,
    default=1e-8,
    show_default=True,
)
@click.option(
    "--resume-from",
    type=click.Path(path_type=Path),
    help="Checkpoint file to resume training.",
    required=False,
)
@click.option(
    "--start-epoch",
    type=int,
    help=(
        "Starting epoch number (0-indexed, default: 0). "
        "If set to 1, training starts from epoch 2."
    ),
    required=False,
)
@click.option(
    "--resume-backbone-from",
    type=click.Path(exists=True, path_type=Path),
    help="Backbone parameter file to resume training.",
    required=False,
)
@click.option(
    "--resume-policy-head-from",
    type=click.Path(exists=True, path_type=Path),
    help="Policy head parameter file to resume training.",
    required=False,
)
@click.option(
    "--resume-value-head-from",
    type=click.Path(exists=True, path_type=Path),
    help="Value head parameter file to resume training.",
    required=False,
)
@click.option(
    "--freeze-backbone",
    is_flag=True,
    default=False,
    help="Freeze backbone parameters (embedding, backbone, pool, hand projection).",
)
@click.option(
    "--trainable-layers",
    type=int,
    default=None,
    help=(
        "Number of trailing backbone layer groups to keep trainable. "
        "0 = freeze all backbone layers. "
        "Unset = no freezing (all layers trainable)."
    ),
)
@click.option(
    "--stage",
    type=click.Choice(
        ["1", "2", "3", "all"], case_sensitive=False
    ),
    default="3",
    help="Training stage: 1=Reachable Squares, 2=Legal Moves, 3=Policy+Value, all=Sequential",
    show_default=True,
)
@click.option(
    "--stage1-data-path",
    type=click.Path(exists=True, path_type=Path),
    help="File or directory path for Stage 1 (reachable squares) training data.",
    required=False,
)
@click.option(
    "--stage2-data-path",
    type=click.Path(exists=True, path_type=Path),
    help="File or directory path for Stage 2 (legal moves) training data.",
    required=False,
)
@click.option(
    "--stage3-data-path",
    type=click.Path(exists=True, path_type=Path),
    help="File or directory path for Stage 3 (policy+value) training data.",
    required=False,
)
@click.option(
    "--stage1-threshold",
    type=float,
    default=0.99,
    help="Accuracy threshold for Stage 1 (default: 0.99 = 99%%).",
    show_default=True,
)
@click.option(
    "--stage2-threshold",
    type=float,
    default=0.85,
    help="F1 threshold for Stage 2 (default: 0.85 = 85%%).",
    show_default=True,
)
@click.option(
    "--stage1-max-epochs",
    type=int,
    default=10,
    help="Maximum epochs for Stage 1.",
    show_default=True,
)
@click.option(
    "--stage2-max-epochs",
    type=int,
    default=10,
    help="Maximum epochs for Stage 2.",
    show_default=True,
)
@click.option(
    "--stage1-batch-size",
    type=int,
    default=None,
    help="Batch size for Stage 1 (default: inherits --batch-size).",
    required=False,
)
@click.option(
    "--stage2-batch-size",
    type=int,
    default=None,
    help="Batch size for Stage 2 (default: inherits --batch-size).",
    required=False,
)
@click.option(
    "--stage1-learning-rate",
    type=float,
    default=None,
    help="Learning rate for Stage 1 (default: inherits --learning-ratio).",
    required=False,
)
@click.option(
    "--stage2-learning-rate",
    type=float,
    default=None,
    help="Learning rate for Stage 2 (default: inherits --learning-ratio).",
    required=False,
)
@click.option(
    "--stage12-lr-scheduler",
    type=click.Choice(
        ["auto", "none"]
        + list(learn.SUPPORTED_LR_SCHEDULERS.values()),
        case_sensitive=False,
    ),
    help=(
        "Learning rate scheduler for Stage 1/2."
        " 'auto' enables Warmup+CosineDecay when batch_size > 256."
        " 'none' disables the scheduler (fixed LR)."
    ),
    required=False,
    default="auto",
    show_default=True,
)
@click.option(
    "--stage12-compilation/--no-stage12-compilation",
    default=False,
    help=(
        "Enable torch.compile for Stage 1/2 backbone model."
        " Provides 10-30%% speedup on A100."
    ),
    show_default=True,
)
@click.option(
    "--stage1-pos-weight",
    type=float,
    default=1.0,
    show_default=True,
    help="Positive class weight for Stage 1 loss.",
)
@click.option(
    "--stage2-pos-weight",
    type=float,
    default=1.0,
    show_default=True,
    help="Positive class weight for Stage 2 BCE loss. Usually 1.0 when ASL is enabled.",
)
@click.option(
    "--stage2-gamma-pos",
    type=float,
    default=0.0,
    show_default=True,
    help=(
        "ASL positive focusing parameter for Stage 2."
        " 0.0 = no down-weighting of positive loss (recommended)."
    ),
)
@click.option(
    "--stage2-gamma-neg",
    type=float,
    default=0.0,
    show_default=True,
    help=(
        "ASL negative focusing parameter for Stage 2."
        " 0.0 = standard BCE, 2.0 = recommended for imbalanced data."
    ),
)
@click.option(
    "--stage2-clip",
    type=float,
    default=0.0,
    show_default=True,
    help=(
        "ASL negative probability clipping margin for Stage 2."
        " 0.0 = disabled, 0.02 = recommended."
    ),
)
@click.option(
    "--stage2-hidden-dim",
    type=int,
    default=None,
    show_default=True,
    help="Hidden layer dimension for Stage 2 head. None = single linear layer.",
)
@click.option(
    "--stage2-head-dropout",
    type=float,
    default=0.0,
    show_default=True,
    help="Dropout rate for Stage 2 head (requires --stage2-hidden-dim).",
)
@click.option(
    "--stage2-test-ratio",
    type=float,
    default=0.0,
    show_default=True,
    help="Validation split ratio for Stage 2 (0.0 = no split, e.g. 0.1 = 10%% validation).",
)
@click.option(
    "--resume-reachable-head-from",
    type=click.Path(exists=True, path_type=Path),
    help="Reachable squares head parameter file to resume training (Stage 1).",
    required=False,
)
@click.option(
    "--resume-legal-moves-head-from",
    type=click.Path(exists=True, path_type=Path),
    help="Legal moves head parameter file to resume training (Stage 2).",
    required=False,
)
@click.option(
    "--no-streaming",
    is_flag=True,
    default=False,
    help="Disable streaming mode for file input (use map-style dataset instead).",
)
@click.option(
    "--log-dir",
    type=click.Path(path_type=Path),
    help="Log directory for SummaryWriter.",
    required=False,
)
@click.option(
    "--model-dir",
    type=click.Path(path_type=Path),
    help="Model output directory.",
    required=False,
)
@click.option(
    "--output-gcs",
    type=bool,
    is_flag=True,
    help="Upload output to Google Cloud Storage.",
    required=False,
)
@click.option(
    "--gcs-bucket-name",
    help="GCS bucket name.",
    type=str,
    required=False,
)
@click.option(
    "--gcs-base-path",
    help="Base path in GCS bucket.",
    type=str,
    required=False,
)
@click.option(
    "--output-s3",
    type=bool,
    is_flag=True,
    help="Upload output to AWS S3.",
    required=False,
)
@click.option(
    "--s3-bucket-name",
    help="S3 bucket name.",
    type=str,
    required=False,
)
@click.option(
    "--s3-base-path",
    help="Base path in S3 bucket.",
    type=str,
    required=False,
)
@common.handle_exception
def learn_model(
    input_path: Optional[Path],
    input_file_packed: bool,
    input_dataset_id: Optional[str],
    input_table_name: Optional[str],
    input_format: str,
    input_batch_size: int,
    input_max_cached_bytes: int,
    input_cache_mode: str,
    input_clustering_key: Optional[str],
    input_partitioning_key_date: Optional[str],
    input_local_cache: bool,
    input_local_cache_dir: Optional[str],
    input_enable_bundling: bool,
    input_bundle_size_gb: float,
    input_gcs: Optional[bool],
    input_s3: Optional[bool],
    input_bucket_name: Optional[str],
    input_prefix: Optional[str],
    input_data_name: Optional[str],
    input_max_workers: int,
    gpu: Optional[str],
    model_architecture: str,
    vit_embed_dim: Optional[int],
    vit_num_layers: Optional[int],
    vit_num_heads: Optional[int],
    vit_mlp_ratio: Optional[float],
    vit_dropout: Optional[float],
    compilation: bool,
    detect_anomaly: bool,
    test_ratio: Optional[float],
    epoch: Optional[int],
    batch_size: Optional[int],
    dataloader_workers: Optional[int],
    pin_memory: Optional[bool],
    prefetch_factor: Optional[int],
    tensorboard_histogram_frequency: int,
    tensorboard_histogram_modules: tuple[str, ...],
    cache_transforms: Optional[bool],
    gce_parameter: Optional[float],
    policy_loss_ratio: Optional[float],
    value_loss_ratio: Optional[float],
    learning_ratio: Optional[float],
    lr_scheduler: str,
    momentum: Optional[float],
    optimizer: str,
    optimizer_beta1: float,
    optimizer_beta2: float,
    optimizer_eps: float,
    resume_from: Optional[Path],
    start_epoch: Optional[int],
    resume_backbone_from: Optional[Path],
    resume_policy_head_from: Optional[Path],
    resume_value_head_from: Optional[Path],
    freeze_backbone: bool,
    trainable_layers: Optional[int],
    stage: str,
    stage1_data_path: Optional[Path],
    stage2_data_path: Optional[Path],
    stage3_data_path: Optional[Path],
    stage1_threshold: float,
    stage2_threshold: float,
    stage1_max_epochs: int,
    stage2_max_epochs: int,
    stage1_batch_size: Optional[int],
    stage2_batch_size: Optional[int],
    stage1_learning_rate: Optional[float],
    stage2_learning_rate: Optional[float],
    stage12_lr_scheduler: Optional[str],
    stage12_compilation: bool,
    stage1_pos_weight: float,
    stage2_pos_weight: float,
    stage2_gamma_pos: float,
    stage2_gamma_neg: float,
    stage2_clip: float,
    stage2_hidden_dim: Optional[int],
    stage2_head_dropout: float,
    stage2_test_ratio: float,
    resume_reachable_head_from: Optional[Path],
    resume_legal_moves_head_from: Optional[Path],
    no_streaming: bool,
    log_dir: Optional[Path],
    model_dir: Optional[Path],
    output_gcs: Optional[bool],
    gcs_bucket_name: Optional[str],
    gcs_base_path: Optional[str],
    output_s3: Optional[bool],
    s3_bucket_name: Optional[str],
    s3_base_path: Optional[str],
) -> None:
    # Normalize cache_mode: "mmap" is deprecated, convert to "file"
    if input_cache_mode.lower() == "mmap":
        import warnings

        warnings.warn(
            "--input-cache-mode 'mmap' is deprecated. Use 'file' instead.",
            DeprecationWarning,
            stacklevel=1,
        )
        input_cache_mode = "file"

    # Validate input_format early
    if input_format not in ("hcpe", "preprocess"):
        raise ValueError(
            "Please specify a valid input_format ('hcpe' or 'preprocess')."
        )
    # Convert input_format to array_type for data sources
    array_type = (
        "preprocessing"
        if input_format == "preprocess"
        else "hcpe"
    )

    # Check for mixing cloud providers for output
    if output_gcs and output_s3:
        error_msg = (
            "Cannot use both GCS and S3 as cloud storage simultaneously. "
            "Please choose only one cloud provider for output."
        )
        app_logger.error(error_msg)
        raise ValueError(error_msg)

    # Initialize cloud storage
    cloud_storage = None
    if (
        output_gcs
        and gcs_bucket_name is not None
        and gcs_base_path is not None
    ):
        if HAS_GCS and GCS is not None:
            try:
                cloud_storage = GCS(
                    bucket_name=gcs_bucket_name,
                    base_path=gcs_base_path,
                )
            except Exception as e:
                app_logger.error(
                    f"Failed to initialize GCS: {e}"
                )
        else:
            app_logger.warning(
                "GCS output requested but required packages are not installed. "
                "Install with 'uv sync --extra gcp'"
            )
    elif (
        output_s3
        and s3_bucket_name is not None
        and s3_base_path is not None
    ):
        if HAS_AWS and S3 is not None:
            try:
                cloud_storage = S3(
                    bucket_name=s3_bucket_name,
                    base_path=s3_base_path,
                )
            except Exception as e:
                app_logger.error(
                    f"Failed to initialize S3: {e}"
                )
        else:
            app_logger.warning(
                "S3 output requested but required packages are not installed. "
                "Install with 'uv sync --extra aws'"
            )

    # Check for mixing cloud providers for input
    cloud_input_count = sum(
        [
            bool(
                input_dataset_id is not None
                or input_table_name is not None
            ),
            bool(input_gcs),
            bool(input_s3),
        ]
    )
    if cloud_input_count > 1:
        error_msg = (
            "Cannot use multiple cloud providers for input simultaneously. "
            "Please choose only one: BigQuery, GCS, or S3."
        )
        app_logger.error(error_msg)
        raise ValueError(error_msg)

    # Check if multi-stage training is requested (moved before datasource init)
    is_multi_stage = (
        stage in ("1", "2", "all")
        or stage1_data_path is not None
        or stage2_data_path is not None
    )

    # Initialize datasource
    # When streaming is active for file input, FileDataSourceSpliter is
    # NOT created here to avoid loading all data into memory eagerly.
    # Instead, file paths are split directly and StreamingFileSource is
    # used.  FileDataSourceSpliter is only created as a fallback when
    # streaming cannot be used (single file, --no-streaming, or
    # non-file input sources).
    datasource = None
    file_paths_for_streaming: list[Path] | None = None
    if input_path is not None:
        if (
            input_format != "hcpe"
            and input_format != "preprocess"
        ):
            raise Exception(
                "Please specify a valid input_format ('hcpe' or 'preprocess')."
            )
        collected_paths = FileSystem.collect_files(input_path)
        if no_streaming or len(collected_paths) < 2:
            # Map-style: load everything into memory
            datasource = FileDataSource.FileDataSourceSpliter(
                file_paths=collected_paths,
                array_type=array_type,
                bit_pack=input_file_packed,
                cache_mode=input_cache_mode.lower(),
            )
            if len(collected_paths) < 2 and not no_streaming:
                app_logger.info(
                    "Single file detected; falling back to map-style dataset."
                )
        else:
            # Streaming: defer data loading, just keep paths
            file_paths_for_streaming = collected_paths
    elif (
        input_dataset_id is not None
        and input_table_name is not None
    ):
        if HAS_BIGQUERY and BigQueryDataSource is not None:
            try:
                # BigQueryDataSourceSpliterを使用
                if hasattr(
                    BigQueryDataSource,
                    "BigQueryDataSourceSpliter",
                ):
                    datasource = BigQueryDataSource.BigQueryDataSourceSpliter(
                        array_type=array_type,
                        dataset_id=input_dataset_id,
                        table_name=input_table_name,
                        batch_size=input_batch_size,
                        max_cached_bytes=input_max_cached_bytes,
                        clustering_key=input_clustering_key,
                        partitioning_key_date=input_partitioning_key_date,
                        use_local_cache=input_local_cache,
                        local_cache_dir=input_local_cache_dir,
                        sample_ratio=None,
                    )
                else:
                    app_logger.error(
                        "BigQueryDataSourceSpliter not available"
                    )
                    raise AttributeError(
                        "BigQueryDataSourceSpliter not available"
                    )
            except Exception as e:
                app_logger.error(
                    f"Failed to initialize BigQueryDataSourceSpliter: {e}"
                )
                raise
        else:
            error_msg = (
                "BigQuery input requested but required packages are not installed. "
                "Install with 'uv sync --extra gcp'"
            )
            app_logger.error(error_msg)
            raise ImportError(error_msg)
    elif (
        input_gcs
        and input_bucket_name is not None
        and input_prefix is not None
        and input_data_name is not None
        and input_local_cache_dir is not None
    ):
        if HAS_GCS and GCSDataSource is not None:
            try:
                # GCSDataSourceSpliterを使用
                if hasattr(GCSDataSource, "DataSourceSpliter"):
                    datasource = GCSDataSource.DataSourceSpliter(
                        cls_ref=GCSDataSource,
                        bucket_name=input_bucket_name,
                        prefix=input_prefix,
                        data_name=input_data_name,
                        array_type=array_type,
                        local_cache_dir=input_local_cache_dir,
                        max_workers=input_max_workers,
                        max_cached_bytes=input_max_cached_bytes,
                        sample_ratio=None,
                        enable_bundling=input_enable_bundling,
                        bundle_size_gb=input_bundle_size_gb,
                    )
                else:
                    app_logger.error(
                        "DataSourceSpliter not available"
                    )
                    raise AttributeError(
                        "DataSourceSpliter not available"
                    )
            except Exception as e:
                app_logger.error(
                    f"Failed to initialize DataSourceSpliter: {e}"
                )
                raise
        else:
            error_msg = (
                "GCS input requested but required packages are not installed. "
                "Install with 'uv sync --extra gcp'"
            )
            app_logger.error(error_msg)
            raise ImportError(error_msg)
    elif (
        input_s3
        and input_bucket_name is not None
        and input_prefix is not None
        and input_data_name is not None
        and input_local_cache_dir is not None
    ):
        if HAS_AWS and S3DataSource is not None:
            try:
                # S3DataSourceSpliterを使用
                if hasattr(S3DataSource, "DataSourceSpliter"):
                    datasource = S3DataSource.DataSourceSpliter(
                        cls_ref=S3DataSource,
                        bucket_name=input_bucket_name,
                        prefix=input_prefix,
                        data_name=input_data_name,
                        array_type=array_type,
                        local_cache_dir=input_local_cache_dir,
                        max_workers=input_max_workers,
                        max_cached_bytes=input_max_cached_bytes,
                        sample_ratio=None,
                        enable_bundling=input_enable_bundling,
                        bundle_size_gb=input_bundle_size_gb,
                    )
                else:
                    app_logger.error(
                        "DataSourceSpliter not available"
                    )
                    raise AttributeError(
                        "DataSourceSpliter not available"
                    )
            except Exception as e:
                app_logger.error(
                    f"Failed to initialize S3DataSourceSpliter: {e}"
                )
                raise
        else:
            error_msg = (
                "S3 input requested but required packages are not installed. "
                "Install with 'uv sync --extra aws'"
            )
            app_logger.error(error_msg)
            raise ImportError(error_msg)
    elif not is_multi_stage:
        raise Exception(
            "Please specify an input directory, a BigQuery table, "
            "a GCS bucket, or an S3 bucket."
        )
    architecture_key = model_architecture.lower()

    # Build architecture_config from ViT-specific options
    architecture_config: dict[str, Any] | None = None
    if architecture_key == "vit":
        vit_overrides: dict[str, Any] = {}
        if vit_embed_dim is not None:
            vit_overrides["embed_dim"] = vit_embed_dim
        if vit_num_layers is not None:
            vit_overrides["num_layers"] = vit_num_layers
        if vit_num_heads is not None:
            vit_overrides["num_heads"] = vit_num_heads
        if vit_mlp_ratio is not None:
            vit_overrides["mlp_ratio"] = vit_mlp_ratio
        if vit_dropout is not None:
            vit_overrides["dropout"] = vit_dropout
        if vit_overrides:
            architecture_config = vit_overrides

    if is_multi_stage:
        # Route to multi-stage training
        app_logger.info(
            f"Multi-stage training requested: stage={stage}"
        )

        # 遅延初期化用のStageDataConfigを生成(この時点ではI/O/メモリ消費なし)
        stage1_data_config: StageDataConfig | None = None
        stage2_data_config: StageDataConfig | None = None
        stage3_data_config: StageDataConfig | None = None

        # Stage1/2 では cache_mode を強制的に "file" にする（OOM防止）
        stage12_cache_mode = "file"
        if input_cache_mode.lower() == "memory":
            app_logger.info(
                "Stage 1/2: cache_mode='memory' is ignored for memory efficiency. "
                "Using individual file mode instead."
            )

        # Streaming source variables for multi-stage
        use_multi_streaming = False
        s1_streaming_source = None
        s2_streaming_source = None
        s3_streaming_train_source = None
        s3_streaming_val_source = None

        if stage1_data_path is not None:
            _s1_paths = FileSystem.collect_files(
                stage1_data_path
            )
            stage1_data_config = StageDataConfig(
                create_datasource=lambda _paths=_s1_paths: (
                    FileDataSource.FileDataSourceSpliter(
                        file_paths=_paths,
                        array_type="stage1",
                        bit_pack=False,
                        cache_mode=stage12_cache_mode,
                    )
                ),
                array_type="stage1",
            )
            if not no_streaming:
                from maou.infra.file_system.streaming_file_source import (
                    StreamingFileSource,
                )

                s1_streaming_source = StreamingFileSource(
                    file_paths=_s1_paths,
                    array_type="stage1",
                )
                use_multi_streaming = True

        if stage2_data_path is not None:
            _s2_paths = FileSystem.collect_files(
                stage2_data_path
            )
            stage2_data_config = StageDataConfig(
                create_datasource=lambda _paths=_s2_paths: (
                    FileDataSource.FileDataSourceSpliter(
                        file_paths=_paths,
                        array_type="stage2",
                        bit_pack=False,
                        cache_mode=stage12_cache_mode,
                    )
                ),
                array_type="stage2",
            )
            if not no_streaming:
                from maou.infra.file_system.streaming_file_source import (
                    StreamingFileSource,
                )

                s2_streaming_source = StreamingFileSource(
                    file_paths=_s2_paths,
                    array_type="stage2",
                )
                use_multi_streaming = True

        if stage3_data_path is not None:
            _s3_paths = FileSystem.collect_files(
                stage3_data_path
            )
            _s3_cache = input_cache_mode.lower()
            _s3_at = "preprocessing"
            _s3_bp = input_file_packed
            stage3_data_config = StageDataConfig(
                create_datasource=lambda _p=_s3_paths,
                _a=_s3_at,
                _b=_s3_bp,
                _c=_s3_cache: (
                    FileDataSource.FileDataSourceSpliter(
                        file_paths=_p,
                        array_type=_a,
                        bit_pack=_b,
                        cache_mode=_c,
                    )
                ),
                array_type=_s3_at,
            )
            if not no_streaming and len(_s3_paths) >= 2:
                import random

                from maou.infra.file_system.streaming_file_source import (
                    StreamingFileSource,
                )

                rng = random.Random(42)
                shuffled = list(_s3_paths)
                rng.shuffle(shuffled)
                effective_ratio = test_ratio or 0.1
                n_val = max(
                    1, int(len(shuffled) * effective_ratio)
                )
                n_train = len(shuffled) - n_val
                if n_train < 1:
                    n_train = 1
                    n_val = len(shuffled) - 1
                s3_streaming_train_source = StreamingFileSource(
                    file_paths=shuffled[:n_train],
                    array_type=_s3_at,
                )
                s3_streaming_val_source = StreamingFileSource(
                    file_paths=shuffled[n_train:],
                    array_type=_s3_at,
                )
                use_multi_streaming = True
            elif not no_streaming:
                app_logger.info(
                    "Stage 3: Single file detected; falling back to map-style dataset."
                )

        if use_multi_streaming:
            app_logger.info(
                "Using streaming mode for multi-stage training."
            )

        click.echo(
            learn.learn_multi_stage(
                stage=stage,
                stage1_data_config=stage1_data_config,
                stage2_data_config=stage2_data_config,
                stage3_data_config=stage3_data_config,
                stage1_threshold=stage1_threshold,
                stage2_threshold=stage2_threshold,
                stage1_max_epochs=stage1_max_epochs,
                stage2_max_epochs=stage2_max_epochs,
                gpu=gpu,
                model_architecture=architecture_key,
                batch_size=batch_size or 4096,
                stage1_batch_size=stage1_batch_size,
                stage2_batch_size=stage2_batch_size,
                stage1_learning_rate=stage1_learning_rate,
                stage2_learning_rate=stage2_learning_rate,
                stage12_lr_scheduler=stage12_lr_scheduler,
                stage12_compilation=stage12_compilation,
                stage1_pos_weight=stage1_pos_weight,
                stage2_pos_weight=stage2_pos_weight,
                stage2_gamma_pos=stage2_gamma_pos,
                stage2_gamma_neg=stage2_gamma_neg,
                stage2_clip=stage2_clip,
                stage2_head_hidden_dim=stage2_hidden_dim,
                stage2_head_dropout=stage2_head_dropout,
                stage2_test_ratio=stage2_test_ratio,
                learning_rate=learning_ratio or 0.001,
                model_dir=model_dir,
                resume_backbone_from=resume_backbone_from,
                resume_reachable_head_from=resume_reachable_head_from,
                resume_legal_moves_head_from=resume_legal_moves_head_from,
                compilation=compilation,
                detect_anomaly=detect_anomaly,
                test_ratio=test_ratio,
                epoch=epoch,
                dataloader_workers=dataloader_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                cache_transforms=cache_transforms,
                gce_parameter=gce_parameter,
                policy_loss_ratio=policy_loss_ratio,
                value_loss_ratio=value_loss_ratio,
                lr_scheduler=lr_scheduler,
                momentum=momentum,
                optimizer_name=optimizer,
                optimizer_beta1=optimizer_beta1,
                optimizer_beta2=optimizer_beta2,
                optimizer_eps=optimizer_eps,
                freeze_backbone=freeze_backbone,
                trainable_layers=trainable_layers,
                log_dir=log_dir,
                cloud_storage=cloud_storage,
                input_cache_mode=input_cache_mode.lower(),
                architecture_config=architecture_config,
                streaming=use_multi_streaming,
                stage1_streaming_source=s1_streaming_source,
                stage2_streaming_source=s2_streaming_source,
                stage3_streaming_train_source=s3_streaming_train_source,
                stage3_streaming_val_source=s3_streaming_val_source,
            )
        )
    else:
        # Standard single-stage training
        use_streaming = False
        streaming_train_source = None
        streaming_val_source = None

        if file_paths_for_streaming is not None:
            # Streaming path: split file paths directly (no data loading)
            import random

            rng = random.Random(42)
            shuffled = list(file_paths_for_streaming)
            rng.shuffle(shuffled)
            effective_ratio = test_ratio or 0.1
            n_val = max(1, int(len(shuffled) * effective_ratio))
            n_train = len(shuffled) - n_val
            if n_train < 1:
                n_train = 1
                n_val = len(shuffled) - 1
            train_paths = shuffled[:n_train]
            val_paths = shuffled[n_train:]

            from maou.infra.file_system.streaming_file_source import (
                StreamingFileSource,
            )

            streaming_train_source = StreamingFileSource(
                file_paths=train_paths,
                array_type=array_type,
            )
            streaming_val_source = StreamingFileSource(
                file_paths=val_paths,
                array_type=array_type,
            )
            use_streaming = True
            app_logger.info(
                "Using streaming mode for file input."
            )

        click.echo(
            learn.learn(
                datasource=datasource,
                datasource_type=input_format,
                gpu=gpu,
                model_architecture=architecture_key,
                compilation=compilation,
                test_ratio=test_ratio,
                epoch=epoch,
                batch_size=batch_size,
                dataloader_workers=dataloader_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                tensorboard_histogram_frequency=tensorboard_histogram_frequency,
                tensorboard_histogram_modules=(
                    tensorboard_histogram_modules or None
                ),
                cache_transforms=cache_transforms,
                gce_parameter=gce_parameter,
                policy_loss_ratio=policy_loss_ratio,
                value_loss_ratio=value_loss_ratio,
                learning_ratio=learning_ratio,
                lr_scheduler=lr_scheduler,
                momentum=momentum,
                optimizer_name=optimizer,
                optimizer_beta1=optimizer_beta1,
                optimizer_beta2=optimizer_beta2,
                optimizer_eps=optimizer_eps,
                resume_from=resume_from,
                start_epoch=start_epoch,
                resume_backbone_from=resume_backbone_from,
                resume_policy_head_from=resume_policy_head_from,
                resume_value_head_from=resume_value_head_from,
                freeze_backbone=freeze_backbone,
                trainable_layers=trainable_layers,
                log_dir=log_dir,
                model_dir=model_dir,
                cloud_storage=cloud_storage,
                input_cache_mode=input_cache_mode.lower(),
                detect_anomaly=detect_anomaly,
                architecture_config=architecture_config,
                streaming=use_streaming,
                streaming_train_source=streaming_train_source,
                streaming_val_source=streaming_val_source,
            )
        )
