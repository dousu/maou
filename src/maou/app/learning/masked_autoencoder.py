from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from maou.app.learning.dl import LearningDataSource
from maou.app.learning.setup import (
    ModelFactory,
    default_worker_init_fn,
)


class _FeatureDataset(Dataset):
    """Dataset that returns flattened feature tensors from a data source."""

    def __init__(self, datasource: LearningDataSource) -> None:
        if len(datasource) == 0:
            msg = "Datasource contains no samples"
            raise ValueError(msg)
        first_record = datasource[0]
        first_features = self._extract_features(first_record)
        if first_features.ndim != 3:
            msg = "features must be 3-dimensional (channels, height, width)"
            raise ValueError(msg)

        self.original_shape: tuple[int, ...] = tuple(
            first_features.shape
        )
        self._num_features = int(first_features.size)
        self._datasource = datasource

    def __len__(
        self,
    ) -> int:  # pragma: no cover - simple delegation
        return len(self._datasource)

    def __getitem__(self, idx: int) -> torch.Tensor:
        record = self._datasource[idx]
        features = self._extract_features(record)
        flattened = np.ascontiguousarray(features.reshape(-1))
        return torch.from_numpy(flattened)

    @property
    def num_features(self) -> int:
        return self._num_features

    @property
    def feature_shape(self) -> tuple[int, int, int]:
        channels, height, width = self.original_shape
        return int(channels), int(height), int(width)

    def _extract_features(
        self, record: np.ndarray
    ) -> np.ndarray:
        if (
            record.dtype.names is None
            or "features" not in record.dtype.names
        ):
            msg = "Record does not contain 'features' field"
            raise ValueError(msg)
        features = np.asarray(
            record["features"], dtype=np.float32
        )
        if features.ndim < 2:
            msg = "features must be at least 2-dimensional"
            raise ValueError(msg)
        return features


class _MaskedAutoencoder(nn.Module):
    """Masked autoencoder that reuses the shogi mixer backbone as encoder."""

    def __init__(
        self,
        *,
        feature_shape: tuple[int, int, int],
        hidden_dim: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            msg = "hidden_dim must be a positive integer"
            raise ValueError(msg)

        self._feature_shape = feature_shape
        self._flattened_size = int(np.prod(feature_shape))
        channels = feature_shape[0]

        self.encoder = ModelFactory.create_shogi_model(device)
        self.decoder = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self._flattened_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        reshaped = x.view(batch_size, *self._feature_shape)
        encoded = self.encoder.backbone.forward_features(
            reshaped
        )
        decoded = self.decoder(encoded)
        return decoded.view(batch_size, -1)

    def encoder_state_dict(self) -> Dict[str, Any]:
        """Return the underlying encoder state for downstream training."""

        return self.encoder.state_dict()


class MaskedAutoencoderPretraining:
    """Masked autoencoder pretraining workflow."""

    logger: logging.Logger = logging.getLogger(__name__)

    @dataclass(kw_only=True, frozen=True)
    class Options:
        """Configuration options for masked autoencoder pretraining."""

        datasource: LearningDataSource.DataSourceSpliter
        config_path: Optional[Path] = None
        output_path: Optional[Path] = None
        epochs: int = 5
        batch_size: int = 64
        learning_rate: float = 1e-3
        mask_ratio: float = 0.75
        device: Optional[str] = None
        compilation: bool = False
        num_workers: int = 0
        hidden_dim: int = 512
        pin_memory: Optional[bool] = None
        prefetch_factor: int = 2
        progress_bar: bool = True

    def run(
        self, options: "MaskedAutoencoderPretraining.Options"
    ) -> str:
        """Execute masked autoencoder pretraining.

        Args:
            options: Pretraining configuration options.

        Returns:
            Description of the completed training job including the
            checkpoint path.
        """

        resolved_options = self._apply_config_overrides(options)
        training_datasource, _ = (
            resolved_options.datasource.train_test_split(
                test_ratio=0.0
            )
        )

        if resolved_options.epochs <= 0:
            msg = "epochs must be greater than zero"
            raise ValueError(msg)
        if resolved_options.batch_size <= 0:
            msg = "batch_size must be greater than zero"
            raise ValueError(msg)
        if not 0.0 <= resolved_options.mask_ratio <= 1.0:
            msg = "mask_ratio must be in the range [0.0, 1.0]"
            raise ValueError(msg)
        if resolved_options.prefetch_factor <= 0:
            msg = "prefetch_factor must be greater than zero"
            raise ValueError(msg)

        dataset = _FeatureDataset(training_datasource)
        output_path = self._resolve_output_path(
            resolved_options.output_path
        )
        device = self._resolve_device(resolved_options.device)
        pin_memory = self._resolve_pin_memory(
            resolved_options.pin_memory, device
        )

        dataloader = self._create_dataloader(
            dataset=dataset,
            batch_size=resolved_options.batch_size,
            num_workers=resolved_options.num_workers,
            pin_memory=pin_memory,
            prefetch_factor=resolved_options.prefetch_factor,
        )

        model = _MaskedAutoencoder(
            feature_shape=dataset.feature_shape,
            hidden_dim=resolved_options.hidden_dim,
            device=device,
        ).to(device)
        training_model: nn.Module
        if resolved_options.compilation:
            training_model = torch.compile(model)
        else:
            training_model = model
        optimizer = torch.optim.Adam(
            training_model.parameters(),
            lr=resolved_options.learning_rate,
        )

        self.logger.info(
            "Starting masked autoencoder pretraining with %s samples",
            len(dataset),
        )

        for epoch in range(resolved_options.epochs):
            epoch_loss = self._run_epoch(
                model=training_model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                mask_ratio=resolved_options.mask_ratio,
                epoch_index=epoch,
                total_epochs=resolved_options.epochs,
                progress_bar=resolved_options.progress_bar,
            )
            self.logger.info(
                "Epoch %d/%d - loss: %.6f",
                epoch + 1,
                resolved_options.epochs,
                epoch_loss,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_encoder_state_dict(model, output_path)
        message = (
            "Masked autoencoder pretraining finished. Saved state_dict to "
            f"{output_path}"
        )
        self.logger.info(message)
        return message

    def _apply_config_overrides(
        self, options: "MaskedAutoencoderPretraining.Options"
    ) -> "MaskedAutoencoderPretraining.Options":
        if options.config_path is None:
            return options
        overrides = self._load_config(options.config_path)
        option_dict = dataclasses.asdict(options)
        for key, value in overrides.items():
            if key == "config_path":
                continue
            if key == "datasource":
                msg = "datasource cannot be overridden via configuration"
                raise ValueError(msg)
            if key not in option_dict:
                msg = f"Unknown configuration option: {key}"
                raise ValueError(msg)
            option_dict[key] = self._coerce_option_value(
                key, value
            )
        return MaskedAutoencoderPretraining.Options(
            **option_dict
        )

    def _load_config(self, path: Path) -> Dict[str, Any]:
        suffix = path.suffix.lower()
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        elif suffix == ".toml":
            import tomllib  # Python 3.11+

            with path.open("rb") as handle:
                data = tomllib.load(handle)
        else:
            msg = (
                "Unsupported configuration format. Use JSON or TOML: "
                f"{path}"
            )
            raise ValueError(msg)
        if not isinstance(data, dict):
            msg = "Configuration file must define a mapping"
            raise ValueError(msg)
        return data

    def _coerce_option_value(self, key: str, value: Any) -> Any:
        path_keys = {"output_path"}
        int_keys = {
            "epochs",
            "batch_size",
            "num_workers",
            "hidden_dim",
            "prefetch_factor",
        }
        float_keys = {"learning_rate", "mask_ratio"}
        bool_keys = {"pin_memory", "progress_bar", "compilation"}
        if key in path_keys and value is not None:
            return Path(value)
        if key in int_keys and value is not None:
            return int(value)
        if key in float_keys and value is not None:
            return float(value)
        if key == "device" and value is not None:
            return str(value)
        if key in bool_keys and value is not None:
            return bool(value)
        return value

    def _resolve_output_path(
        self, output_path: Optional[Path]
    ) -> Path:
        if output_path is not None:
            return output_path
        return Path.cwd() / "masked_autoencoder_state.pt"

    def _resolve_device(
        self, device_str: Optional[str]
    ) -> torch.device:
        if device_str is not None:
            return torch.device(device_str)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _resolve_pin_memory(
        self, option: Optional[bool], device: torch.device
    ) -> bool:
        if option is not None:
            return option
        return device.type == "cuda"

    def _create_dataloader(
        self,
        *,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        prefetch_factor: int,
    ) -> DataLoader:
        persistent_workers = num_workers > 0
        worker_init_fn = (
            default_worker_init_fn if persistent_workers else None
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
            if persistent_workers
            else None,
            drop_last=False,
            timeout=120 if persistent_workers else 0,
            worker_init_fn=worker_init_fn,
        )
        self.logger.info("Training DataLoader: %d batches", len(dataloader))
        return dataloader

    def _run_epoch(
        self,
        *,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        mask_ratio: float,
        epoch_index: int,
        total_epochs: int,
        progress_bar: bool,
    ) -> float:
        model.train()
        total_loss = 0.0
        sample_count = 0
        progress = None
        log_interval: Optional[int] = None
        if progress_bar:
            progress = tqdm(
                enumerate(dataloader),
                desc=f"MAE Epoch {epoch_index + 1}/{total_epochs}",
                total=len(dataloader),
            )
            batch_iterable = progress
        else:
            batch_iterable = enumerate(dataloader)
            log_interval = max(1, len(dataloader) // 10)

        for batch_index, batch in batch_iterable:
            inputs = batch.to(device)
            mask = self._generate_mask(inputs, mask_ratio)
            masked_inputs = inputs.clone()
            masked_inputs[mask] = 0.0

            optimizer.zero_grad()
            reconstructions = model(masked_inputs)
            loss = self._compute_loss(
                reconstructions, inputs, mask, device
            )
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            loss_value = float(loss.item())
            total_loss += loss_value * batch_size
            sample_count += batch_size

            average_loss = total_loss / sample_count

            if progress is not None:
                progress.set_postfix(
                    loss=f"{loss_value:.6f}",
                    avg_loss=f"{average_loss:.6f}",
                )
            else:
                if (
                    log_interval is not None
                    and (batch_index + 1) % log_interval == 0
                ):
                    self.logger.info(
                        (
                            "Batch %d/%d - loss: %.6f (avg: %.6f)"
                        ),
                        batch_index + 1,
                        len(dataloader),
                        loss_value,
                        average_loss,
                    )

        if sample_count == 0:
            msg = "Dataloader returned zero samples"
            raise ValueError(msg)
        return total_loss / sample_count

    def _generate_mask(
        self, inputs: torch.Tensor, mask_ratio: float
    ) -> torch.Tensor:
        if mask_ratio <= 0.0:
            return torch.zeros_like(inputs, dtype=torch.bool)
        if mask_ratio >= 1.0:
            return torch.ones_like(inputs, dtype=torch.bool)
        probabilities = torch.rand_like(
            inputs, dtype=torch.float32
        )
        return probabilities < mask_ratio

    def _compute_loss(
        self,
        reconstructions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        mse = (reconstructions - targets) ** 2
        mask_float = mask.to(device=device, dtype=targets.dtype)
        if mask_float.sum().item() > 0:
            loss = (mse * mask_float).sum() / mask_float.sum()
        else:
            loss = mse.mean()
        return loss

    def _persist_encoder_state_dict(
        self, model: _MaskedAutoencoder, output_path: Path
    ) -> None:
        """Persist only the encoder parameters for downstream training."""

        encoder_state = {
            key: tensor.detach().cpu()
            for key, tensor in model.encoder.state_dict().items()
        }
        torch.save(encoder_state, output_path)


__all__ = ["MaskedAutoencoderPretraining"]
