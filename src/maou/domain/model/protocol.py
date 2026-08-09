"""Protocol for backbone models that support layer-wise freezing."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor, nn


@runtime_checkable
class FreezableBackbone(Protocol):
    """Backbone that exposes freezable layer groups ordered from low to high."""

    def get_freezable_groups(self) -> list[nn.Module]:
        """Return freezable layer groups ordered from lowest to highest.

        Returns:
            list[nn.Module]: Layer groups in low-to-high order.
                Freezing proceeds from the first element (lowest layer).
        """
        ...

    def get_output_norm(self) -> nn.Module | None:
        """Return the final normalization module applied after all groups.

        This module is unfrozen whenever any encoder groups are trainable,
        since it must adapt to distribution shifts from trainable blocks.

        Returns:
            nn.Module | None: The output normalization module, or None
                if the backbone has no such module.
        """
        ...

    def preprocess_for_blocks(self, x: Tensor) -> Tensor:
        """Return the tensor in the shape the freezable groups consume.

        Callers that run a truncated prefix of ``get_freezable_groups()``
        must apply this first; the groups themselves do not perform it.

        Args:
            x: Backbone input tensor.

        Returns:
            Tensor: Input reshaped for the freezable groups. Backbones
                whose groups accept the raw input return ``x`` unchanged.
        """
        ...

    def forward_features(self, x: Tensor) -> Tensor:
        """Return pooled features for each sample, without any task head.

        Args:
            x: Backbone input tensor.

        Returns:
            Tensor: Pooled features shaped ``(batch, embedding_dim)``.
        """
        ...
