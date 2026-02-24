"""Protocol for backbone models that support layer-wise freezing."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import nn


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
