"""Tests for the learning module utilities."""

from __future__ import annotations

from typing import Mapping, cast

import pytest
import torch

from maou.app.learning.dl import Learning
from maou.app.learning.network import HeadlessNetwork, Network


class _DummyCompiledModule(torch.nn.Module):
    """Minimal stand-in for ``torch.compile`` wrapped modules."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self._orig_mod = module

    def forward(
        self, *args: torch.Tensor, **kwargs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._orig_mod(*args, **kwargs)


def _assert_state_dict_equality(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
) -> None:
    for key, tensor in first.items():
        assert torch.equal(tensor, second[key])


def test_load_resume_state_dict_adds_prefix_for_compiled_models() -> None:
    """State dicts saved before compilation can be loaded into compiled models."""

    source_model = Network()
    target_model = Network()
    compiled_model = _DummyCompiledModule(target_model)

    learning = Learning()
    learning.model = cast(Network, compiled_model)

    state_dict = source_model.state_dict()

    learning._load_resume_state_dict(state_dict)

    _assert_state_dict_equality(state_dict, target_model.state_dict())


def test_load_resume_state_dict_without_compilation() -> None:
    """Pure PyTorch modules continue to load state dicts without modification."""

    source_model = Network()
    target_model = Network()

    learning = Learning()
    learning.model = target_model

    state_dict = source_model.state_dict()

    learning._load_resume_state_dict(state_dict)

    _assert_state_dict_equality(state_dict, target_model.state_dict())


def test_load_resume_state_dict_requires_complete_state() -> None:
    """Checkpoints missing head weights should raise a descriptive error."""

    source_model = HeadlessNetwork()
    target_model = Network()

    learning = Learning()
    learning.model = target_model

    state_dict = source_model.state_dict()

    with pytest.raises(RuntimeError):
        learning._load_resume_state_dict(state_dict)
