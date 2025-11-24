"""Tests for learning rate scheduler resumption."""

from __future__ import annotations

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from maou.app.learning.setup import WarmupCosineDecayScheduler


def test_scheduler_advances_with_start_epoch() -> None:
    """LR scheduler should advance when start_epoch > 0."""

    # Create a simple model and optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Create scheduler with known behavior
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    # Record initial learning rate
    initial_lr = optimizer.param_groups[0]["lr"]

    # Advance scheduler by 5 epochs
    start_epoch = 5
    for _ in range(start_epoch):
        scheduler.step()

    # Learning rate should have changed
    advanced_lr = optimizer.param_groups[0]["lr"]
    assert advanced_lr != initial_lr
    assert advanced_lr < initial_lr  # Cosine annealing decreases LR


def test_warmup_cosine_decay_scheduler_advances() -> None:
    """WarmupCosineDecayScheduler should advance correctly."""

    # Create a simple model and optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Create warmup scheduler
    max_epochs = 20
    warmup_epochs = 2
    scheduler = WarmupCosineDecayScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
    )

    # Record learning rates at different epochs
    lrs = [optimizer.param_groups[0]["lr"]]

    # Advance scheduler through warmup and decay phases
    for epoch in range(10):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    # During warmup (epochs 0-1), LR should increase
    assert lrs[1] < lrs[2]  # Warmup: increasing

    # After warmup (epochs 2+), LR should decrease (cosine decay)
    assert lrs[5] > lrs[10]  # Decay: decreasing


def test_scheduler_state_after_multiple_steps() -> None:
    """Verify scheduler state is consistent after multiple steps."""

    # Create two identical setups
    model1 = torch.nn.Linear(10, 10)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.1)
    scheduler1 = CosineAnnealingLR(optimizer1, T_max=10)

    model2 = torch.nn.Linear(10, 10)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=10)

    # Advance first scheduler step by step
    for _ in range(5):
        scheduler1.step()

    lr_step_by_step = optimizer1.param_groups[0]["lr"]

    # Advance second scheduler all at once (simulating resume)
    for _ in range(5):
        scheduler2.step()

    lr_bulk = optimizer2.param_groups[0]["lr"]

    # Both should have the same learning rate
    assert lr_step_by_step == lr_bulk
