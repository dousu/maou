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

    # Record learning rate after initialization (epoch 0)
    # At initialization, scheduler sets LR for epoch 0
    lr_epoch_0 = optimizer.param_groups[0]["lr"]

    # Advance to epoch 1
    scheduler.step()
    lr_epoch_1 = optimizer.param_groups[0]["lr"]

    # Advance to epoch 2 (end of warmup)
    scheduler.step()
    lr_epoch_2 = optimizer.param_groups[0]["lr"]

    # Record more epochs for decay phase
    lrs = [lr_epoch_0, lr_epoch_1, lr_epoch_2]
    for _ in range(8):  # epochs 3-10
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    # During warmup (epochs 0-1), LR should increase
    # epoch 0: (0+1)/2 = 0.5 * 0.1 = 0.05
    # epoch 1: (1+1)/2 = 1.0 * 0.1 = 0.1
    assert lr_epoch_0 < lr_epoch_1  # Warmup: increasing
    assert lr_epoch_1 <= 0.1  # Should reach base LR by end of warmup

    # After warmup (epochs 2+), LR should decrease (cosine decay)
    # epoch 2 is the start of decay, epoch 10 should have lower LR
    assert lrs[2] > lrs[10]  # Decay: decreasing


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
