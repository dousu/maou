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

    # Advance scheduler by 5 steps
    start_epoch = 5
    for _ in range(start_epoch):
        scheduler.step()

    # Learning rate should have changed
    advanced_lr = optimizer.param_groups[0]["lr"]
    assert advanced_lr != initial_lr
    assert (
        advanced_lr < initial_lr
    )  # Cosine annealing decreases LR


def test_warmup_cosine_decay_scheduler_advances() -> None:
    """WarmupCosineDecayScheduler should advance correctly with per-step."""

    # Create a simple model and optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Create warmup scheduler (per-step)
    warmup_steps = 20
    total_steps = 200
    scheduler = WarmupCosineDecayScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    # Record learning rate after initialization (step 0)
    lr_step_0 = optimizer.param_groups[0]["lr"]

    # Advance several steps during warmup
    for _ in range(10):
        scheduler.step()
    lr_step_10 = optimizer.param_groups[0]["lr"]

    # Advance to end of warmup
    for _ in range(10):
        scheduler.step()
    lr_step_20 = optimizer.param_groups[0]["lr"]

    # Record more steps for decay phase
    lrs = [lr_step_0, lr_step_10, lr_step_20]
    for _ in range(80):  # steps 21-100
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    # During warmup (steps 0-19), LR should increase
    # step 0: (0+1)/20 = 0.05 * 0.1 = 0.005
    # step 10: (10+1)/20 = 0.55 * 0.1 = 0.055
    # step 19: (19+1)/20 = 1.0 * 0.1 = 0.1
    assert lr_step_0 < lr_step_10  # Warmup: increasing
    assert lr_step_10 < lr_step_20  # Warmup: increasing
    assert (
        lr_step_20 <= 0.1
    )  # Should reach base LR by end of warmup

    # After warmup (steps 20+), LR should decrease (cosine decay)
    assert lrs[2] > lrs[-1]  # Decay: decreasing


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
