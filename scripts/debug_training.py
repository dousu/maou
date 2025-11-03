"""Debug script to investigate training issues."""
import torch
import numpy as np
from pathlib import Path
from maou.app.learning.network import Network
from maou.app.learning.dataset import KifDataset
from maou.infra.file_system.file_data_source import FileDataSource
from torch.utils.data import DataLoader

def main():
    print("=" * 80)
    print("Training Debug Investigation")
    print("=" * 80)

    # Load a small sample of data
    data_dir = Path("preprocess/floodgate/2020")
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return

    # Get file paths
    file_paths = sorted(data_dir.glob("*.npy"))
    if not file_paths:
        print(f"Error: No .npy files found in {data_dir}")
        return

    print(f"\nFound {len(file_paths)} files")
    print(f"Using first file: {file_paths[0]}")

    # Load data source
    datasource = FileDataSource(
        file_paths=[file_paths[0]],
        array_type="preprocessing"
    )

    # Create dataset
    dataset = KifDataset(datasource=datasource)
    print(f"\nDataset size: {len(dataset)} samples")

    # Check value label distribution
    print("\n" + "=" * 80)
    print("Checking value label distribution...")
    print("=" * 80)

    # Sample 10000 points to check distribution
    sample_size = min(10000, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)

    value_labels = []

    for idx in indices[:1000]:  # Check first 1000 samples
        _, (_, value, _) = dataset[idx]
        value_labels.append(value.item())

    value_labels = np.array(value_labels)

    print(f"\nValue label statistics (n={len(value_labels)}):")
    print(f"  Mean: {value_labels.mean():.6f}")
    print(f"  Std:  {value_labels.std():.6f}")
    print(f"  Min:  {value_labels.min():.6f}")
    print(f"  Max:  {value_labels.max():.6f}")
    print(f"  Median: {np.median(value_labels):.6f}")

    # Check histogram
    print("\nValue label histogram:")
    hist, bins = np.histogram(value_labels, bins=10)
    for i in range(len(hist)):
        bar = "█" * int(hist[i] / hist.max() * 50)
        print(f"  [{bins[i]:.3f}, {bins[i+1]:.3f}): {hist[i]:4d} {bar}")

    # Test model predictions
    print("\n" + "=" * 80)
    print("Testing model predictions...")
    print("=" * 80)

    # Create a simple model
    model = Network(
        embed_dim=256,
        depth=4,
        num_heads=8,
        dropout_rate=0.1,
        policy_hidden_dim=None,
        value_hidden_dim=None,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"\nModel device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create a small dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=100,
        shuffle=False,
        num_workers=0,
    )

    # Get one batch
    batch = next(iter(dataloader))
    inputs, (labels_policy, labels_value, legal_move_mask) = batch

    inputs = inputs.to(device)
    labels_value = labels_value.to(device)

    print("\nBatch shapes:")
    print(f"  inputs: {inputs.shape}")
    print(f"  labels_value: {labels_value.shape}")
    print(f"  labels_value range: [{labels_value.min():.6f}, {labels_value.max():.6f}]")
    print(f"  labels_value mean: {labels_value.mean():.6f}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        policy_logits, value_pred = model(inputs)

    print("\nModel predictions:")
    print(f"  value_pred shape: {value_pred.shape}")
    print(f"  value_pred range: [{value_pred.min():.6f}, {value_pred.max():.6f}]")
    print(f"  value_pred mean: {value_pred.mean():.6f}")
    print(f"  value_pred std: {value_pred.std():.6f}")

    # Test loss computation
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(value_pred, labels_value)
    print(f"\nInitial value loss: {loss.item():.6f}")

    # Test with training mode
    print("\n" + "=" * 80)
    print("Testing gradient flow...")
    print("=" * 80)

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Single training step
    optimizer.zero_grad()
    policy_logits, value_pred = model(inputs)
    loss = loss_fn(value_pred, labels_value)
    loss.backward()

    # Check gradients
    value_head_grads = []
    for name, param in model.named_parameters():
        if "value_head" in name and param.grad is not None:
            grad_norm = param.grad.norm().item()
            value_head_grads.append(grad_norm)
            print(f"  {name}: grad_norm = {grad_norm:.6f}")

    if value_head_grads:
        print(f"\nValue head gradient norms: min={min(value_head_grads):.6f}, max={max(value_head_grads):.6f}")
    else:
        print("\nWARNING: No gradients found for value head!")

    # Test a few training steps
    print("\n" + "=" * 80)
    print("Testing short training run (10 steps)...")
    print("=" * 80)

    model.train()
    losses = []
    value_preds_over_time = []

    for step in range(10):
        optimizer.zero_grad()
        policy_logits, value_pred = model(inputs)
        loss = loss_fn(value_pred, labels_value)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        value_preds_over_time.append(value_pred.mean().item())

        if step % 3 == 0:
            print(f"  Step {step}: loss={loss.item():.6f}, pred_mean={value_pred.mean().item():.6f}")

    print("\nLoss over 10 steps:")
    print(f"  Initial: {losses[0]:.6f}")
    print(f"  Final:   {losses[-1]:.6f}")
    print(f"  Change:  {losses[-1] - losses[0]:.6f}")

    print("\nValue prediction mean over 10 steps:")
    print(f"  Initial: {value_preds_over_time[0]:.6f}")
    print(f"  Final:   {value_preds_over_time[-1]:.6f}")
    print(f"  Change:  {value_preds_over_time[-1] - value_preds_over_time[0]:.6f}")

    print("\n" + "=" * 80)
    print("Debug investigation complete")
    print("=" * 80)

if __name__ == "__main__":
    main()
