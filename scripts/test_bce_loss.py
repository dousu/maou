"""Test BCE Loss implementation for value head."""
import torch
from maou.app.learning.network import Network
from maou.app.learning.setup import LossOptimizerFactory

def main():
    print("=" * 80)
    print("Testing BCE Loss Implementation")
    print("=" * 80)

    # Create model
    model = Network(
        embed_dim=256,
        depth=4,
        num_heads=8,
        dropout_rate=0.1,
    )

    # Create loss functions
    loss_fn_policy, loss_fn_value = LossOptimizerFactory.create_loss_functions(
        gce_parameter=0.7
    )

    print(f"\nValue loss function: {loss_fn_value.__class__.__name__}")
    print("Expected: BCELoss")

    # Create sample data
    batch_size = 100
    inputs = torch.randn(batch_size, 104, 9, 9)

    # Simulate bimodal distribution (0s and 1s)
    labels_value = torch.zeros(batch_size, 1)
    labels_value[:50] = 0.0  # 50% losses
    labels_value[50:] = 1.0  # 50% wins

    print("\nLabel distribution:")
    print(f"  Zeros (loss): {(labels_value == 0.0).sum().item()}")
    print(f"  Ones (win): {(labels_value == 1.0).sum().item()}")
    print(f"  Mean: {labels_value.mean().item():.4f}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        policy_logits, value_pred = model(inputs)

    print("\nValue predictions:")
    print(f"  Range: [{value_pred.min().item():.4f}, {value_pred.max().item():.4f}]")
    print(f"  Mean: {value_pred.mean().item():.4f}")
    print(f"  Std: {value_pred.std().item():.4f}")

    # Test loss computation with BCE
    loss = loss_fn_value(value_pred, labels_value)
    print(f"\nBCE Loss: {loss.item():.6f}")

    # Compare with MSE for reference
    mse_loss = torch.nn.MSELoss()(value_pred, labels_value)
    print(f"MSE Loss (reference): {mse_loss.item():.6f}")

    # Test gradient flow
    print("\n" + "=" * 80)
    print("Testing gradient flow with BCE Loss")
    print("=" * 80)

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    losses_bce = []
    pred_means = []

    for step in range(10):
        optimizer.zero_grad()
        policy_logits, value_pred = model(inputs)
        loss = loss_fn_value(value_pred, labels_value)
        loss.backward()
        optimizer.step()

        losses_bce.append(loss.item())
        pred_means.append(value_pred.mean().item())

        if step % 3 == 0:
            print(f"  Step {step}: loss={loss.item():.6f}, pred_mean={value_pred.mean().item():.4f}")

    print("\nLoss progression:")
    print(f"  Initial: {losses_bce[0]:.6f}")
    print(f"  Final:   {losses_bce[-1]:.6f}")
    print(f"  Change:  {losses_bce[-1] - losses_bce[0]:.6f}")

    print("\nPrediction mean progression:")
    print(f"  Initial: {pred_means[0]:.4f}")
    print(f"  Final:   {pred_means[-1]:.4f}")
    print(f"  Change:  {pred_means[-1] - pred_means[0]:.4f}")

    # Expected behavior
    print("\n" + "=" * 80)
    print("Expected Behavior")
    print("=" * 80)
    print("✓ Loss should decrease significantly")
    print("✓ Predictions should NOT converge to 0.5")
    print("✓ Predictions should learn the bimodal pattern")

    # Verification
    print("\n" + "=" * 80)
    print("Verification")
    print("=" * 80)

    if losses_bce[-1] < losses_bce[0] * 0.9:
        print("✓ Loss decreased by >10%")
    else:
        print("✗ Loss did not decrease enough")

    if abs(pred_means[-1] - 0.5) > 0.05:
        print(f"✓ Predictions moved away from 0.5 (mean={pred_means[-1]:.4f})")
    else:
        print(f"✗ Predictions stuck near 0.5 (mean={pred_means[-1]:.4f})")

    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)

if __name__ == "__main__":
    main()
