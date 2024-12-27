import torch


# Generalized Cross Entropy Loss (GCE) の定義
class GCELoss(torch.nn.Module):
    # qはGCEのパラメータ (0 < q <= 1)
    def __init__(self, q: float = 0.7) -> None:
        super(GCELoss, self).__init__()
        self.q = q

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(
            targets, num_classes=probs.size(1)
        ).float()
        loss = (1 - probs**self.q) / self.q
        return (loss * targets_one_hot).sum(dim=1).mean()
