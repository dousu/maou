import torch


# Generalized Cross Entropy Loss (GCE) の定義
class GCELoss(torch.nn.Module):
    def __init__(self, q=0.7):  # qはGCEのパラメータ (0 < q <= 1)
        super(GCELoss, self).__init__()
        self.q = q

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(
            targets, num_classes=probs.size(1)
        ).float()
        loss = (1 - probs**self.q) / self.q
        return (loss * targets_one_hot).sum(dim=1).mean()
