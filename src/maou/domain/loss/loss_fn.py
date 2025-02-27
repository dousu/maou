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


class MaskedGCELoss(torch.nn.Module):
    """GCE Loss with mask.
    マスクされた値は0として扱うか，倍率をかけるか．

    0として扱うと学習フェーズにおいてはマスクされた値の評価は学習されない (無視される)．
    ```
    masked_loss = loss * targets_one_hot * mask
    ```

    マスクされた値に倍率をかけておく方法はパラメータが増えるが，
    maskがルールのようなものから生成されていると学習させるのに役立ちそう．
    ```
    masked_loss = loss * targets_one_hot * (1 + alpha * (1 - mask))
    ```

    ペナルティをかける方向にしておく．
    """

    # qはGCEのパラメータ (0 < q <= 1)
    def __init__(self, q: float = 0.7, alpha: float = 0.5) -> None:
        super(MaskedGCELoss, self).__init__()
        self.q = q
        self.alpha = alpha

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(
            targets, num_classes=probs.size(1)
        ).float()
        loss = (1 - probs**self.q) / self.q
        masked_loss = loss * targets_one_hot * (1 + self.alpha * (1 - mask))
        return masked_loss.sum(dim=1).mean()
