import torch


# Generalized Cross Entropy Loss (GCE) の定義
class GCELoss(torch.nn.Module):
    """
    targetsはラベルが入っている想定でそのインデックスでone-hotに変換して使う
    """

    # qはGCEのパラメータ (0 < q <= 1)
    def __init__(self, q: float = 0.7) -> None:
        super(GCELoss, self).__init__()
        self.q = q

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(
            targets, num_classes=probs.size(1)
        ).float()
        loss = (1 - probs**self.q) / self.q
        return (loss * targets_one_hot).sum(dim=1).mean()


class MaskedGCELoss(torch.nn.Module):
    """GCE Loss with mask.
    targetsには各ラベルの確立分布がはいっている．

    マスクされた値は0として扱うか，倍率をかけるかの戦略がある．

    0として扱うと学習フェーズにおいてはマスクされた値の評価は学習されない (無視される)．
    ```
    masked_loss = loss * targets * mask
    ```

    マスクされた値に倍率をかけておく方法はパラメータが増えるが，
    maskがルールのようなものから生成されていると学習させるのに役立ちそう．
    ```
    masked_loss = loss * targets * (1 + alpha * (1 - mask))
    ```

    ペナルティをかける方向にしておく．
    """

    # qはGCEのパラメータ (0 < q <= 1)
    def __init__(
        self, q: float = 0.7, alpha: float = 0.1
    ) -> None:
        super(MaskedGCELoss, self).__init__()
        self.q = q
        self.alpha = alpha

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # 数値的安定性のためにlogitsをクリップ
        logits = torch.clamp(logits, -100, 100)

        probs = torch.softmax(logits, dim=1)
        # 小さな値を追加して0除算を防ぐ
        probs = torch.clamp(probs, min=1e-7, max=1.0)

        loss = (1 - probs**self.q) / max(self.q, 1e-5)
        masked_loss = (
            loss * targets * (1 + self.alpha * (1 - mask))
        )
        return masked_loss.sum(dim=1).mean()
