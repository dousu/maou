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
        self, q: float = 0.7, alpha: float = 0.2
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


class GCEwithNegativePenaltyLoss(torch.nn.Module):
    def __init__(
        self,
        q: float = 0.7,
        alpha: float = 0.1,
        negative_weight: float = 0.01,
    ) -> None:
        super(GCEwithNegativePenaltyLoss, self).__init__()
        self.q = q
        self.alpha = alpha
        self.negative_weight = negative_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # 数値的安定性のためにlogitsをクリップ
        logits = torch.clamp(logits, -100, 100)

        probs = torch.softmax(logits, dim=1)
        probs = torch.clamp(probs, min=1e-7, max=1.0)

        positive_loss = (1 - torch.pow(probs, self.q)) / max(
            self.q, 1e-5
        )
        positive_term = positive_loss * targets

        negative_term = (
            probs * (1 - targets) * self.negative_weight
        )

        # 自然にmaskが0のところにはnegative_termのみがはいっているはず
        total_loss = (positive_term + negative_term) * (
            1 + self.alpha * (1 - mask)
        )

        return total_loss.sum(dim=1).mean()


class ReachableSquaresLoss(torch.nn.Module):
    """Binary cross-entropy loss for reachable squares prediction.

    This loss function is designed for Stage 1 training where the model
    learns which board squares pieces can move to (9×9 binary output).

    Uses BCEWithLogitsLoss which combines sigmoid activation and BCE loss
    for numerical stability，especially important for mixed precision training.

    The pos_weight parameter can be used to handle class imbalance，as there
    are typically more unreachable squares than reachable ones.
    """

    def __init__(
        self, pos_weight: float = 1.0, reduction: str = "mean"
    ) -> None:
        """Initialize ReachableSquaresLoss.

        Args:
            pos_weight: Weight for positive class (reachable squares).
                Values > 1.0 increase recall，< 1.0 increase precision.
                Default: 1.0 (balanced)
            reduction: Reduction method ('mean'，'sum'，or 'none').
                Default: 'mean'
        """
        super(ReachableSquaresLoss, self).__init__()
        self.pos_weight = torch.tensor([pos_weight])
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute binary cross-entropy loss for reachable squares.

        Args:
            logits: Predicted logits (batch，81) - raw scores before sigmoid
            targets: Target binary labels (batch，81) with values 0 or 1

        Returns:
            Scalar loss value (if reduction='mean' or 'sum')
            or per-element loss (batch，81) if reduction='none'
        """
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(logits.device),
            reduction=self.reduction,
        )
        return loss_fn(logits, targets)


class LegalMovesLoss(torch.nn.Module):
    """Multi-label binary cross-entropy loss for legal moves prediction.

    This loss function is designed for Stage 2 training where the model
    learns which moves are legal in a given position (MOVE_LABELS_NUM binary outputs).

    Unlike policy loss (multi-class classification with softmax)，this is
    multi-label classification where multiple moves can be legal simultaneously.
    Each move is treated as an independent binary classification problem.

    Uses BCEWithLogitsLoss for numerical stability and mixed precision compatibility.
    """

    def __init__(
        self, pos_weight: float = 1.0, reduction: str = "mean"
    ) -> None:
        """Initialize LegalMovesLoss.

        Args:
            pos_weight: Weight for positive class (legal moves).
                Values > 1.0 increase recall，< 1.0 increase precision.
                Default: 1.0 (balanced)
            reduction: Reduction method ('mean'，'sum'，or 'none').
                Default: 'mean'
        """
        super(LegalMovesLoss, self).__init__()
        self.pos_weight = torch.tensor([pos_weight])
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute multi-label binary cross-entropy loss for legal moves.

        Args:
            logits: Predicted logits (batch，MOVE_LABELS_NUM) - raw scores before sigmoid
            targets: Target binary labels (batch，MOVE_LABELS_NUM) with values 0 or 1

        Returns:
            Scalar loss value (if reduction='mean' or 'sum')
            or per-element loss (batch，MOVE_LABELS_NUM) if reduction='none'
        """
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(logits.device),
            reduction=self.reduction,
        )
        return loss_fn(logits, targets)
