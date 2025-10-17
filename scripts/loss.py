import torch

from maou.domain.loss.loss_fn import MaskedGCELoss


class CrossEntropy(torch.nn.Module):
    def __init__(
        self, q: float = 0.7, alpha: float = 0.1
    ) -> None:
        super(CrossEntropy, self).__init__()
        self.q = q
        self.alpha = alpha

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        probs = torch.clamp(probs, min=1e-7, max=1.0)
        loss = (1 - torch.pow(probs, self.q)) / self.q
        weighted_loss = loss * targets
        weighted_loss = weighted_loss * (
            1 + self.alpha * (1 - mask)
        )
        return weighted_loss.sum(dim=1).mean()


class KLDivergence(torch.nn.Module):
    def __init__(
        self, q: float = 0.7, alpha: float = 0.1
    ) -> None:
        super(KLDivergence, self).__init__()
        self.q = q
        self.alpha = alpha

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        if self.q < 1:
            gce_term = (1 - torch.pow(probs, self.q)) / self.q
        else:
            gce_term = -log_probs
        loss = gce_term * targets
        weighted_loss = loss * (1 + self.alpha * (1 - mask))

        return weighted_loss.sum(dim=1).mean()


class GCEwithNegativePenaltyLoss(torch.nn.Module):
    def __init__(
        self,
        q: float = 0.7,
        alpha: float = 0.1,
        negative_weight: float = 0.01,
    ):
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


logits1 = torch.tensor(
    [[2.0, 1.0, 0.5, 0.3, 0.2]], requires_grad=True
)
print(torch.softmax(logits1, dim=1))
logits2 = torch.tensor(
    [[2.0, 1.0, 0.5, 0.3, 0.2]], requires_grad=True
)
# print(torch.softmax(logits2, dim=1))
logits3 = torch.tensor(
    [[2.0, 1.0, 0.5, 0.3, 0.2]], requires_grad=True
)
# print(torch.softmax(logits3, dim=1))
logits4 = torch.tensor(
    [[2.0, 1.0, 0.5, 0.3, 0.2]], requires_grad=True
)
# print(torch.softmax(logits4, dim=1))
targets = torch.tensor([[0.5, 0.25, 0.25, 0.0, 0.0]])
print(targets)
mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
print(mask)

loss_fn1 = MaskedGCELoss()
loss_fn2 = CrossEntropy()
loss_fn3 = KLDivergence()
loss_fn4 = GCEwithNegativePenaltyLoss()

loss1 = loss_fn1(logits1, targets, mask)
print(f"Loss1 (MaskedGCELoss): {loss1}")
loss2 = loss_fn2(logits2, targets, mask)
print(f"Loss2 (CrossEntropy): {loss2}")
loss3 = loss_fn3(logits3, targets, mask)
print(f"Loss3 (KLDivergence): {loss3}")
loss4 = loss_fn4(logits4, targets, mask)
print(f"Loss4 (GCEwithNegativePenalty): {loss4}")

loss1.backward()
print(f"Grad1 (MaskedGCELoss): {logits1.grad}")
loss2.backward()
print(f"Grad2 (CrossEntropy): {logits2.grad}")
loss3.backward()
print(f"Grad3 (KLDivergence): {logits3.grad}")
loss4.backward()
print(f"Grad4 (GCEwithNegativePenalty): {logits4.grad}")

# 勾配の比較（targets=0の位置の勾配を確認）
print("\n=== 勾配の比較（targets=0の位置） ===")
print(
    f"MaskedGCELoss - 位置0: {logits1.grad[0, 0].item():.6f}, 位置2: {logits1.grad[0, 2].item():.6f}"  # type: ignore
)
print(
    f"CrossEntropy - 位置0: {logits2.grad[0, 0].item():.6f}, 位置2: {logits2.grad[0, 2].item():.6f}"  # type: ignore
)
print(
    f"KLDivergence - 位置0: {logits3.grad[0, 0].item():.6f}, 位置2: {logits3.grad[0, 2].item():.6f}"  # type: ignore
)
print(
    f"GCEwithNegativePenaltyLoss - 位置0: {logits4.grad[0, 0].item():.6f}, 位置2: {logits4.grad[0, 2].item():.6f}"  # type: ignore
)
