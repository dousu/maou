import logging

import torch
import torch.nn.functional as F


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
        self._pos_weight = torch.tensor([pos_weight])
        self._reduction = reduction
        self._loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]),
            reduction=reduction,
        )

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
        self._loss_fn.pos_weight = self._pos_weight.to(
            logits.device
        )
        return self._loss_fn(logits, targets)


class AsymmetricLoss(torch.nn.Module):
    """Asymmetric Loss for multi-label classification with extreme class imbalance.

    正例と負例に独立したfocusing parameterを使用し，
    マルチラベル分類における極端なクラス不均衡に対処する．

    将棋の合法手予測(1496ラベル中~20正例)のような
    extreme multi-label imbalance に特に有効．

    γ+=0, γ-=0, clip=0.0 の場合は標準BCEWithLogitsLossと同等の動作．

    References:
        Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 2.0,
        clip: float = 0.02,
        reduction: str = "mean",
    ) -> None:
        """Initialize AsymmetricLoss.

        Args:
            gamma_pos: 正例のfocusing parameter．
                0 = 正例の損失を一切軽視しない(推奨)．
                デフォルト: 0.0
            gamma_neg: 負例のfocusing parameter．
                大きいほど容易な負例の損失を強く抑制．
                推奨範囲: 1.0-4.0，デフォルト: 2.0
            clip: 負例確率のクリッピングマージン．
                確率が clip 未満の負例を完全に無視する．
                推奨範囲: 0.0-0.05，デフォルト: 0.02
            reduction: 損失の集約方法 ('mean', 'sum', 'none')．
                デフォルト: 'mean'
        """
        super().__init__()
        if gamma_pos < 0:
            raise ValueError(
                f"gamma_pos must be non-negative, got {gamma_pos}"
            )
        if gamma_neg < 0:
            raise ValueError(
                f"gamma_neg must be non-negative, got {gamma_neg}"
            )
        if clip < 0:
            raise ValueError(
                f"clip must be non-negative, got {clip}"
            )
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute asymmetric loss.

        Args:
            logits: 予測logits (batch, num_labels) - sigmoid前の生スコア
            targets: ターゲット二値ラベル (batch, num_labels)，値は0または1

        Returns:
            reduction='mean'または'sum'の場合はスカラー損失値，
            reduction='none'の場合は要素ごとの損失 (batch, num_labels)

        Note:
            正例log確率に ``F.logsigmoid`` を使用し，極端なlogitsでも
            勾配 ≈ ±1 を維持する(``BCEWithLogitsLoss`` 同等の数値安定性)．
            Focusing weight は**クリップ済み確率**から計算される．
            これにより clip で抑制された easy negative は focusing weight も 0 となり，
            完全に損失から除外される(Alibaba-MIIL/ASL 参照実装準拠)．
        """
        # FP32にキャストして数値安定性を確保 (AMP対応)
        logits = logits.float()
        targets = targets.float()

        # 数値安定な log-sigmoid を使用 (log-sum-exp trick)
        # F.logsigmoid(x) = -softplus(-x) は極端なlogitsでも勾配を維持
        log_probs_pos = F.logsigmoid(logits)

        # 負例log確率: clip有無で計算方法を分岐
        if self.clip > 0:
            # clip > 0: sigmoid(-x) + clip で下限保証 (probs_neg ≥ clip)
            probs_neg = (
                torch.sigmoid(-logits) + self.clip
            ).clamp(max=1.0)
            log_probs_neg = torch.log(probs_neg)
        else:
            # clip = 0: F.logsigmoid(-x) で数値安定に計算
            log_probs_neg = F.logsigmoid(-logits)

        # Positive loss with focusing
        if self.gamma_pos > 0:
            probs_pos = torch.sigmoid(logits)
            pos_weight = (1 - probs_pos) ** self.gamma_pos
            loss_pos = targets * pos_weight * log_probs_pos
        else:
            loss_pos = targets * log_probs_pos

        # Negative loss with focusing (クリップ済み確率で focusing weight を計算)
        if self.gamma_neg > 0:
            if self.clip > 0:
                # probs_neg はクリップ済み → (1 - probs_neg) で focusing weight
                one_minus_probs_neg = (1.0 - probs_neg).clamp(
                    min=0
                )
            else:
                one_minus_probs_neg = torch.sigmoid(logits)
            neg_weight = one_minus_probs_neg**self.gamma_neg
            loss_neg = (
                (1 - targets) * neg_weight * log_probs_neg
            )
        else:
            loss_neg = (1 - targets) * log_probs_neg

        loss = -(loss_pos + loss_neg)

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LegalMovesLoss(torch.nn.Module):
    """Multi-label binary cross-entropy loss for legal moves prediction.

    This loss function is designed for Stage 2 training where the model
    learns which moves are legal in a given position (MOVE_LABELS_NUM binary outputs).

    クラス不均衡対策としてAsymmetric Loss (ASL)をサポート．
    1496ラベル中平均~20個が正例(1.3%)という極端な不均衡に対処する．

    gamma_neg=0.0 かつ clip=0.0 の場合は標準BCEWithLogitsLossと同等の動作．
    """

    def __init__(
        self,
        pos_weight: float = 1.0,
        reduction: str = "mean",
        gamma_pos: float = 0.0,
        gamma_neg: float = 0.0,
        clip: float = 0.0,
    ) -> None:
        """Initialize LegalMovesLoss.

        Args:
            pos_weight: Weight for positive class (legal moves).
                Values > 1.0 increase recall，< 1.0 increase precision.
                Default: 1.0 (balanced).
                ASL使用時は通常1.0のまま(ASLが不均衡を処理するため)．
            reduction: Reduction method ('mean'，'sum'，or 'none').
                Default: 'mean'
            gamma_pos: ASL positive focusing parameter.
                0.0 = 正例損失を軽視しない (default/recommended).
            gamma_neg: ASL negative focusing parameter.
                0.0 = standard BCE (default)，2.0 = recommended for imbalanced.
            clip: ASL negative probability clipping margin.
                0.0 = no clipping (default)，0.02 = recommended.
        """
        super().__init__()
        self._use_asl = (
            gamma_neg > 0.0 or gamma_pos > 0.0 or clip > 0.0
        )

        if self._use_asl and pos_weight != 1.0:
            logger = logging.getLogger(__name__)
            logger.warning(
                "pos_weight=%.1f is ignored when ASL is active "
                "(gamma_neg=%.1f, gamma_pos=%.1f, clip=%.3f). "
                "ASL handles class imbalance internally.",
                pos_weight,
                gamma_neg,
                gamma_pos,
                clip,
            )

        if self._use_asl:
            self._loss_fn: torch.nn.Module = AsymmetricLoss(
                gamma_pos=gamma_pos,
                gamma_neg=gamma_neg,
                clip=clip,
                reduction=reduction,
            )
        else:
            # ASL無効時は標準BCEを使用 (性能最適化パス)
            self._pos_weight = torch.tensor([pos_weight])
            self._reduction = reduction
            self._loss_fn = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight]),
                reduction=reduction,
            )

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
        if self._use_asl:
            return self._loss_fn(logits, targets)

        # Standard BCE path: pos_weight のデバイス同期が必要
        self._loss_fn.pos_weight = self._pos_weight.to(
            logits.device
        )
        return self._loss_fn(logits, targets)
