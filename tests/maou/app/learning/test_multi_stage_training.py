"""Tests for multi-stage training loop with HeadlessNetwork backbone."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from maou.app.learning.multi_stage_training import (
    SingleStageTrainingLoop,
    StageConfig,
    TrainingStage,
)
from maou.app.learning.network import (
    PIECES_IN_HAND_VECTOR_SIZE,
    HeadlessNetwork,
    LegalMovesHead,
    ReachableSquaresHead,
)


def _make_dummy_dataloader(
    *,
    batch_size: int = 4,
    num_samples: int = 8,
    board_vocab_size: int = 32,
    board_size: tuple[int, int] = (9, 9),
    target_dim: int = 81,
    hand_dim: int = PIECES_IN_HAND_VECTOR_SIZE,
) -> DataLoader:
    """ダミーの (board_tensor, hand_tensor), target データローダーを作成する．

    Stage 1/2 のデータセットは ((board_tensor, hand_tensor), target) を返す．
    """
    board_tensor = torch.randint(
        0, board_vocab_size, (num_samples, *board_size)
    )
    hand_tensor = torch.randn(num_samples, hand_dim)
    targets = torch.zeros(num_samples, target_dim)

    dataset = TensorDataset(board_tensor, hand_tensor, targets)

    def collate_fn(
        batch: list[tuple[torch.Tensor, ...]],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        boards = torch.stack([b[0] for b in batch])
        hands = torch.stack([b[1] for b in batch])
        tgts = torch.stack([b[2] for b in batch])
        return (boards, hands), tgts

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )


class TestSingleStageTrainingLoopWithHeadlessNetwork:
    """_train_epoch が HeadlessNetwork で正常に動作するテスト．"""

    def test_train_epoch_stage1_completes(self) -> None:
        """Stage 1: _train_epoch が HeadlessNetwork + ReachableSquaresHead で完走する．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim
        )
        dataloader = _make_dummy_dataloader(
            board_vocab_size=32, target_dim=81
        )

        config = StageConfig(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=1,
            accuracy_threshold=0.99,
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            optimizer=torch.optim.Adam(
                list(backbone.parameters())
                + list(head.parameters()),
                lr=1e-3,
            ),
        )

        loop = SingleStageTrainingLoop(
            model=backbone,
            head=head,
            device=torch.device("cpu"),
            config=config,
        )

        loss, accuracy = loop._train_epoch(0)

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert loss > 0.0
        assert 0.0 <= accuracy <= 1.0

    def test_train_epoch_stage2_completes(self) -> None:
        """Stage 2: _train_epoch が HeadlessNetwork + LegalMovesHead で完走する．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        from maou.domain.move.label import MOVE_LABELS_NUM

        head = LegalMovesHead(input_dim=backbone.embedding_dim)
        dataloader = _make_dummy_dataloader(
            board_vocab_size=32, target_dim=MOVE_LABELS_NUM
        )

        config = StageConfig(
            stage=TrainingStage.LEGAL_MOVES,
            max_epochs=1,
            accuracy_threshold=0.99,
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            optimizer=torch.optim.Adam(
                list(backbone.parameters())
                + list(head.parameters()),
                lr=1e-3,
            ),
        )

        loop = SingleStageTrainingLoop(
            model=backbone,
            head=head,
            device=torch.device("cpu"),
            config=config,
        )

        loss, accuracy = loop._train_epoch(0)

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert loss > 0.0
        assert 0.0 <= accuracy <= 1.0

    def test_forward_features_receives_model_inputs(
        self,
    ) -> None:
        """forward_features が ModelInputs (テンソルまたはタプル) を受け取ることを検証する．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )

        board_tensor = torch.randint(0, 32, (2, 9, 9))
        features = backbone.forward_features(board_tensor)

        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == 2
        assert features.shape[1] == backbone.embedding_dim

    def test_run_stage1_e2e(self) -> None:
        """Stage 1 E2E: SingleStageTrainingLoop.run が StageResult を返す．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim
        )
        dataloader = _make_dummy_dataloader(
            board_vocab_size=32, target_dim=81
        )

        config = StageConfig(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=2,
            accuracy_threshold=0.99,
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            optimizer=torch.optim.Adam(
                list(backbone.parameters())
                + list(head.parameters()),
                lr=1e-3,
            ),
        )

        loop = SingleStageTrainingLoop(
            model=backbone,
            head=head,
            device=torch.device("cpu"),
            config=config,
        )

        result = loop.run()

        assert result.stage == TrainingStage.REACHABLE_SQUARES
        assert result.epochs_trained >= 1
        assert isinstance(result.achieved_accuracy, float)
        assert isinstance(result.final_loss, float)

    def test_run_stage2_e2e(self) -> None:
        """Stage 2 E2E: SingleStageTrainingLoop.run が StageResult を返す．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        from maou.domain.move.label import MOVE_LABELS_NUM

        head = LegalMovesHead(input_dim=backbone.embedding_dim)
        dataloader = _make_dummy_dataloader(
            board_vocab_size=32, target_dim=MOVE_LABELS_NUM
        )

        config = StageConfig(
            stage=TrainingStage.LEGAL_MOVES,
            max_epochs=2,
            accuracy_threshold=0.99,
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            optimizer=torch.optim.Adam(
                list(backbone.parameters())
                + list(head.parameters()),
                lr=1e-3,
            ),
        )

        loop = SingleStageTrainingLoop(
            model=backbone,
            head=head,
            device=torch.device("cpu"),
            config=config,
        )

        result = loop.run()

        assert result.stage == TrainingStage.LEGAL_MOVES
        assert result.epochs_trained >= 1
        assert isinstance(result.achieved_accuracy, float)
        assert isinstance(result.final_loss, float)


class TestSingleStageTrainingLoopWithHandProjection:
    """hand_projection_dim > 0 の HeadlessNetwork で Stage 1/2 が動作するテスト．"""

    def test_train_epoch_stage1_with_hand_projection(
        self,
    ) -> None:
        """Stage 1: hand_projection_dim > 0 で _train_epoch が完走する．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=16,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim
        )
        dataloader = _make_dummy_dataloader(
            board_vocab_size=32, target_dim=81
        )

        config = StageConfig(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=1,
            accuracy_threshold=0.99,
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            optimizer=torch.optim.Adam(
                list(backbone.parameters())
                + list(head.parameters()),
                lr=1e-3,
            ),
        )

        loop = SingleStageTrainingLoop(
            model=backbone,
            head=head,
            device=torch.device("cpu"),
            config=config,
        )

        loss, accuracy = loop._train_epoch(0)

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert loss > 0.0
        assert 0.0 <= accuracy <= 1.0

    def test_train_epoch_stage2_with_hand_projection(
        self,
    ) -> None:
        """Stage 2: hand_projection_dim > 0 で _train_epoch が完走する．"""
        from maou.domain.move.label import MOVE_LABELS_NUM

        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=16,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        head = LegalMovesHead(input_dim=backbone.embedding_dim)
        dataloader = _make_dummy_dataloader(
            board_vocab_size=32, target_dim=MOVE_LABELS_NUM
        )

        config = StageConfig(
            stage=TrainingStage.LEGAL_MOVES,
            max_epochs=1,
            accuracy_threshold=0.99,
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            optimizer=torch.optim.Adam(
                list(backbone.parameters())
                + list(head.parameters()),
                lr=1e-3,
            ),
        )

        loop = SingleStageTrainingLoop(
            model=backbone,
            head=head,
            device=torch.device("cpu"),
            config=config,
        )

        loss, accuracy = loop._train_epoch(0)

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert loss > 0.0
        assert 0.0 <= accuracy <= 1.0

    def test_run_stage1_e2e_with_hand_projection(
        self,
    ) -> None:
        """Stage 1 E2E: hand_projection_dim > 0 で StageResult を返す．"""
        backbone = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=16,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        head = ReachableSquaresHead(
            input_dim=backbone.embedding_dim
        )
        dataloader = _make_dummy_dataloader(
            board_vocab_size=32, target_dim=81
        )

        config = StageConfig(
            stage=TrainingStage.REACHABLE_SQUARES,
            max_epochs=2,
            accuracy_threshold=0.99,
            dataloader=dataloader,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            optimizer=torch.optim.Adam(
                list(backbone.parameters())
                + list(head.parameters()),
                lr=1e-3,
            ),
        )

        loop = SingleStageTrainingLoop(
            model=backbone,
            head=head,
            device=torch.device("cpu"),
            config=config,
        )

        result = loop.run()

        assert result.stage == TrainingStage.REACHABLE_SQUARES
        assert result.epochs_trained >= 1
        assert isinstance(result.achieved_accuracy, float)
        assert isinstance(result.final_loss, float)
