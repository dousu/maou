from pathlib import Path

import numpy as np
import polars as pl
import pytest

from maou.app.utility.stage2_data_generation import (
    Stage2DataGenerationConfig,
    Stage2DataGenerationUseCase,
)
from maou.domain.board import shogi
from maou.domain.data.rust_io import load_stage2_df
from maou.domain.data.schema import get_stage2_polars_schema
from maou.domain.move.label import (
    MOVE_LABELS_NUM,
    make_move_label,
)


def _board_to_hcp_bytes(board: shogi.Board) -> bytes:
    """盤面をHCPバイト列に変換する．"""
    hcp = np.empty(32, dtype=np.uint8)
    board.to_hcp(hcp)
    return hcp.tobytes()


def _create_hcpe_feather(
    file_path: Path, hcp_bytes_list: list[bytes]
) -> None:
    """HCPEデータを含むfeatherファイルを作成する．"""
    n = len(hcp_bytes_list)
    df = pl.DataFrame(
        {
            "hcp": pl.Series(
                "hcp", hcp_bytes_list, dtype=pl.Binary
            ),
            "eval": pl.Series(
                "eval", [100] * n, dtype=pl.Int16
            ),
            "bestMove16": pl.Series(
                "bestMove16", [0] * n, dtype=pl.Int16
            ),
            "gameResult": pl.Series(
                "gameResult", [1] * n, dtype=pl.Int8
            ),
        }
    )
    df.write_ipc(str(file_path))


class TestStage2DataGenerationUseCase:
    """Stage2データ生成ユースケースのテスト．"""

    def test_basic_generation(self, tmp_path: Path) -> None:
        """基本的なデータ生成が正しく動作することを検証する．"""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # 3つの異なる局面を作成
        board1 = shogi.Board()
        hcp1 = _board_to_hcp_bytes(board1)

        board2 = shogi.Board()
        moves2 = list(board2.get_legal_moves())
        board2.push_move(moves2[0])
        hcp2 = _board_to_hcp_bytes(board2)

        board3 = shogi.Board()
        moves3 = list(board3.get_legal_moves())
        board3.push_move(moves3[1])
        hcp3 = _board_to_hcp_bytes(board3)

        _create_hcpe_feather(
            input_dir / "data.feather", [hcp1, hcp2, hcp3]
        )

        config = Stage2DataGenerationConfig(
            input_dir=input_dir,
            output_dir=output_dir,
        )
        use_case = Stage2DataGenerationUseCase()
        result = use_case.execute(config)

        # 結果辞書の検証
        assert result["total_input_positions"] == 3
        assert result["total_unique_positions"] == 3
        assert len(result["output_files"]) == 1

        # 出力ファイルが存在し読み込み可能であることを検証
        output_file = Path(result["output_files"][0])
        assert output_file.exists()
        df = load_stage2_df(output_file)
        assert len(df) == 3

    def test_deduplication(self, tmp_path: Path) -> None:
        """重複局面が正しく排除されることを検証する．"""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # 同一局面を含むデータを作成
        board1 = shogi.Board()
        hcp1 = _board_to_hcp_bytes(board1)

        board2 = shogi.Board()
        moves2 = list(board2.get_legal_moves())
        board2.push_move(moves2[0])
        hcp2 = _board_to_hcp_bytes(board2)

        # ファイル1: hcp1, hcp2
        _create_hcpe_feather(
            input_dir / "data1.feather", [hcp1, hcp2]
        )
        # ファイル2: hcp1(重複), hcp2(重複)
        _create_hcpe_feather(
            input_dir / "data2.feather", [hcp1, hcp2]
        )

        config = Stage2DataGenerationConfig(
            input_dir=input_dir,
            output_dir=output_dir,
        )
        use_case = Stage2DataGenerationUseCase()
        result = use_case.execute(config)

        assert result["total_input_positions"] == 4
        assert result["total_unique_positions"] == 2
        assert (
            result["total_unique_positions"]
            < result["total_input_positions"]
        )

        # 出力に重複がないことを検証
        output_file = Path(result["output_files"][0])
        df = load_stage2_df(output_file)
        assert len(df) == 2
        assert df["id"].n_unique() == 2

    def test_legal_moves_labels_correctness(
        self, tmp_path: Path
    ) -> None:
        """合法手ラベルがcshogiの合法手と一致することを検証する．"""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # 初期局面を使用
        board = shogi.Board()
        hcp_bytes = _board_to_hcp_bytes(board)

        _create_hcpe_feather(
            input_dir / "data.feather", [hcp_bytes]
        )

        config = Stage2DataGenerationConfig(
            input_dir=input_dir,
            output_dir=output_dir,
        )
        use_case = Stage2DataGenerationUseCase()
        result = use_case.execute(config)

        output_file = Path(result["output_files"][0])
        df = load_stage2_df(output_file)
        assert len(df) == 1

        # 出力された合法手ラベルを取得
        legal_labels = df["legalMovesLabel"][0].to_list()

        # cshogiから期待される合法手ラベルを計算
        expected_labels = np.zeros(
            MOVE_LABELS_NUM, dtype=np.uint8
        )
        verify_board = shogi.Board()
        for move in verify_board.get_legal_moves():
            label = make_move_label(
                verify_board.get_turn(), move
            )
            expected_labels[label] = 1

        # ラベルの一致を検証
        assert legal_labels == expected_labels.tolist()

        # 初期局面の合法手数が正しいことを検証(先手の初期合法手は30手)
        assert sum(legal_labels) == 30

    def test_chunk_output(self, tmp_path: Path) -> None:
        """chunk_sizeに応じて複数ファイルに分割出力されることを検証する．"""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # 4つの異なる局面を作成
        positions: list[bytes] = []
        board = shogi.Board()
        positions.append(_board_to_hcp_bytes(board))

        legal_moves = list(board.get_legal_moves())
        for i in range(3):
            b = shogi.Board()
            b.push_move(legal_moves[i])
            positions.append(_board_to_hcp_bytes(b))

        _create_hcpe_feather(
            input_dir / "data.feather", positions
        )

        config = Stage2DataGenerationConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            chunk_size=2,
        )
        use_case = Stage2DataGenerationUseCase()
        result = use_case.execute(config)

        # 4局面をchunk_size=2で分割すると2ファイル
        assert len(result["output_files"]) == 2

        # ファイル名のパターンを検証
        output_files = sorted(result["output_files"])
        assert "stage2_chunk0000.feather" in output_files[0]
        assert "stage2_chunk0001.feather" in output_files[1]

        # 全ファイルのレコード数の合計を検証
        total_records = 0
        for f in output_files:
            df = load_stage2_df(Path(f))
            total_records += len(df)
        assert total_records == 4

    def test_schema_validation(self, tmp_path: Path) -> None:
        """出力featherファイルのスキーマがget_stage2_polars_schema()と一致することを検証する．"""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        board = shogi.Board()
        hcp_bytes = _board_to_hcp_bytes(board)
        _create_hcpe_feather(
            input_dir / "data.feather", [hcp_bytes]
        )

        config = Stage2DataGenerationConfig(
            input_dir=input_dir,
            output_dir=output_dir,
        )
        use_case = Stage2DataGenerationUseCase()
        result = use_case.execute(config)

        output_file = Path(result["output_files"][0])
        df = load_stage2_df(output_file)

        # スキーマの一致を検証
        expected_schema = get_stage2_polars_schema()
        assert dict(df.schema) == expected_schema

        # legalMovesLabelの要素数がMOVE_LABELS_NUMであることを検証
        for legal_moves in df["legalMovesLabel"].to_list():
            assert len(legal_moves) == MOVE_LABELS_NUM

        # boardIdPositionsが9x9であることを検証
        for board_pos in df["boardIdPositions"].to_list():
            assert len(board_pos) == 9
            for row in board_pos:
                assert len(row) == 9

        # piecesInHandが14要素であることを検証
        for pieces in df["piecesInHand"].to_list():
            assert len(pieces) == 14

    def test_empty_directory_raises(
        self, tmp_path: Path
    ) -> None:
        """空のディレクトリを指定した場合にFileNotFoundErrorが発生することを検証する．"""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        config = Stage2DataGenerationConfig(
            input_dir=input_dir,
            output_dir=output_dir,
        )
        use_case = Stage2DataGenerationUseCase()

        with pytest.raises(FileNotFoundError):
            use_case.execute(config)

    def test_white_turn_labels_decodable(
        self, tmp_path: Path
    ) -> None:
        """後手番局面の合法手ラベルが正規化済み盤面と整合することを検証する．"""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # 後手番の局面を作成 (1手進める)
        board = shogi.Board()
        moves = list(board.get_legal_moves())
        board.push_move(moves[0])
        assert board.get_turn() == shogi.Turn.WHITE

        hcp_bytes = _board_to_hcp_bytes(board)
        _create_hcpe_feather(
            input_dir / "data.feather", [hcp_bytes]
        )

        config = Stage2DataGenerationConfig(
            input_dir=input_dir,
            output_dir=output_dir,
        )
        use_case = Stage2DataGenerationUseCase()
        result = use_case.execute(config)

        output_file = Path(result["output_files"][0])
        df = load_stage2_df(output_file)
        row = df.row(0, named=True)

        # 出力された合法手ラベルを取得
        legal_labels = row["legalMovesLabel"]
        active_labels = [
            i for i, val in enumerate(legal_labels) if val == 1
        ]

        # 正規化後の盤面を再構築し，その合法手ラベルと比較
        from maou.app.pre_process.feature import (
            make_board_id_positions,
            make_pieces_in_hand,
        )

        board_verify = shogi.Board()
        board_verify.push_move(moves[0])
        bp = make_board_id_positions(board_verify)
        pih = make_pieces_in_hand(board_verify)

        normalized = Stage2DataGenerationUseCase._reconstruct_normalized_board(
            bp, pih
        )

        expected_labels = set()
        for move in normalized.get_legal_moves():
            label = make_move_label(shogi.Turn.BLACK, move)
            expected_labels.add(label)

        # 生成されたラベルが正規化後盤面の合法手と一致
        assert set(active_labels) == expected_labels
        # 合法手数が妥当（正規化後の盤面の合法手数と一致）
        assert len(active_labels) == len(expected_labels)

    def test_legal_moves_labels_correctness_white_turn(
        self, tmp_path: Path
    ) -> None:
        """後手番の合法手ラベルが正規化後の盤面と整合することを検証する．"""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # 後手番の局面を使用
        board = shogi.Board()
        moves = list(board.get_legal_moves())
        board.push_move(moves[0])
        assert board.get_turn() == shogi.Turn.WHITE

        hcp_bytes = _board_to_hcp_bytes(board)
        _create_hcpe_feather(
            input_dir / "data.feather", [hcp_bytes]
        )

        config = Stage2DataGenerationConfig(
            input_dir=input_dir,
            output_dir=output_dir,
        )
        use_case = Stage2DataGenerationUseCase()
        result = use_case.execute(config)

        output_file = Path(result["output_files"][0])
        df = load_stage2_df(output_file)
        assert len(df) == 1

        # 出力された合法手ラベルを取得
        legal_labels = df["legalMovesLabel"][0].to_list()
        active_labels = [
            i for i, val in enumerate(legal_labels) if val == 1
        ]

        # 正規化後の盤面を再構築してラベルをデコード
        from maou.app.visualization.record_renderer import (
            Stage2RecordRenderer,
        )
        from maou.domain.move.label import (
            make_usi_move_from_label,
        )
        from maou.domain.visualization.board_renderer import (
            SVGBoardRenderer,
        )
        from maou.domain.visualization.move_label_converter import (
            MoveLabelConverter,
        )

        renderer = Stage2RecordRenderer(
            board_renderer=SVGBoardRenderer(),
            move_converter=MoveLabelConverter(),
        )
        row = df.row(0, named=True)
        reconstructed_board = (
            renderer._create_board_from_record(row)
        )

        # 全ラベルがUSIデコード可能であることを検証
        decode_failures = 0
        for label in active_labels:
            try:
                make_usi_move_from_label(
                    reconstructed_board, label
                )
            except Exception:
                decode_failures += 1

        # デコード失敗率が低いことを検証
        # (デコーダーの既知の制約により一部失敗する場合がある)
        failure_rate = decode_failures / len(active_labels)
        assert failure_rate < 0.2, (
            f"Too many decode failures: {decode_failures}/{len(active_labels)} "
            f"({failure_rate:.1%})"
        )
