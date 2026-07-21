"""DataRetrieverのテスト．"""

from pathlib import Path

import polars as pl
import pytest

from maou.app.visualization.data_retrieval import DataRetriever
from maou.domain.board.shogi import (
    DOMAIN_WHITE_MIN,
    Board,
    PieceId,
)
from maou.domain.data.rust_io import (
    save_hcpe_df,
    save_preprocessing_df,
    save_stage1_df,
    save_stage2_df,
)
from maou.domain.data.schema import (
    get_hcpe_polars_schema,
    get_preprocessing_polars_schema,
    get_stage1_polars_schema,
    get_stage2_polars_schema,
)
from maou.infra.visualization.search_index import SearchIndex
from maou.interface.data_io import (
    load_hcpe_df,
    load_preprocessing_df,
    load_stage1_df,
    load_stage2_df,
)

INITIAL_SFEN = (
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/"
    "PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
)


class TestDataRetriever:
    """DataRetrieverのテスト．"""

    @pytest.fixture
    def search_index(self, tmp_path: Path) -> SearchIndex:
        """テスト用SearchIndexを作成．"""
        dummy_file = tmp_path / "test.feather"
        dummy_file.touch()

        index = SearchIndex.build(
            file_paths=[dummy_file],
            array_type="hcpe",
            num_mock_records=100,
            use_mock_data=True,
        )
        return index

    @pytest.fixture
    def data_retriever(
        self, search_index: SearchIndex, tmp_path: Path
    ) -> DataRetriever:
        """テスト用DataRetrieverを作成．"""
        dummy_file = tmp_path / "test.feather"
        return DataRetriever(
            search_index=search_index,
            file_paths=[dummy_file],
            array_type="hcpe",
            load_df=lambda path: pl.read_ipc(path),
        )

    def test_initialization(
        self, data_retriever: DataRetriever
    ) -> None:
        """DataRetrieverが正常に初期化される．"""
        assert data_retriever.search_index is not None
        assert data_retriever.file_paths is not None
        assert data_retriever.array_type == "hcpe"

    def test_get_by_id_existing(
        self, data_retriever: DataRetriever
    ) -> None:
        """存在するIDでレコードを取得できる．"""
        # モックデータではmock_id_0からmock_id_99まで存在
        record = data_retriever.get_by_id("mock_id_0")

        assert record is not None
        assert record["id"] == "mock_id_0"
        assert "boardIdPositions" in record
        assert "piecesInHand" in record
        assert "eval" in record

    def test_get_by_id_nonexistent(
        self, data_retriever: DataRetriever
    ) -> None:
        """存在しないIDでNoneが返る．"""
        record = data_retriever.get_by_id("nonexistent_id")
        assert record is None

    def test_get_by_eval_range(
        self, data_retriever: DataRetriever
    ) -> None:
        """評価値範囲でレコードを取得できる．"""
        records = data_retriever.get_by_eval_range(
            min_eval=-100,
            max_eval=100,
            offset=0,
            limit=10,
        )

        assert isinstance(records, list)
        assert len(records) <= 10

        # 各レコードが必要なフィールドを持つ
        for record in records:
            assert "id" in record
            assert "eval" in record
            assert "boardIdPositions" in record
            assert "piecesInHand" in record

    def test_get_by_eval_range_pagination(
        self, data_retriever: DataRetriever
    ) -> None:
        """ページネーションが正しく動作する．"""
        # 1ページ目
        page1 = data_retriever.get_by_eval_range(
            min_eval=-1000,
            max_eval=1000,
            offset=0,
            limit=5,
        )

        # 2ページ目
        page2 = data_retriever.get_by_eval_range(
            min_eval=-1000,
            max_eval=1000,
            offset=5,
            limit=5,
        )

        assert len(page1) == 5
        assert len(page2) == 5

        # 異なるレコードが取得される
        page1_ids = {r["id"] for r in page1}
        page2_ids = {r["id"] for r in page2}
        assert page1_ids != page2_ids

    def test_get_by_eval_range_empty(
        self, data_retriever: DataRetriever
    ) -> None:
        """範囲外の評価値で空リストが返る．"""
        records = data_retriever.get_by_eval_range(
            min_eval=9000,
            max_eval=10000,
            offset=0,
            limit=10,
        )

        assert records == []

    def test_get_by_eval_range_none_values(
        self, data_retriever: DataRetriever
    ) -> None:
        """min/maxがNoneの場合，全範囲を検索する．"""
        records = data_retriever.get_by_eval_range(
            min_eval=None,
            max_eval=None,
            offset=0,
            limit=20,
        )

        assert isinstance(records, list)
        assert len(records) <= 20

    def test_decode_hcp_to_board_info_success(
        self, data_retriever: DataRetriever
    ) -> None:
        """HCPデータから盤面情報を正しくデコードできる．"""
        # 初期局面のBoardを作成
        board = Board()
        board.set_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        )

        # HCPバイナリを取得
        hcp_bytes = board.to_hcp()

        # テスト用レコード
        record = {
            "id": "test_id",
            "eval": 0,
            "hcp": hcp_bytes,
        }

        # デコード実行
        decoded_record = (
            data_retriever._decode_hcp_to_board_info(record)
        )

        # 検証
        assert "boardIdPositions" in decoded_record
        assert "piecesInHand" in decoded_record

        # 盤面が9x9であることを確認
        board_positions = decoded_record["boardIdPositions"]
        assert len(board_positions) == 9
        for row in board_positions:
            assert len(row) == 9

        # 持ち駒が14要素であることを確認
        pieces_in_hand = decoded_record["piecesInHand"]
        assert len(pieces_in_hand) == 14

        # 初期局面では持ち駒はすべて0
        assert all(count == 0 for count in pieces_in_hand)

        # 盤面に駒が配置されていることを確認（空マスでないマスが存在）
        non_empty_squares = sum(
            1
            for row in board_positions
            for piece_id in row
            if piece_id != 0
        )
        assert non_empty_squares > 0

    def test_decode_hcp_to_board_info_with_pieces_in_hand(
        self, data_retriever: DataRetriever
    ) -> None:
        """持ち駒がある局面のHCPデコードが正しく動作する．"""
        # 持ち駒がある局面を作成
        board = Board()
        # 角落ちの初期局面（先手が角を1枚持っている状態をシミュレート）
        board.set_sfen(
            "lnsgkgsnl/1r7/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b B 1"
        )

        # HCPバイナリを取得
        hcp_bytes = board.to_hcp()

        # テスト用レコード
        record = {
            "id": "test_id_with_hand",
            "eval": 100,
            "hcp": hcp_bytes,
        }

        # デコード実行
        decoded_record = (
            data_retriever._decode_hcp_to_board_info(record)
        )

        # 持ち駒を確認
        pieces_in_hand = decoded_record["piecesInHand"]
        assert len(pieces_in_hand) == 14

        # 先手の角（インデックス5）が1枚あることを確認
        assert pieces_in_hand[5] == 1

    def test_decode_hcp_to_board_info_missing_hcp(
        self, data_retriever: DataRetriever
    ) -> None:
        """hcpフィールドが欠落している場合にエラーが発生する．"""
        # hcpフィールドがないレコード
        record = {
            "id": "test_id_no_hcp",
            "eval": 0,
        }

        # エラーが発生することを確認
        with pytest.raises(
            ValueError, match="hcp field is missing or empty"
        ):
            data_retriever._decode_hcp_to_board_info(record)

    def test_decode_hcp_to_board_info_empty_hcp(
        self, data_retriever: DataRetriever
    ) -> None:
        """hcpフィールドが空の場合にエラーが発生する．"""
        # hcpフィールドが空のレコード
        record = {
            "id": "test_id_empty_hcp",
            "eval": 0,
            "hcp": b"",
        }

        # エラーが発生することを確認
        with pytest.raises(
            ValueError, match="hcp field is missing or empty"
        ):
            data_retriever._decode_hcp_to_board_info(record)

    def test_decode_preserves_original_fields(
        self, data_retriever: DataRetriever
    ) -> None:
        """デコード時に元のフィールドが保持される．"""
        # 初期局面のBoardを作成
        board = Board()
        board.set_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        )

        # HCPバイナリを取得
        hcp_bytes = board.to_hcp()

        # 複数のフィールドを持つレコード
        record = {
            "id": "test_id_preserve",
            "eval": 150,
            "moves": 50,
            "hcp": hcp_bytes,
            "gameResult": 1,
        }

        # デコード実行
        decoded_record = (
            data_retriever._decode_hcp_to_board_info(record)
        )

        # 元のフィールドが保持されていることを確認
        assert decoded_record["id"] == "test_id_preserve"
        assert decoded_record["eval"] == 150
        assert decoded_record["moves"] == 50
        assert decoded_record["gameResult"] == 1
        assert decoded_record["hcp"] == hcp_bytes

        # 新しいフィールドが追加されていることを確認
        assert "boardIdPositions" in decoded_record
        assert "piecesInHand" in decoded_record

    def test_decode_hcp_board_positions_correct(
        self, data_retriever: DataRetriever
    ) -> None:
        """HCPデコード後の盤面配置が正しいことを検証．"""
        # 初期局面のBoardを作成
        board = Board()
        board.set_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        )

        # HCPバイナリを取得
        hcp_bytes = board.to_hcp()

        record = {"id": "test", "eval": 0, "hcp": hcp_bytes}
        decoded = data_retriever._decode_hcp_to_board_info(
            record
        )

        positions = decoded["boardIdPositions"]

        # 先手の玉の位置を確認（row=8, col=4 = 段i, 筋5）
        # 注: colは0=筋9なので、col=4は筋5（中央）
        assert positions[8][4] == PieceId.OU  # 先手王

        # 後手の玉の位置を確認（row=0, col=4 = 段a, 筋5）
        # 後手の駒は domain PieceId + 14 (22 = OU(8) + 14)．
        # raw 駒 ID の白玉(24)は RAW_PIECE_TO_PIECEID 変換で
        # domain 22 になる
        white_king_id = 22  # domain PieceId: OU + 14
        assert positions[0][4] == white_king_id  # 後手王

    def test_white_piece_detection_with_real_data(
        self, data_retriever: DataRetriever
    ) -> None:
        """実際のHCPデータで白駒が正しく判定されることを確認．"""
        # 持ち駒がある局面（先手が角を持っている）
        board = Board()
        board.set_sfen(
            "lnsgkgsnl/1r7/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b B 1"
        )

        hcp_bytes = board.to_hcp()

        record = {"id": "test", "eval": 0, "hcp": hcp_bytes}
        decoded = data_retriever._decode_hcp_to_board_info(
            record
        )

        positions = decoded["boardIdPositions"]

        # 後手の駒（row=0の駒）がすべてDOMAIN_WHITE_MIN以上であることを確認
        # boardIdPositionsはdomain形式（白駒=15-28）を返す
        for col in range(9):
            piece_id = positions[0][col]
            if piece_id != 0:  # 空マスでない場合
                assert piece_id >= DOMAIN_WHITE_MIN, (
                    f"White piece at (0, {col}) has ID {piece_id} "
                    f"< DOMAIN_WHITE_MIN ({DOMAIN_WHITE_MIN})"
                )


class TestGetBySfen:
    """DataRetriever.get_by_sfenのテスト．"""

    @pytest.fixture
    def data_retriever(self, tmp_path: Path) -> DataRetriever:
        """テスト用DataRetriever(hcpe/モックモード)を作成．"""
        dummy_file = tmp_path / "test.feather"
        dummy_file.touch()

        search_index = SearchIndex.build(
            file_paths=[dummy_file],
            array_type="hcpe",
            num_mock_records=100,
            use_mock_data=True,
        )
        return DataRetriever(
            search_index=search_index,
            file_paths=[dummy_file],
            array_type="hcpe",
            load_df=lambda path: pl.read_ipc(path),
        )

    def test_invalid_sfen_raises(
        self, data_retriever: DataRetriever
    ) -> None:
        """不正なSFEN文字列でValueErrorが発生する．"""
        with pytest.raises(ValueError):
            data_retriever.get_by_sfen("not a valid sfen")

    def test_mock_mode_returns_none(
        self, data_retriever: DataRetriever
    ) -> None:
        """モックモードのhcpeではSFEN検索は常にNoneを返す．"""
        record = data_retriever.get_by_sfen(INITIAL_SFEN)
        assert record is None

    def test_hcpe_real_data_hit(self, tmp_path: Path) -> None:
        """HCPE実データでSFENに一致するレコードを取得できる．"""
        target_board = Board()
        target_board.set_sfen(INITIAL_SFEN)
        target_hcp = target_board.to_hcp()

        # 別局面(初期局面から1手進めた局面)も混ぜて識別性を確認する
        other_board = Board()
        other_board.set_sfen(INITIAL_SFEN)
        other_board.push_move(
            next(iter(other_board.get_legal_moves()))
        )
        other_hcp = other_board.to_hcp()

        schema = get_hcpe_polars_schema()
        df = pl.DataFrame(
            {
                "hcp": pl.Series(
                    "hcp",
                    [other_hcp, target_hcp],
                    dtype=pl.Binary,
                ),
                "eval": pl.Series(
                    "eval", [0, 0], dtype=pl.Int16
                ),
                "bestMove16": pl.Series(
                    "bestMove16", [0, 0], dtype=pl.Int16
                ),
                "gameResult": pl.Series(
                    "gameResult", [1, 1], dtype=pl.Int8
                ),
                "id": pl.Series(
                    "id",
                    ["pos_other", "pos_target"],
                    dtype=pl.Utf8,
                ),
                "partitioningKey": pl.Series(
                    "partitioningKey",
                    [None, None],
                    dtype=pl.Date,
                ),
                "ratings": pl.Series(
                    "ratings",
                    [None, None],
                    dtype=pl.List(pl.UInt16),
                ),
                "endgameStatus": pl.Series(
                    "endgameStatus",
                    [None, None],
                    dtype=pl.Utf8,
                ),
                "moves": pl.Series(
                    "moves", [0, 1], dtype=pl.Int16
                ),
            },
            schema=schema,
        )

        file_path = tmp_path / "hcpe_real.feather"
        save_hcpe_df(df, file_path)

        search_index = SearchIndex.build(
            file_paths=[file_path],
            array_type="hcpe",
            use_mock_data=False,
        )
        retriever = DataRetriever(
            search_index=search_index,
            file_paths=[file_path],
            array_type="hcpe",
            load_df=load_hcpe_df,
        )

        record = retriever.get_by_sfen(INITIAL_SFEN)

        assert record is not None
        assert record["id"] == "pos_target"

    def test_hcpe_real_data_not_found(
        self, tmp_path: Path
    ) -> None:
        """HCPE実データに一致局面がない場合はNoneを返す．"""
        stored_board = Board()
        stored_board.set_sfen(INITIAL_SFEN)
        stored_board.push_move(
            next(iter(stored_board.get_legal_moves()))
        )
        stored_hcp = stored_board.to_hcp()

        schema = get_hcpe_polars_schema()
        df = pl.DataFrame(
            {
                "hcp": pl.Series(
                    "hcp", [stored_hcp], dtype=pl.Binary
                ),
                "eval": pl.Series("eval", [0], dtype=pl.Int16),
                "bestMove16": pl.Series(
                    "bestMove16", [0], dtype=pl.Int16
                ),
                "gameResult": pl.Series(
                    "gameResult", [1], dtype=pl.Int8
                ),
                "id": pl.Series("id", ["pos_0"], dtype=pl.Utf8),
                "partitioningKey": pl.Series(
                    "partitioningKey", [None], dtype=pl.Date
                ),
                "ratings": pl.Series(
                    "ratings",
                    [None],
                    dtype=pl.List(pl.UInt16),
                ),
                "endgameStatus": pl.Series(
                    "endgameStatus", [None], dtype=pl.Utf8
                ),
                "moves": pl.Series(
                    "moves", [0], dtype=pl.Int16
                ),
            },
            schema=schema,
        )

        file_path = tmp_path / "hcpe_real.feather"
        save_hcpe_df(df, file_path)

        search_index = SearchIndex.build(
            file_paths=[file_path],
            array_type="hcpe",
            use_mock_data=False,
        )
        retriever = DataRetriever(
            search_index=search_index,
            file_paths=[file_path],
            array_type="hcpe",
            load_df=load_hcpe_df,
        )

        # 初期局面は保存していないので見つからないはず
        record = retriever.get_by_sfen(INITIAL_SFEN)
        assert record is None

    def test_preprocessing_real_data_hit(
        self, tmp_path: Path
    ) -> None:
        """PreprocessingデータでSFENに一致するレコードをid経由で取得できる．"""
        board = Board()
        board.set_sfen(INITIAL_SFEN)
        position_hash = board.hash()

        board_positions = (
            board.get_normalized_board_id_positions().tolist()
        )
        pieces_in_hand = (
            board.get_normalized_pieces_in_hand().tolist()
        )

        from maou.domain.move.label import MOVE_LABELS_NUM

        schema = get_preprocessing_polars_schema()
        df = pl.DataFrame(
            {
                "id": pl.Series(
                    "id", [position_hash], dtype=pl.UInt64
                ),
                "boardIdPositions": pl.Series(
                    "boardIdPositions",
                    [board_positions],
                    dtype=pl.List(pl.List(pl.UInt8)),
                ),
                "piecesInHand": pl.Series(
                    "piecesInHand",
                    [pieces_in_hand],
                    dtype=pl.List(pl.UInt8),
                ),
                "moveLabel": pl.Series(
                    "moveLabel",
                    [[0.0] * MOVE_LABELS_NUM],
                    dtype=pl.List(pl.Float32),
                ),
                "moveWinRate": pl.Series(
                    "moveWinRate",
                    [[0.0] * MOVE_LABELS_NUM],
                    dtype=pl.List(pl.Float32),
                ),
                "bestMoveWinRate": pl.Series(
                    "bestMoveWinRate",
                    [0.5],
                    dtype=pl.Float32,
                ),
                "resultValue": pl.Series(
                    "resultValue", [0.0], dtype=pl.Float32
                ),
            },
            schema=schema,
        )

        file_path = tmp_path / "preprocessing_real.feather"
        save_preprocessing_df(df, file_path)

        search_index = SearchIndex.build(
            file_paths=[file_path],
            array_type="preprocessing",
            use_mock_data=False,
        )
        retriever = DataRetriever(
            search_index=search_index,
            file_paths=[file_path],
            array_type="preprocessing",
            load_df=load_preprocessing_df,
        )

        record = retriever.get_by_sfen(INITIAL_SFEN)

        assert record is not None
        assert record["id"] == str(position_hash)

    def test_preprocessing_real_data_not_found(
        self, tmp_path: Path
    ) -> None:
        """Preprocessingデータに一致局面がない場合はNoneを返す．"""
        stored_board = Board()
        stored_board.set_sfen(INITIAL_SFEN)
        stored_board.push_move(
            next(iter(stored_board.get_legal_moves()))
        )
        stored_hash = stored_board.hash()

        from maou.domain.move.label import MOVE_LABELS_NUM

        schema = get_preprocessing_polars_schema()
        df = pl.DataFrame(
            {
                "id": pl.Series(
                    "id", [stored_hash], dtype=pl.UInt64
                ),
                "boardIdPositions": pl.Series(
                    "boardIdPositions",
                    [
                        stored_board.get_normalized_board_id_positions().tolist()
                    ],
                    dtype=pl.List(pl.List(pl.UInt8)),
                ),
                "piecesInHand": pl.Series(
                    "piecesInHand",
                    [
                        stored_board.get_normalized_pieces_in_hand().tolist()
                    ],
                    dtype=pl.List(pl.UInt8),
                ),
                "moveLabel": pl.Series(
                    "moveLabel",
                    [[0.0] * MOVE_LABELS_NUM],
                    dtype=pl.List(pl.Float32),
                ),
                "moveWinRate": pl.Series(
                    "moveWinRate",
                    [[0.0] * MOVE_LABELS_NUM],
                    dtype=pl.List(pl.Float32),
                ),
                "bestMoveWinRate": pl.Series(
                    "bestMoveWinRate",
                    [0.5],
                    dtype=pl.Float32,
                ),
                "resultValue": pl.Series(
                    "resultValue", [0.0], dtype=pl.Float32
                ),
            },
            schema=schema,
        )

        file_path = tmp_path / "preprocessing_real.feather"
        save_preprocessing_df(df, file_path)

        search_index = SearchIndex.build(
            file_paths=[file_path],
            array_type="preprocessing",
            use_mock_data=False,
        )
        retriever = DataRetriever(
            search_index=search_index,
            file_paths=[file_path],
            array_type="preprocessing",
            load_df=load_preprocessing_df,
        )

        record = retriever.get_by_sfen(INITIAL_SFEN)
        assert record is None

    def test_stage2_real_data_hit(self, tmp_path: Path) -> None:
        """Stage2データでSFENに一致するレコードをid経由で取得できる．"""
        board = Board()
        board.set_sfen(INITIAL_SFEN)
        position_hash = board.hash()

        from maou.domain.move.label import MOVE_LABELS_NUM

        schema = get_stage2_polars_schema()
        df = pl.DataFrame(
            {
                "id": pl.Series(
                    "id", [position_hash], dtype=pl.UInt64
                ),
                "boardIdPositions": pl.Series(
                    "boardIdPositions",
                    [
                        board.get_normalized_board_id_positions().tolist()
                    ],
                    dtype=pl.List(pl.List(pl.UInt8)),
                ),
                "piecesInHand": pl.Series(
                    "piecesInHand",
                    [
                        board.get_normalized_pieces_in_hand().tolist()
                    ],
                    dtype=pl.List(pl.UInt8),
                ),
                "legalMovesLabel": pl.Series(
                    "legalMovesLabel",
                    [[0] * MOVE_LABELS_NUM],
                    dtype=pl.List(pl.UInt8),
                ),
            },
            schema=schema,
        )

        file_path = tmp_path / "stage2_real.feather"
        save_stage2_df(df, file_path)

        search_index = SearchIndex.build(
            file_paths=[file_path],
            array_type="stage2",
            use_mock_data=False,
        )
        retriever = DataRetriever(
            search_index=search_index,
            file_paths=[file_path],
            array_type="stage2",
            load_df=load_stage2_df,
        )

        record = retriever.get_by_sfen(INITIAL_SFEN)

        assert record is not None
        assert record["id"] == str(position_hash)

    def test_stage1_real_data_hit(self, tmp_path: Path) -> None:
        """Stage1データは合成idのため盤面配列の一致で検索する．"""
        board = Board()
        board.set_sfen(INITIAL_SFEN)
        board_positions = (
            board.get_normalized_board_id_positions().tolist()
        )
        pieces_in_hand = (
            board.get_normalized_pieces_in_hand().tolist()
        )
        empty_reachable = [[0] * 9 for _ in range(9)]

        schema = get_stage1_polars_schema()
        df = pl.DataFrame(
            {
                "id": pl.Series("id", [12345], dtype=pl.UInt64),
                "boardIdPositions": pl.Series(
                    "boardIdPositions",
                    [board_positions],
                    dtype=pl.List(pl.List(pl.UInt8)),
                ),
                "piecesInHand": pl.Series(
                    "piecesInHand",
                    [pieces_in_hand],
                    dtype=pl.List(pl.UInt8),
                ),
                "reachableSquares": pl.Series(
                    "reachableSquares",
                    [empty_reachable],
                    dtype=pl.List(pl.List(pl.UInt8)),
                ),
            },
            schema=schema,
        )

        file_path = tmp_path / "stage1_real.feather"
        save_stage1_df(df, file_path)

        search_index = SearchIndex.build(
            file_paths=[file_path],
            array_type="stage1",
            use_mock_data=False,
        )
        retriever = DataRetriever(
            search_index=search_index,
            file_paths=[file_path],
            array_type="stage1",
            load_df=load_stage1_df,
        )

        record = retriever.get_by_sfen(INITIAL_SFEN)

        assert record is not None
        assert record["id"] == "12345"

    def test_stage1_real_data_not_found(
        self, tmp_path: Path
    ) -> None:
        """Stage1データに一致局面がない場合はNoneを返す．"""
        stored_board = Board()
        stored_board.set_sfen(INITIAL_SFEN)
        stored_board.push_move(
            next(iter(stored_board.get_legal_moves()))
        )
        empty_reachable = [[0] * 9 for _ in range(9)]

        schema = get_stage1_polars_schema()
        df = pl.DataFrame(
            {
                "id": pl.Series("id", [99999], dtype=pl.UInt64),
                "boardIdPositions": pl.Series(
                    "boardIdPositions",
                    [
                        stored_board.get_normalized_board_id_positions().tolist()
                    ],
                    dtype=pl.List(pl.List(pl.UInt8)),
                ),
                "piecesInHand": pl.Series(
                    "piecesInHand",
                    [
                        stored_board.get_normalized_pieces_in_hand().tolist()
                    ],
                    dtype=pl.List(pl.UInt8),
                ),
                "reachableSquares": pl.Series(
                    "reachableSquares",
                    [empty_reachable],
                    dtype=pl.List(pl.List(pl.UInt8)),
                ),
            },
            schema=schema,
        )

        file_path = tmp_path / "stage1_real.feather"
        save_stage1_df(df, file_path)

        search_index = SearchIndex.build(
            file_paths=[file_path],
            array_type="stage1",
            use_mock_data=False,
        )
        retriever = DataRetriever(
            search_index=search_index,
            file_paths=[file_path],
            array_type="stage1",
            load_df=load_stage1_df,
        )

        record = retriever.get_by_sfen(INITIAL_SFEN)
        assert record is None


class TestPreprocessingMockData:
    """_create_preprocessing_mockのテスト．"""

    @pytest.fixture
    def data_retriever(self, tmp_path: Path) -> DataRetriever:
        """Preprocessing用DataRetrieverを作成．"""
        from maou.infra.visualization.search_index import (
            SearchIndex,
        )

        dummy_file = tmp_path / "test.feather"
        search_index = SearchIndex(
            file_paths=[dummy_file],
            array_type="preprocessing",
            use_mock_data=True,
        )
        dummy_file = tmp_path / "test.feather"
        return DataRetriever(
            search_index=search_index,
            file_paths=[dummy_file],
            array_type="preprocessing",
            load_df=lambda path: pl.read_ipc(path),
        )

    def test_mock_includes_move_win_rate(
        self, data_retriever: DataRetriever
    ) -> None:
        """モックデータにmoveWinRateが含まれる．"""
        record = data_retriever._create_preprocessing_mock(
            record_id="test", row_number=0
        )

        assert "moveWinRate" in record
        assert "bestMoveWinRate" in record
        assert isinstance(record["moveWinRate"], list)
        assert isinstance(record["bestMoveWinRate"], float)

    def test_mock_move_win_rate_consistency(
        self, data_retriever: DataRetriever
    ) -> None:
        """moveWinRateはmoveLabelが非ゼロの位置のみ非ゼロ．"""
        record = data_retriever._create_preprocessing_mock(
            record_id="test", row_number=1
        )

        move_label = record["moveLabel"]
        move_win_rate = record["moveWinRate"]

        for i in range(len(move_label)):
            if move_label[i] == 0.0:
                assert move_win_rate[i] == 0.0
            else:
                assert move_win_rate[i] > 0.0

    def test_mock_best_move_win_rate_is_max(
        self, data_retriever: DataRetriever
    ) -> None:
        """bestMoveWinRateはmoveWinRateの最大値．"""
        record = data_retriever._create_preprocessing_mock(
            record_id="test", row_number=2
        )

        non_zero = [
            wr for wr in record["moveWinRate"] if wr > 0.0
        ]
        assert record["bestMoveWinRate"] == max(non_zero)
