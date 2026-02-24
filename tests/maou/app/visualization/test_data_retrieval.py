"""DataRetrieverのテスト．"""

from pathlib import Path

import pytest

from maou.app.visualization.data_retrieval import DataRetriever
from maou.domain.board.shogi import (
    DOMAIN_WHITE_MIN,
    Board,
    PieceId,
)
from maou.infra.visualization.search_index import SearchIndex


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
        hcp_df = board.get_hcp_df()
        hcp_bytes = hcp_df["hcp"][0]

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
        hcp_df = board.get_hcp_df()
        hcp_bytes = hcp_df["hcp"][0]

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
        hcp_df = board.get_hcp_df()
        hcp_bytes = hcp_df["hcp"][0]

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
        hcp_df = board.get_hcp_df()
        hcp_bytes = hcp_df["hcp"][0]

        record = {"id": "test", "eval": 0, "hcp": hcp_bytes}
        decoded = data_retriever._decode_hcp_to_board_info(
            record
        )

        positions = decoded["boardIdPositions"]

        # 先手の玉の位置を確認（row=8, col=4 = 段i, 筋5）
        # 注: colは0=筋9なので、col=4は筋5（中央）
        assert positions[8][4] == PieceId.OU  # 先手王

        # 後手の玉の位置を確認（row=0, col=4 = 段a, 筋5）
        # 後手の駒はPieceId + 15なので、22 = 8 + 14 = OU + 14
        # ※ map_cshogi_to_piece_idで変換後の値
        # cshogiのWKING(24) → project ID (24-2=22)
        white_king_id = 22  # cshogiのWKING(24)がmap後に22になる
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

        hcp_df = board.get_hcp_df()
        hcp_bytes = hcp_df["hcp"][0]

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
