"""VisualizationInterfaceのテスト．"""

from pathlib import Path

import polars as pl
import pytest

from maou.domain.board.shogi import Board
from maou.domain.data.rust_io import save_preprocessing_df
from maou.domain.data.schema import (
    get_preprocessing_polars_schema,
)
from maou.domain.move.label import MOVE_LABELS_NUM
from maou.infra.visualization.search_index import SearchIndex
from maou.interface.visualization import VisualizationInterface

INITIAL_SFEN = (
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/"
    "PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
)


class TestVisualizationInterface:
    """VisualizationInterfaceのテスト．"""

    @pytest.fixture
    def viz_interface(
        self, tmp_path: Path
    ) -> VisualizationInterface:
        """テスト用VisualizationInterfaceを作成．"""
        dummy_file = tmp_path / "test.feather"
        dummy_file.touch()

        search_index = SearchIndex.build(
            file_paths=[dummy_file],
            array_type="hcpe",
            num_mock_records=100,
            use_mock_data=True,
        )

        return VisualizationInterface(
            search_index=search_index,
            file_paths=[dummy_file],
            array_type="hcpe",
        )

    def test_initialization(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """VisualizationInterfaceが正常に初期化される．"""
        assert viz_interface.search_index is not None
        assert viz_interface.data_retriever is not None
        assert viz_interface.renderer is not None

    def test_search_by_id_valid(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """存在するIDで検索できる．"""
        board_svg, record_details = viz_interface.search_by_id(
            "mock_id_0"
        )

        assert "<svg" in board_svg
        assert "id" in record_details
        assert record_details["id"] == "mock_id_0"
        assert "eval" in record_details

    def test_search_by_id_empty_string(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """空文字列でエラーメッセージが返る．"""
        board_svg, record_details = viz_interface.search_by_id(
            ""
        )

        assert "IDを入力してください" in board_svg
        assert record_details == {}

    def test_search_by_id_nonexistent(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """存在しないIDでエラーメッセージが返る．"""
        board_svg, record_details = viz_interface.search_by_id(
            "nonexistent_id"
        )

        assert "見つかりません" in board_svg
        assert "error" in record_details

    def test_search_by_id_whitespace(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """空白のみのIDで検索エラーが返る．"""
        board_svg, record_details = viz_interface.search_by_id(
            "   "
        )

        assert "IDを入力してください" in board_svg
        assert record_details == {}

    def test_search_by_sfen_empty_string(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """空文字列でエラーメッセージが返る．"""
        board_svg, record_details = (
            viz_interface.search_by_sfen("")
        )

        assert "SFENを入力してください" in board_svg
        assert record_details == {}

    def test_search_by_sfen_whitespace(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """空白のみのSFENで検索エラーが返る．"""
        board_svg, record_details = (
            viz_interface.search_by_sfen("   ")
        )

        assert "SFENを入力してください" in board_svg
        assert record_details == {}

    def test_search_by_sfen_invalid(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """不正なSFEN文字列でエラーメッセージが返る．"""
        board_svg, record_details = (
            viz_interface.search_by_sfen("not a valid sfen")
        )

        assert "不正なSFEN文字列です" in board_svg
        assert record_details["error"] == "Invalid SFEN"

    def test_search_by_sfen_not_found_mock_mode(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """モックモードのhcpeでは一致局面が存在せず未検出になる．"""
        board_svg, record_details = (
            viz_interface.search_by_sfen(INITIAL_SFEN)
        )

        assert "見つかりません" in board_svg
        assert record_details["error"] == "Record not found"

    def test_search_by_sfen_hit_real_preprocessing_data(
        self, tmp_path: Path
    ) -> None:
        """Preprocessing実データでSFEN検索が一致レコードを描画する．"""
        board = Board()
        board.set_sfen(INITIAL_SFEN)
        position_hash = board.hash()

        schema = get_preprocessing_polars_schema()
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
        viz_interface = VisualizationInterface(
            search_index=search_index,
            file_paths=[file_path],
            array_type="preprocessing",
        )

        board_svg, record_details = (
            viz_interface.search_by_sfen(INITIAL_SFEN)
        )

        assert "<svg" in board_svg
        assert record_details["id"] == str(position_hash)

    def test_search_by_eval_range_valid(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """評価値範囲で検索できる．"""
        (
            table_data,
            page_info,
            board_svg,
            record_details,
            cached_records,
        ) = viz_interface.search_by_eval_range(
            min_eval=-1000,
            max_eval=1000,
            page=1,
            page_size=10,
        )

        assert isinstance(table_data, list)
        assert len(table_data) > 0  # 結果があることを確認
        assert "ページ 1" in page_info
        assert "<svg" in board_svg
        assert isinstance(record_details, dict)
        assert isinstance(cached_records, list)

    def test_search_by_eval_range_invalid_range(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """min > maxの場合にエラーが返る．"""
        (
            table_data,
            page_info,
            board_svg,
            record_details,
            cached_records,
        ) = viz_interface.search_by_eval_range(
            min_eval=100,
            max_eval=-100,
            page=1,
            page_size=10,
        )

        assert table_data == []
        assert "エラー" in page_info
        assert "無効な範囲" in board_svg
        assert "error" in record_details
        assert cached_records == []

    def test_search_by_eval_range_pagination(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """ページネーションが正しく動作する．"""
        # 1ページ目
        (
            table1,
            page_info1,
            _,
            _,
            _,
        ) = viz_interface.search_by_eval_range(
            min_eval=-1000,
            max_eval=1000,
            page=1,
            page_size=5,
        )

        # 2ページ目
        (
            table2,
            page_info2,
            _,
            _,
            _,
        ) = viz_interface.search_by_eval_range(
            min_eval=-1000,
            max_eval=1000,
            page=2,
            page_size=5,
        )

        assert len(table1) <= 5
        assert len(table2) <= 5
        assert "ページ 1" in page_info1
        assert "ページ 2" in page_info2

    def test_search_by_eval_range_no_results(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """結果がない場合に適切なメッセージが返る．"""
        (
            table_data,
            page_info,
            board_svg,
            record_details,
            cached_records,
        ) = viz_interface.search_by_eval_range(
            min_eval=9000,
            max_eval=10000,
            page=1,
            page_size=10,
        )

        assert table_data == []
        assert "結果なし" in page_info
        assert "検索結果がありません" in board_svg
        assert record_details == {}
        assert cached_records == []

    def test_get_dataset_stats(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """データセット統計情報を取得できる．"""
        stats = viz_interface.get_dataset_stats()

        assert "total_records" in stats
        assert stats["total_records"] == 100
        assert stats["array_type"] == "hcpe"
        assert stats["num_files"] == 1

    def test_search_by_eval_range_table_format(
        self, viz_interface: VisualizationInterface
    ) -> None:
        """テーブルデータが正しいフォーマットである．"""
        (
            table_data,
            _,
            _,
            _,
            _,
        ) = viz_interface.search_by_eval_range(
            min_eval=-100,
            max_eval=100,
            page=1,
            page_size=10,
        )

        if table_data:
            # 各行が4要素（インデックス，ID，評価値，手数）
            for row in table_data:
                assert len(row) == 4
                assert isinstance(row[0], int)  # インデックス
                assert isinstance(row[1], str)  # ID
                # row[2]: 評価値（int or None）
                # row[3]: 手数（int or None）
