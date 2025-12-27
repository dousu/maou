"""可視化ツールの統合テスト．"""

from pathlib import Path

import pytest

from maou.app.visualization.board_display import (
    BoardDisplayService,
)
from maou.app.visualization.data_retrieval import DataRetriever
from maou.domain.visualization.board_renderer import (
    SVGBoardRenderer,
)
from maou.infra.visualization.search_index import SearchIndex
from maou.interface.visualization import VisualizationInterface


class TestVisualizationIntegration:
    """可視化ツール全体の統合テスト．"""

    @pytest.fixture
    def test_data_dir(self, tmp_path: Path) -> Path:
        """テストデータディレクトリを作成．"""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()

        # ダミーファイルを複数作成
        for i in range(3):
            dummy_file = data_dir / f"test_{i}.feather"
            dummy_file.touch()

        return data_dir

    def test_end_to_end_id_search(
        self, test_data_dir: Path
    ) -> None:
        """ID検索のエンドツーエンドテスト．"""
        file_paths = list(test_data_dir.glob("*.feather"))

        # 1. SearchIndexを構築
        search_index = SearchIndex.build(
            file_paths=file_paths,
            array_type="hcpe",
            num_mock_records=50,
        )

        # 2. VisualizationInterfaceを作成
        viz_interface = VisualizationInterface(
            search_index=search_index,
            file_paths=file_paths,
            array_type="hcpe",
        )

        # 3. ID検索を実行
        board_svg, record_details = viz_interface.search_by_id(
            "mock_id_0"
        )

        # 4. 結果を検証
        assert "<svg" in board_svg
        assert record_details["id"] == "mock_id_0"
        assert (
            "boardIdPositions" not in record_details
        )  # 簡略化された詳細

    def test_end_to_end_eval_range_search(
        self, test_data_dir: Path
    ) -> None:
        """評価値範囲検索のエンドツーエンドテスト．"""
        file_paths = list(test_data_dir.glob("*.feather"))

        # 1. SearchIndexを構築
        search_index = SearchIndex.build(
            file_paths=file_paths,
            array_type="hcpe",
            num_mock_records=100,
        )

        # 2. VisualizationInterfaceを作成
        viz_interface = VisualizationInterface(
            search_index=search_index,
            file_paths=file_paths,
            array_type="hcpe",
        )

        # 3. 評価値範囲検索を実行
        (
            table_data,
            page_info,
            board_svg,
            record_details,
        ) = viz_interface.search_by_eval_range(
            min_eval=-1000,
            max_eval=1000,
            page=1,
            page_size=10,
        )

        # 4. 結果を検証
        assert isinstance(table_data, list)
        assert len(table_data) > 0  # 結果が存在する
        assert len(table_data) <= 10
        assert "ページ 1" in page_info
        assert "<svg" in board_svg

    def test_layer_communication(
        self, test_data_dir: Path
    ) -> None:
        """各レイヤー間の通信が正しく動作することを確認．"""
        file_paths = list(test_data_dir.glob("*.feather"))

        # Domain layer
        renderer = SVGBoardRenderer()

        # Infra layer
        search_index = SearchIndex.build(
            file_paths=file_paths,
            array_type="preprocessing",
            num_mock_records=30,
        )

        # App layer
        data_retriever = DataRetriever(
            search_index=search_index,
            file_paths=file_paths,
            array_type="preprocessing",
        )

        board_display = BoardDisplayService(renderer=renderer)

        # Interface layer
        viz_interface = VisualizationInterface(
            search_index=search_index,
            file_paths=file_paths,
            array_type="preprocessing",
        )

        # 各レイヤーが正しく初期化されている
        assert renderer is not None
        assert search_index is not None
        assert data_retriever is not None
        assert board_display is not None
        assert viz_interface is not None

        # 検索が動作する
        record = data_retriever.get_by_id("mock_id_0")
        assert record is not None

        # レンダリングが動作する
        svg = board_display.render_from_record(record)
        assert "<svg" in svg

    def test_multiple_file_sources(
        self, test_data_dir: Path
    ) -> None:
        """複数ファイルからのデータ読み込みが動作する．"""
        file_paths = list(test_data_dir.glob("*.feather"))
        assert len(file_paths) == 3

        # SearchIndexが全ファイルを認識
        search_index = SearchIndex.build(
            file_paths=file_paths,
            array_type="hcpe",
            num_mock_records=100,
        )

        assert len(search_index.file_paths) == 3
        assert search_index.total_records() == 100

    def test_different_array_types(
        self, tmp_path: Path
    ) -> None:
        """異なるarray_typeで動作することを確認．"""
        dummy_file = tmp_path / "test.feather"
        dummy_file.touch()

        array_types = [
            "hcpe",
            "preprocessing",
            "stage1",
            "stage2",
        ]

        for array_type in array_types:
            search_index = SearchIndex.build(
                file_paths=[dummy_file],
                array_type=array_type,
                num_mock_records=20,
            )

            viz_interface = VisualizationInterface(
                search_index=search_index,
                file_paths=[dummy_file],
                array_type=array_type,
            )

            # 統計情報が正しい
            stats = viz_interface.get_dataset_stats()
            assert stats["array_type"] == array_type
            assert stats["total_records"] == 20
