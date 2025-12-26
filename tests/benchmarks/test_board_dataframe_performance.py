"""Performance benchmarks for Board DataFrame methods．

This module benchmarks the conversion overhead of DataFrame-returning
methods compared to direct numpy operations．

To run these benchmarks:
    poetry run pytest tests/benchmarks/test_board_dataframe_performance.py -v -s
"""

import timeit

import numpy as np

from maou.domain.board.shogi import Board


class TestBoardDataFramePerformance:
    """Benchmark DataFrame method conversion overhead．"""

    def test_get_board_id_positions_df_performance(
        self,
    ) -> None:
        """Benchmark get_board_id_positions_df() conversion overhead．

        Expected: < 1ms per call for single board position．
        """
        board = Board()
        board.set_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        )

        # Benchmark with timeit (1000 iterations)
        iterations = 1000
        elapsed = timeit.timeit(
            lambda: board.get_board_id_positions_df(),
            number=iterations,
        )
        avg_time_ms = (elapsed / iterations) * 1000

        print(
            f"\nget_board_id_positions_df(): {avg_time_ms:.3f}ms per call"
        )

        # Verify result is valid
        result = board.get_board_id_positions_df()
        assert len(result) == 1
        assert "boardIdPositions" in result.columns

        # Performance check
        assert avg_time_ms < 1.0, (
            f"Too slow: {avg_time_ms:.3f}ms > 1ms"
        )

    def test_get_hcp_df_performance(self) -> None:
        """Benchmark get_hcp_df() conversion overhead．

        Expected: < 0.5ms per call for HCP conversion．
        """
        board = Board()
        board.set_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        )

        # Benchmark with timeit (1000 iterations)
        iterations = 1000
        elapsed = timeit.timeit(
            lambda: board.get_hcp_df(), number=iterations
        )
        avg_time_ms = (elapsed / iterations) * 1000

        print(f"\nget_hcp_df(): {avg_time_ms:.3f}ms per call")

        # Verify result is valid
        result = board.get_hcp_df()
        assert len(result) == 1
        assert "hcp" in result.columns

        # Performance check
        assert avg_time_ms < 0.5, (
            f"Too slow: {avg_time_ms:.3f}ms > 0.5ms"
        )

    def test_get_piece_planes_df_performance(self) -> None:
        """Benchmark get_piece_planes_df() conversion overhead．

        Expected: < 10ms per call for 104x9x9 array conversion．
        """
        board = Board()
        board.set_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        )

        # Benchmark with timeit (100 iterations for larger data)
        iterations = 100
        elapsed = timeit.timeit(
            lambda: board.get_piece_planes_df(),
            number=iterations,
        )
        avg_time_ms = (elapsed / iterations) * 1000

        print(
            f"\nget_piece_planes_df(): {avg_time_ms:.3f}ms per call"
        )

        # Verify result is valid
        result = board.get_piece_planes_df()
        assert len(result) == 1
        assert "piecePlanes" in result.columns

        # Performance check (relaxed threshold for variability)
        assert avg_time_ms < 11.0, (
            f"Too slow: {avg_time_ms:.3f}ms > 11ms"
        )

    def test_get_piece_planes_rotate_df_performance(
        self,
    ) -> None:
        """Benchmark get_piece_planes_rotate_df() conversion overhead．

        Expected: < 11ms per call for 104x9x9 array conversion．
        """
        board = Board()
        board.set_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1"
        )

        # Benchmark with timeit (100 iterations for larger data)
        iterations = 100
        elapsed = timeit.timeit(
            lambda: board.get_piece_planes_rotate_df(),
            number=iterations,
        )
        avg_time_ms = (elapsed / iterations) * 1000

        print(
            f"\nget_piece_planes_rotate_df(): {avg_time_ms:.3f}ms per call"
        )

        # Verify result is valid
        result = board.get_piece_planes_rotate_df()
        assert len(result) == 1
        assert "piecePlanes" in result.columns

        # Performance check (relaxed threshold for variability)
        assert avg_time_ms < 11.0, (
            f"Too slow: {avg_time_ms:.3f}ms > 11ms"
        )


class TestDataFrameToNumpyConversion:
    """Benchmark DataFrame to numpy conversion overhead．"""

    def test_board_positions_df_to_numpy_conversion(
        self,
    ) -> None:
        """Benchmark conversion from DataFrame back to numpy array．"""
        board = Board()
        df = board.get_board_id_positions_df()

        def convert_to_numpy() -> np.ndarray:
            positions_list = df["boardIdPositions"].to_list()[0]
            return np.array(positions_list, dtype=np.uint8)

        # Benchmark with timeit (1000 iterations)
        iterations = 1000
        elapsed = timeit.timeit(
            convert_to_numpy, number=iterations
        )
        avg_time_ms = (elapsed / iterations) * 1000

        print(
            f"\nDataFrame→numpy (board positions): {avg_time_ms:.3f}ms per call"
        )

        # Verify result
        result = convert_to_numpy()
        assert result.shape == (9, 9)
        assert result.dtype == np.uint8

    def test_hcp_df_to_numpy_conversion(self) -> None:
        """Benchmark HCP DataFrame to numpy conversion．"""
        board = Board()
        df = board.get_hcp_df()

        def convert_to_numpy() -> np.ndarray:
            import cshogi

            hcp_bytes = df["hcp"][0]
            return np.frombuffer(
                hcp_bytes,
                dtype=cshogi.HuffmanCodedPos,  # type: ignore
            )

        # Benchmark with timeit (1000 iterations)
        iterations = 1000
        elapsed = timeit.timeit(
            convert_to_numpy, number=iterations
        )
        avg_time_ms = (elapsed / iterations) * 1000

        print(
            f"\nDataFrame→numpy (HCP): {avg_time_ms:.3f}ms per call"
        )

        # Verify result
        result = convert_to_numpy()
        assert len(result) == 1

    def test_piece_planes_df_to_numpy_conversion(self) -> None:
        """Benchmark piece planes DataFrame to numpy conversion．"""
        board = Board()
        df = board.get_piece_planes_df()

        def convert_to_numpy() -> np.ndarray:
            planes_list = df["piecePlanes"].to_list()[0]
            return np.array(planes_list, dtype=np.float32)

        # Benchmark with timeit (100 iterations for larger data)
        iterations = 100
        elapsed = timeit.timeit(
            convert_to_numpy, number=iterations
        )
        avg_time_ms = (elapsed / iterations) * 1000

        print(
            f"\nDataFrame→numpy (piece planes): {avg_time_ms:.3f}ms per call"
        )

        # Verify result
        result = convert_to_numpy()
        assert result.shape == (104, 9, 9)
        assert result.dtype == np.float32


class TestEndToEndPerformance:
    """Benchmark end-to-end workflows．"""

    def test_preprocessing_workflow_performance(self) -> None:
        """Benchmark typical preprocessing workflow．

        Workflow: Board → DataFrame → numpy for training．
        """
        board = Board()
        board.set_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        )

        def preprocessing_workflow() -> np.ndarray:
            # Get board positions as DataFrame
            df = board.get_board_id_positions_df()

            # Convert to numpy for training
            positions_list = df["boardIdPositions"].to_list()[0]
            positions = np.array(positions_list, dtype=np.uint8)

            return positions

        # Benchmark with timeit (1000 iterations)
        iterations = 1000
        elapsed = timeit.timeit(
            preprocessing_workflow, number=iterations
        )
        avg_time_ms = (elapsed / iterations) * 1000

        print(
            f"\nEnd-to-end workflow (Board→DF→numpy): {avg_time_ms:.3f}ms per call"
        )

        # Verify result
        result = preprocessing_workflow()
        assert result.shape == (9, 9)
        assert result.dtype == np.uint8

    def test_batch_processing_performance(self) -> None:
        """Benchmark processing multiple board positions．

        Simulates processing 100 board positions．
        """
        boards = [Board() for _ in range(100)]
        for i, board in enumerate(boards):
            if i % 2 == 0:
                board.set_sfen(
                    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
                )
            else:
                board.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1")

        def batch_processing() -> list[np.ndarray]:
            results = []
            for board in boards:
                df = board.get_board_id_positions_df()
                positions_list = df[
                    "boardIdPositions"
                ].to_list()[0]
                positions = np.array(
                    positions_list, dtype=np.uint8
                )
                results.append(positions)
            return results

        # Benchmark with timeit (10 iterations for batch)
        iterations = 10
        elapsed = timeit.timeit(
            batch_processing, number=iterations
        )
        avg_time_ms = (elapsed / iterations) * 1000

        print(
            f"\nBatch processing (100 boards): {avg_time_ms:.1f}ms total, {avg_time_ms / 100:.3f}ms per board"
        )

        # Verify results
        results = batch_processing()
        assert len(results) == 100
        assert all(r.shape == (9, 9) for r in results)
