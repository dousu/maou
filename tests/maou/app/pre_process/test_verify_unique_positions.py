"""ユニーク局面計算の正確性を検証するテスト．

preprocessパイプラインがcshogiのzobrist_hashに基づいて
正しくユニーク局面を判定しているかを検証する．

- 異なる局面は異なるユニーク局面として保持されること
- 同一局面は正しくマージされること
- 並列処理でもユニーク局面数が変わらないこと
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from maou.app.pre_process.hcpe_transform import PreProcess
from maou.domain.board.shogi import Board, move16
from maou.domain.data.rust_io import save_hcpe_df
from maou.domain.data.schema import get_hcpe_polars_schema
from maou.infra.file_system.streaming_hcpe_source import (
    StreamingHcpeDataSource,
)


def _generate_distinct_positions(
    n: int,
) -> list[tuple[bytes, int, int]]:
    """初期局面から合法手を順に指して異なる局面を生成する．

    Args:
        n: 生成する局面数

    Returns:
        list of (hcp_bytes, move16_value, zobrist_hash)
    """
    board = Board()
    positions: list[tuple[bytes, int, int]] = []

    for _ in range(n):
        hcp = np.zeros(32, dtype=np.uint8)
        board.to_hcp(hcp)
        z_hash = board.hash()

        legal_moves = list(board.get_legal_moves())
        m = legal_moves[0]
        m16 = move16(m)

        positions.append((hcp.tobytes(), int(m16), z_hash))
        board.push_move(m)

    return positions


def _create_hcpe_feather(
    file_path: Path,
    positions: list[tuple[bytes, int, int]],
) -> Path:
    """指定した局面からHCPE featherファイルを作成する．

    Args:
        file_path: 出力先のパス
        positions: (hcp_bytes, move16_value, zobrist_hash) のリスト

    Returns:
        作成したファイルのパス
    """
    n = len(positions)
    schema = get_hcpe_polars_schema()

    hcp_list = [p[0] for p in positions]
    move16_list = [p[1] for p in positions]

    df = pl.DataFrame(
        {
            "hcp": pl.Series(
                "hcp", hcp_list, dtype=pl.Binary
            ),
            "eval": pl.Series(
                "eval", [0] * n, dtype=pl.Int16
            ),
            "bestMove16": pl.Series(
                "bestMove16", move16_list, dtype=pl.Int16
            ),
            "gameResult": pl.Series(
                "gameResult", [1] * n, dtype=pl.Int8
            ),
            "id": pl.Series(
                "id",
                [f"pos_{i}" for i in range(n)],
                dtype=pl.Utf8,
            ),
            "partitioningKey": pl.Series(
                "partitioningKey",
                [None] * n,
                dtype=pl.Date,
            ),
            "ratings": pl.Series(
                "ratings",
                [None] * n,
                dtype=pl.List(pl.UInt16),
            ),
            "endgameStatus": pl.Series(
                "endgameStatus",
                [None] * n,
                dtype=pl.Utf8,
            ),
            "moves": pl.Series(
                "moves", [0] * n, dtype=pl.Int16
            ),
        },
        schema=schema,
    )

    file_path.parent.mkdir(parents=True, exist_ok=True)
    save_hcpe_df(df, file_path)
    return file_path


def _run_preprocess(
    file_paths: list[Path],
    output_dir: Path,
    max_workers: int = 1,
) -> int:
    """前処理を実行してユニーク局面数を返す．

    Args:
        file_paths: 入力ファイルのリスト
        output_dir: 出力ディレクトリ
        max_workers: ワーカー数

    Returns:
        ユニーク局面数
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    datasource = StreamingHcpeDataSource(
        file_paths=file_paths,
    )

    option = PreProcess.PreProcessOption(
        output_dir=output_dir,
        max_workers=max_workers,
    )

    preprocessor = PreProcess(
        datasource=datasource,
        feature_store=None,
    )

    preprocessor.transform(option)

    # 出力ファイルの行数を合計
    total_rows = 0
    for f in output_dir.glob("*.feather"):
        from maou.domain.data.rust_io import (
            load_preprocessing_df,
        )

        df = load_preprocessing_df(f)
        total_rows += len(df)

    return total_rows


class TestUniquePositionVerification:
    """ユニーク局面計算の正確性を検証するテストクラス．"""

    def test_all_distinct_positions_preserved(
        self, tmp_path: Path
    ) -> None:
        """全て異なる局面が正しくユニーク局面として保持されることを検証する．"""
        n_positions = 10
        positions = _generate_distinct_positions(n_positions)

        # zobrist_hashが全て異なることを事前確認
        hashes = [p[2] for p in positions]
        assert len(set(hashes)) == n_positions, (
            f"テストデータの前提条件: {n_positions}局面全て異なるhashであるべき"
        )

        file_path = _create_hcpe_feather(
            tmp_path / "input" / "distinct.feather",
            positions,
        )

        unique_count = _run_preprocess(
            file_paths=[file_path],
            output_dir=tmp_path / "output",
        )

        assert unique_count == n_positions, (
            f"全て異なる{n_positions}局面の入力に対して"
            f"ユニーク局面数は{n_positions}であるべきだが，{unique_count}だった"
        )

    def test_duplicate_positions_merged(
        self, tmp_path: Path
    ) -> None:
        """同一局面が正しくマージされることを検証する．"""
        n_distinct = 5
        positions = _generate_distinct_positions(n_distinct)

        # 各局面を3回ずつ重複させる
        duplicated = positions * 3

        file_path = _create_hcpe_feather(
            tmp_path / "input" / "duplicated.feather",
            duplicated,
        )

        unique_count = _run_preprocess(
            file_paths=[file_path],
            output_dir=tmp_path / "output",
        )

        assert unique_count == n_distinct, (
            f"{n_distinct}種類の局面を各3回重複させた入力({len(duplicated)}局面)に対して"
            f"ユニーク局面数は{n_distinct}であるべきだが，{unique_count}だった"
        )

    def test_cross_file_deduplication(
        self, tmp_path: Path
    ) -> None:
        """ファイルをまたいだ重複局面が正しくマージされることを検証する．"""
        n_distinct = 5
        positions = _generate_distinct_positions(n_distinct)

        # 同じ局面を複数ファイルに分散
        file_paths = []
        for i in range(3):
            fp = _create_hcpe_feather(
                tmp_path / "input" / f"file_{i}.feather",
                positions,  # 全ファイルに同じ局面
            )
            file_paths.append(fp)

        unique_count = _run_preprocess(
            file_paths=file_paths,
            output_dir=tmp_path / "output",
        )

        assert unique_count == n_distinct, (
            f"3ファイルに同じ{n_distinct}局面を含む入力に対して"
            f"ユニーク局面数は{n_distinct}であるべきだが，{unique_count}だった"
        )

    def test_mixed_unique_and_duplicate(
        self, tmp_path: Path
    ) -> None:
        """ユニーク局面と重複局面が混在する場合の正確性を検証する．"""
        positions = _generate_distinct_positions(10)

        # 前半5局面は1回のみ，後半5局面は各5回重複
        mixed = list(positions[:5]) + list(positions[5:]) * 5

        file_path = _create_hcpe_feather(
            tmp_path / "input" / "mixed.feather",
            mixed,
        )

        unique_count = _run_preprocess(
            file_paths=[file_path],
            output_dir=tmp_path / "output",
        )

        assert unique_count == 10, (
            f"10種類の局面(うち5種類は5回重複)の入力に対して"
            f"ユニーク局面数は10であるべきだが，{unique_count}だった"
        )


class TestParallelConsistency:
    """並列処理によるユニーク局面計算の一貫性を検証するテストクラス．"""

    def test_parallel_same_unique_count(
        self, tmp_path: Path
    ) -> None:
        """シングルスレッドと並列処理で同じユニーク局面数になることを検証する．"""
        n_distinct = 8
        positions = _generate_distinct_positions(n_distinct)

        # 重複を含むデータを複数ファイルに分割
        file_paths = []
        for i in range(4):
            # 各ファイルに一部重複・一部ユニークな局面を含める
            start = i * 2
            end = start + 4
            file_positions = positions[
                start : min(end, n_distinct)
            ]
            if not file_positions:
                continue

            fp = _create_hcpe_feather(
                tmp_path / "input" / f"file_{i}.feather",
                file_positions,
            )
            file_paths.append(fp)

        # シングルスレッド実行
        single_count = _run_preprocess(
            file_paths=file_paths,
            output_dir=tmp_path / "output_single",
            max_workers=1,
        )

        # 並列実行
        parallel_count = _run_preprocess(
            file_paths=file_paths,
            output_dir=tmp_path / "output_parallel",
            max_workers=2,
        )

        assert single_count == parallel_count, (
            f"シングルスレッドのユニーク局面数({single_count})と"
            f"並列処理のユニーク局面数({parallel_count})が一致しない"
        )
        assert single_count == n_distinct

    def test_parallel_cross_file_duplicates(
        self, tmp_path: Path
    ) -> None:
        """並列処理でファイル間重複が正しく処理されることを検証する．"""
        positions = _generate_distinct_positions(6)

        # 全ファイルに全局面を含める(完全重複)
        file_paths = []
        for i in range(4):
            fp = _create_hcpe_feather(
                tmp_path / "input" / f"dup_{i}.feather",
                positions,
            )
            file_paths.append(fp)

        # 並列実行
        parallel_count = _run_preprocess(
            file_paths=file_paths,
            output_dir=tmp_path / "output",
            max_workers=2,
        )

        assert parallel_count == 6, (
            f"4ファイルに同じ6局面を含む入力の並列処理で"
            f"ユニーク局面数は6であるべきだが，{parallel_count}だった"
        )


class TestZobristHashCorrectness:
    """zobrist_hashに基づくユニーク判定の正確性を直接検証するテストクラス．"""

    def test_same_position_same_hash(self) -> None:
        """同一局面は同一のzobrist_hashを返すことを検証する．"""
        board1 = Board()
        board2 = Board()

        # 同じSFENからセットした2つの盤面
        sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        board1.set_sfen(sfen)
        board2.set_sfen(sfen)

        assert board1.hash() == board2.hash()

    def test_different_positions_different_hash(self) -> None:
        """異なる局面は異なるzobrist_hashを返すことを検証する．"""
        board = Board()

        # 初期局面のハッシュ
        hash_initial = board.hash()

        # 1手指した後のハッシュ
        moves = list(board.get_legal_moves())
        board.push_move(moves[0])
        hash_after_move = board.hash()

        assert hash_initial != hash_after_move

    def test_hcp_roundtrip_preserves_hash(self) -> None:
        """HCP経由のラウンドトリップでzobrist_hashが保存されることを検証する．

        preprocessでは board.set_hcp() → board.hash() の順で
        ハッシュを計算するため，HCPラウンドトリップでの一致が必須．
        """
        board = Board()
        positions = _generate_distinct_positions(20)

        for hcp_bytes, _, expected_hash in positions:
            hcp_array = np.frombuffer(hcp_bytes, dtype=np.uint8)
            board.set_hcp(hcp_array)
            actual_hash = board.hash()

            assert actual_hash == expected_hash, (
                f"HCPラウンドトリップ後のhashが一致しない: "
                f"expected={expected_hash}, actual={actual_hash}"
            )

    def test_process_single_array_unique_count(self) -> None:
        """_process_single_arrayが正しいユニーク局面数を返すことを検証する．

        preprocessパイプラインの最もコアなロジックを直接テストする．
        """
        n_distinct = 5
        positions = _generate_distinct_positions(n_distinct)
        expected_hashes = {p[2] for p in positions}

        # 重複を含むデータを作成
        duplicated = positions * 3  # 5局面 × 3 = 15レコード

        # numpy structured arrayを作成
        from maou.domain.data.schema import get_hcpe_dtype

        dtype = get_hcpe_dtype()
        data = np.zeros(len(duplicated), dtype=dtype)

        for i, (hcp_bytes, m16, _) in enumerate(duplicated):
            data["hcp"][i] = np.frombuffer(
                hcp_bytes, dtype=np.uint8
            )
            data["bestMove16"][i] = m16
            data["gameResult"][i] = 1

        # _process_single_arrayを直接呼び出す
        result = PreProcess._process_single_array(data)

        assert len(result) == n_distinct, (
            f"_process_single_arrayは{n_distinct}個のユニーク局面を返すべきだが，"
            f"{len(result)}個だった"
        )

        # 返されたhash_idがexpected_hashesと一致するか
        result_hashes = set(result.keys())
        assert result_hashes == expected_hashes, (
            f"_process_single_arrayが返したhash_idセットが期待と異なる"
        )

        # 各ユニーク局面のcountが3であることを確認
        for hash_id, data_dict in result.items():
            assert data_dict["count"] == 3, (
                f"hash_id={hash_id}のcountは3であるべきだが，"
                f"{data_dict['count']}だった"
            )
