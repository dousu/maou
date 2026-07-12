"""Rust HCPE 一括変換 (maou._rust.maou_convert) の golden parity 検証．

Phase 3 の parity gate: Rust パイプライン (convert_hcpe_str) の出力が
Phase 0 golden (リファクタ前 Python 実装の出力) と bit-exact に一致することを
`assert_frame_equal` (dtype 厳密) で検証する．この時点では Python の
hcpe_converter は未切替 (Rust 経路は併存)．

意図的な挙動変更 (複数局 CSA 全変換 / cp932 fallback) は「新挙動期待」golden
(expected_*) との一致で検証する．
"""

from pathlib import Path

import polars as pl
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal

from maou._rust.maou_convert import convert_hcpe_str

INPUT_DIR = Path(
    "tests/maou/app/converter/resources/test_dir/input"
)
GOLDEN_DIR = Path("tests/maou/app/converter/resources/golden")


def _convert(
    input_name: str, input_format: str, id_stem: str
) -> pl.DataFrame:
    content = (INPUT_DIR / input_name).read_bytes()
    # Rust 側の cp932 fallback を通すため bytes→str は Rust に任せたいが，
    # convert_hcpe_str は str 契約なので UTF-8 のみここでデコードする．
    # cp932 の検証は test_sjis_via_bytes で decode 経路 (convert_files) を使う．
    text = content.decode("utf-8")
    batch = convert_hcpe_str(
        text, input_format, f"{id_stem}.hcpe"
    )
    table = pa.Table.from_batches([batch])
    return pl.from_arrow(table)  # type: ignore[return-value]


class TestRustConvertParity:
    @pytest.mark.parametrize(
        "input_name,input_format,golden_name,id_stem",
        [
            (
                "test_data_1.csa",
                "csa",
                "csa_test_data_1.feather",
                "test_data_1",
            ),
            (
                "test_data_2.csa",
                "csa",
                "csa_test_data_2.feather",
                "test_data_2",
            ),
            (
                "test_data_3.csa",
                "csa",
                "csa_test_data_3.feather",
                "test_data_3",
            ),
            (
                "floodgate2025_sennichite.csa",
                "csa",
                "csa_floodgate2025_sennichite.feather",
                "floodgate2025_sennichite",
            ),
            (
                "floodgate2025_timeup.csa",
                "csa",
                "csa_floodgate2025_timeup.feather",
                "floodgate2025_timeup",
            ),
            (
                "floodgate2025_toryo.csa",
                "csa",
                "csa_floodgate2025_toryo.feather",
                "floodgate2025_toryo",
            ),
            (
                "test_data_1.kifu",
                "kif",
                "kif_test_data_1.feather",
                "test_data_1",
            ),
            (
                "test_data_no_date.kifu",
                "kif",
                "kif_test_data_no_date.feather",
                "test_data_no_date",
            ),
        ],
    )
    def test_single_game_matches_golden(
        self,
        input_name: str,
        input_format: str,
        golden_name: str,
        id_stem: str,
    ) -> None:
        """単一局の Rust 変換が Python golden と bit-exact に一致する．"""
        golden = pl.read_ipc(GOLDEN_DIR / golden_name)
        actual = _convert(input_name, input_format, id_stem)
        assert_frame_equal(actual, golden)

    def test_multi_game_all_games(self) -> None:
        """複数局 CSA を Rust が全局変換する (新挙動期待 golden と一致)．"""
        golden = pl.read_ipc(
            GOLDEN_DIR
            / "csa_multi_game.expected_all_games.feather"
        )
        actual = _convert(
            "csa_multi_game.csa", "csa", "csa_multi_game"
        )
        assert_frame_equal(actual, golden)

    def test_schema_matches_golden(self) -> None:
        """Rust 出力の Arrow schema が現行 Polars 出力と一致する．"""
        content = (INPUT_DIR / "test_data_1.csa").read_text(
            encoding="utf-8"
        )
        batch = convert_hcpe_str(
            content, "csa", "test_data_1.hcpe"
        )
        golden_schema = (
            (GOLDEN_DIR / "hcpe_schema.txt")
            .read_text(encoding="utf-8")
            .rstrip("\n")
        )
        assert str(batch.schema) == golden_schema


class TestRustConvertFiles:
    """convert_hcpe_files (ファイル直読み + cp932 decode + feather 書き出し)．"""

    def test_files_write_and_status(
        self, tmp_path: Path
    ) -> None:
        """複数ファイル一括変換で feather が書かれ status が返る．"""
        from maou._rust.maou_convert import convert_hcpe_files

        paths = [
            str(INPUT_DIR / "test_data_1.csa"),
            str(INPUT_DIR / "test_data_no_moves.csa"),
        ]
        results = dict(
            convert_hcpe_files(paths, "csa", str(tmp_path))
        )
        assert results[
            str(INPUT_DIR / "test_data_1.csa")
        ].startswith("success 163 rows")
        assert (
            results[str(INPUT_DIR / "test_data_no_moves.csa")]
            == "skipped (no moves)"
        )
        # 成功ファイルは golden と bit-exact
        golden = pl.read_ipc(
            GOLDEN_DIR / "csa_test_data_1.feather"
        )
        actual = pl.read_ipc(tmp_path / "test_data_1.feather")
        assert_frame_equal(actual, golden)
        # no-moves は出力なし
        assert not (
            tmp_path / "test_data_no_moves.feather"
        ).exists()

    def test_sjis_kif_decoded(self, tmp_path: Path) -> None:
        """cp932 の .kif がファイル直読み経路で UTF-8 版と同一内容に変換される．"""
        from maou._rust.maou_convert import convert_hcpe_files

        results = dict(
            convert_hcpe_files(
                [str(INPUT_DIR / "test_data_sjis.kif")],
                "kif",
                str(tmp_path),
            )
        )
        assert results[
            str(INPUT_DIR / "test_data_sjis.kif")
        ].startswith("success")
        golden = pl.read_ipc(
            GOLDEN_DIR / "kif_test_data_sjis.expected.feather"
        )
        actual = pl.read_ipc(
            tmp_path / "test_data_sjis.feather"
        )
        assert_frame_equal(actual, golden)
