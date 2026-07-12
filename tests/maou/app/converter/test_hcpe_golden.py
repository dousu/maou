"""HCPE 変換の golden fixture 一致検証 (cshogi 遺構リファクタの parity gate)．

golden はリファクタ前実装の出力を固定したもの．
生成器: scratchpad/gen_refactor_golden.py (リファクタ前実装で実行．
gitignore されているため，再生成手順はこの docstring とともに保守する)．

HCPE 変換の実装を置き換えても，これらのテストが green である限り
出力は bit-exact に保たれている．意図的な挙動変更
(複数局 CSA 全変換 / cp932 KIF 対応) を有効化するコミットでは，
該当テストの期待値を expected_* golden へ明示的に切り替えること．
"""

from pathlib import Path

import polars as pl
import pyarrow.feather as paf
import pytest
from polars.testing import assert_frame_equal

from maou.app.converter.hcpe_converter import HCPEConverter

INPUT_DIR = Path(
    "tests/maou/app/converter/resources/test_dir/input"
)
GOLDEN_DIR = Path("tests/maou/app/converter/resources/golden")


def _convert(
    paths: list[Path], input_format: str, output_dir: Path
) -> dict[str, str]:
    option = HCPEConverter.ConvertOption(
        input_paths=paths,
        input_format=input_format,
        output_dir=output_dir,
        min_rating=None,
        min_moves=None,
        max_moves=None,
        allowed_endgame_status=None,
        max_workers=1,
        chunk_size=0,
    )
    return HCPEConverter().convert(option)


class TestHcpeGolden:
    @pytest.mark.parametrize(
        "input_name,input_format,golden_name",
        [
            (
                "test_data_1.csa",
                "csa",
                "csa_test_data_1.feather",
            ),
            (
                "test_data_2.csa",
                "csa",
                "csa_test_data_2.feather",
            ),
            (
                "test_data_3.csa",
                "csa",
                "csa_test_data_3.feather",
            ),
            (
                "floodgate2025_sennichite.csa",
                "csa",
                "csa_floodgate2025_sennichite.feather",
            ),
            (
                "floodgate2025_timeup.csa",
                "csa",
                "csa_floodgate2025_timeup.feather",
            ),
            (
                "floodgate2025_toryo.csa",
                "csa",
                "csa_floodgate2025_toryo.feather",
            ),
            (
                "test_data_1.kifu",
                "kif",
                "kif_test_data_1.feather",
            ),
            (
                "test_data_no_date.kifu",
                "kif",
                "kif_test_data_no_date.feather",
            ),
        ],
    )
    def test_single_game_matches_golden(
        self,
        tmp_path: Path,
        input_name: str,
        input_format: str,
        golden_name: str,
    ) -> None:
        """単一局の変換出力が golden と bit-exact に一致する．"""
        input_path = INPUT_DIR / input_name
        result = _convert([input_path], input_format, tmp_path)

        golden = pl.read_ipc(GOLDEN_DIR / golden_name)
        actual = pl.read_ipc(
            tmp_path / input_path.with_suffix(".feather").name
        )
        assert_frame_equal(actual, golden)
        # status 文字列も互換仕様の一部として固定する
        assert (
            result[str(input_path)]
            == f"success {golden.height} rows"
        )

    def test_multi_game_csa(self, tmp_path: Path) -> None:
        """複数局 CSA の変換挙動．

        現行実装は先頭 1 局のみ変換する (残局は破棄)．
        Rust パイプライン切替コミットで全局変換
        (golden/csa_multi_game.expected_all_games.feather,
        id: game0 は従来形式, game>=1 は {stem}.hcpe_g{g}_{idx})
        へ期待値を切り替える (承認済み挙動変更)．
        """
        input_path = INPUT_DIR / "csa_multi_game.csa"
        _convert([input_path], "csa", tmp_path)

        golden = pl.read_ipc(
            GOLDEN_DIR
            / "csa_multi_game.current_first_game.feather"
        )
        actual = pl.read_ipc(
            tmp_path / "csa_multi_game.feather"
        )
        assert_frame_equal(actual, golden)

    def test_sjis_kif(self, tmp_path: Path) -> None:
        """cp932 エンコードの .kif の変換挙動．

        現行実装は read_text() が UTF-8 固定のためデコードエラーになる
        (sequential モードは例外を再送出する)．
        Rust パイプライン切替コミットで UTF-8→cp932 fallback により
        UTF-8 版と同一内容 (id の stem のみ異なる,
        golden/kif_test_data_sjis.expected.feather) へ期待値を
        切り替える (承認済み挙動変更)．
        """
        input_path = INPUT_DIR / "test_data_sjis.kif"
        with pytest.raises(
            Exception, match="codec can't decode"
        ):
            _convert([input_path], "kif", tmp_path)

    def test_arrow_schema_stable(self, tmp_path: Path) -> None:
        """出力 .feather の Arrow schema が固定されている．

        Rust 側で RecordBatch を直接組み立てる実装 (maou_convert) が
        現行 Polars 出力と同一 schema を生成することの検証に使う．
        """
        input_path = INPUT_DIR / "test_data_1.csa"
        _convert([input_path], "csa", tmp_path)

        schema = paf.read_table(
            tmp_path / "test_data_1.feather"
        ).schema
        golden_schema = (
            (GOLDEN_DIR / "hcpe_schema.txt")
            .read_text(encoding="utf-8")
            .rstrip("\n")
        )
        assert str(schema) == golden_schema
