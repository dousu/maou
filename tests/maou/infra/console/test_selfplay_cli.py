"""`maou selfplay` の CLI テスト (mock 評価器)．

Rust 側の対局セマンティクス (終局判定・並列・script 追従) は
rust/maou_usi/src/selfplay.rs の単体テストが担う．ここでは CLI 貫通と
JSONL 出力・サマリ表示を確認する．
"""

import json
from pathlib import Path

from click.testing import CliRunner

from maou.infra.console.selfplay import selfplay

# mock 評価器で数秒以内に完走する軽量設定
_FAST = [
    "--playouts",
    "16",
    "--max-moves",
    "16",
    "--node-capacity",
    "4096",
    "--no-root-dfpn",
    "--no-leaf-mate",
    "--quiet",
]


def test_selfplay_help() -> None:
    """--help が対局を実行せずに使い方を表示する．"""
    result = CliRunner().invoke(selfplay, ["--help"])
    assert result.exit_code == 0
    assert "self-play" in result.output
    assert "--games" in result.output


def test_selfplay_smoke_writes_jsonl(tmp_path: Path) -> None:
    """mock 2 局が完走し，JSONL 記録とサマリが出る．"""
    out = tmp_path / "records.jsonl"
    result = CliRunner().invoke(
        selfplay,
        [
            "--games",
            "2",
            "--opening-random-plies",
            "4",
            "--seed",
            "42",
            "--output",
            str(out),
            *_FAST,
        ],
    )
    assert result.exit_code == 0, result.output
    assert "games: 2" in result.output
    assert "results:" in result.output

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for i, line in enumerate(lines):
        record = json.loads(line)
        assert record["game_index"] == i
        assert record["plies"] == len(record["moves"])
        assert record["plies"] <= 16
        assert record["winner"] in ("black", "white", None)
        assert isinstance(record["reason"], str)
        assert record["sfen"].startswith("lnsgkgsnl/")


def test_selfplay_opening_script_prefix(tmp_path: Path) -> None:
    """--opening-script の手順を両側が追従する (強制手順対局)．"""
    out = tmp_path / "records.jsonl"
    script = "7g7f 3c3d 2g2f 8c8d"
    result = CliRunner().invoke(
        selfplay,
        [
            "--games",
            "1",
            "--opening-script",
            script,
            "--output",
            str(out),
            *_FAST,
        ],
    )
    assert result.exit_code == 0, result.output
    record = json.loads(
        out.read_text(encoding="utf-8").splitlines()[0]
    )
    assert record["moves"][:4] == script.split()


def test_selfplay_rejects_conflicting_budgets() -> None:
    """--playouts と --movetime-ms の同時指定はエラー．"""
    result = CliRunner().invoke(
        selfplay,
        ["--playouts", "16", "--movetime-ms", "100"],
    )
    assert result.exit_code != 0


def test_selfplay_reports_throughput() -> None:
    """壁時計ベースの playouts/秒 が出る (並列スケール計測の材料)．"""
    result = CliRunner().invoke(
        selfplay, ["--games", "1", *_FAST]
    )
    assert result.exit_code == 0, result.output
    assert "throughput:" in result.output
    assert "playouts/s" in result.output
    assert "parallel=1" in result.output


def test_selfplay_ab_mode_reports_a_view_statistics(
    tmp_path: Path,
) -> None:
    """--ab-mode で A 視点の勝率統計とペア比較が出る．"""
    out = tmp_path / "ab.jsonl"
    result = CliRunner().invoke(
        selfplay,
        [
            "--games",
            "2",
            "--ab-mode",
            "subtree",
            "--opening-random-plies",
            "2",
            "--seed",
            "5",
            "--output",
            str(out),
            *_FAST,
        ],
    )
    assert result.exit_code == 0, result.output
    assert "A/B mode: subtree" in result.output
    assert "Wilson 95% CI" in result.output
    assert "A Elo:" in result.output
    assert "paired: 1 pairs" in result.output

    # 色交代は ab-mode の既定 on (ペア 2 局で先後が入れ替わる)
    records = [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
    ]
    assert [r["black_player"] for r in records] == ["a", "b"]


def test_selfplay_clock_mode_tracks_remaining_time(
    tmp_path: Path,
) -> None:
    """持ち時間モードで残り時間が記録される (時間配分の診断)．"""
    out = tmp_path / "clock.jsonl"
    result = CliRunner().invoke(
        selfplay,
        [
            "--games",
            "1",
            "--clock-ms",
            "2000",
            "--inc-ms",
            "50",
            "--max-moves",
            "8",
            "--node-capacity",
            "4096",
            "--no-root-dfpn",
            "--no-leaf-mate",
            "--quiet",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    record = json.loads(
        out.read_text(encoding="utf-8").splitlines()[0]
    )
    remaining = record["remaining_ms"]
    assert len(remaining) == 2
    assert all(0 <= ms <= 2000 + 50 * 8 for ms in remaining)


def test_selfplay_rejects_clock_with_playouts() -> None:
    """持ち時間モードと playout 予算は排他．"""
    result = CliRunner().invoke(
        selfplay,
        ["--playouts", "16", "--clock-ms", "1000"],
    )
    assert result.exit_code != 0


def test_selfplay_rejects_horizon_without_clock() -> None:
    """--ab-mode horizon は持ち時間モードが必須．"""
    result = CliRunner().invoke(
        selfplay, ["--ab-mode", "horizon", *_FAST]
    )
    assert result.exit_code != 0


def test_selfplay_rejects_clock_with_parallel() -> None:
    """持ち時間モードは壁時計依存なので parallel=1 限定．"""
    result = CliRunner().invoke(
        selfplay,
        ["--clock-ms", "1000", "--parallel", "2"],
    )
    assert result.exit_code != 0
