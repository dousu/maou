"""``test_ratio`` / ``stage2_test_ratio`` の既定値解決に関する回帰テスト．

`/audit-backlog` 2026-08-12 / backlog 行 O4．

`x or DEFAULT` は ``0.0`` を falsy として握り潰す．影響は 2 通りある:

- ``--stage2-test-ratio`` は **0.0 が CLI の既定値かつ documented な
  「分割なし」** なので，`or 0.2` は毎回 20% を黙って持っていっていた．
- ``--test-ratio`` の ``0.0`` は ``learn()`` が拒否する値だが，検証が
  Stage 3 の入口にしかなかったため，Stage 1 と Stage 2 を回し切った
  あとで初めて ``ValueError`` になっていた．
"""

from __future__ import annotations

import inspect

import pytest

from maou.interface import learn as learn_interface


def test_learn_multi_stage_rejects_zero_test_ratio_up_front() -> (
    None
):
    """``test_ratio=0.0`` が Stage 1 を回す前に弾かれること．

    データソースを一切渡していないので，検証が入口に無ければ
    「stage1_data_config is required」など別の ``ValueError`` が先に
    出る．``test_ratio`` の文言で落ちることが，検証がその手前にある
    ことの証拠になる．
    """
    with pytest.raises(
        ValueError, match="test_ratio must be between 0 and 1"
    ):
        learn_interface.learn_multi_stage(
            stage="all",
            test_ratio=0.0,
        )


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
def test_learn_multi_stage_rejects_out_of_range_test_ratio(
    bad: float,
) -> None:
    """範囲外の ``test_ratio`` はすべて入口で弾かれること．"""
    with pytest.raises(
        ValueError, match="test_ratio must be between 0 and 1"
    ):
        learn_interface.learn_multi_stage(
            stage="all",
            test_ratio=bad,
        )


def test_learn_multi_stage_accepts_none_test_ratio() -> None:
    """``test_ratio=None`` は入口を素通りすること．

    None は「既定値を使う」の意味なので，``test_ratio`` の検証では
    落ちてはならない (後続の別の検証で落ちるのが正しい)．
    """
    with pytest.raises(ValueError) as excinfo:
        learn_interface.learn_multi_stage(
            stage="all",
            test_ratio=None,
        )

    assert "test_ratio" not in str(excinfo.value)


def test_no_falsy_or_default_on_ratio_arguments() -> None:
    """``ratio or DEFAULT`` の書き方が残っていないこと．

    0.0 を握り潰す書き方はこの 1 パターンで再発するので，ソースを
    直接見て固定する (実行経路が GPU / BigQuery を要求するものも
    あるため，値ベースでは全サイトを覆えない)．
    """
    from maou.infra.console import learn_model, utility
    from maou.interface import utility_interface

    offenders: list[str] = []
    for module in (
        learn_model,
        utility,
        utility_interface,
        learn_interface,
    ):
        source = inspect.getsource(module)
        for lineno, line in enumerate(
            source.splitlines(), start=1
        ):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "test_ratio or " in stripped:
                offenders.append(
                    f"{module.__name__}:{lineno}: {stripped}"
                )

    assert not offenders, (
        "`test_ratio or DEFAULT` silently rewrites 0.0:\n"
        + "\n".join(offenders)
    )


def test_default_test_ratio_matches_documented_value() -> None:
    """``--test-ratio`` の既定値が doc の値と一致すること．

    `docs/commands/learn_model.md` は ``interface default 0.2`` と
    明記している．定数を動かすと doc が黙って嘘になるので固定する．
    """
    assert learn_interface.DEFAULT_TEST_RATIO == 0.2


def test_stage3_file_split_uses_the_shared_default() -> None:
    """Stage 3 のファイル単位分割が直値の既定を持たないこと．

    `/audit-backlog` 2026-08-12 / backlog 行 N1．

    console 側は ``--test-ratio`` 未指定時に 0.1，interface 側は 0.2 を
    使っていた．streaming 経路では分割済みソースを console が作るので
    interface 側の値は使われず，``--no-streaming`` に切り替えた瞬間に
    検証割合が 10% から 20% へ黙って変わる．

    値ベースで固定できない: この分岐は S3/GCS の streaming 経路の
    奥にあり，到達させるにはクラウド資格情報が要る．そこで
    「既定を引く側が共有定数を参照していること」をソースで固定する
    (同ファイルの `test_no_falsy_or_default_on_ratio_arguments` と
    同じ方針)．
    """
    from maou.infra.console import learn_model

    source = inspect.getsource(learn_model)
    lines = source.splitlines()

    offenders: list[str] = []
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # `test_ratio if test_ratio is not None else <直値>` の
        # else 側に数値が直接置かれていないこと．
        if stripped.startswith("else ") and any(
            "test_ratio is not None" in prev.strip()
            for prev in lines[max(0, lineno - 3) : lineno]
        ):
            tail = stripped[len("else ") :].strip()
            if tail and tail[0].isdigit():
                offenders.append(
                    f"learn_model:{lineno}: {stripped}"
                )

    assert not offenders, (
        "Stage 3 の分割が interface と別の既定値を持っている "
        "(learn.DEFAULT_TEST_RATIO を参照すること):\n"
        + "\n".join(offenders)
    )
