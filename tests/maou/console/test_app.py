from maou.infra.console import app


def test_hoge() -> None:
    assert app.hoge() == "hoge"
