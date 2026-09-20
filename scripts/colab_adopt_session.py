#!/usr/bin/env python
"""ブラウザで起動した Colab ランタイムを `colab` CLI のローカルセッションとして登録する．

`colab` CLI (googlecolab/google-colab-cli) の `exec` / `upload` / `status` は
`~/.config/colab-cli/sessions.json` に登録されたセッション名しか受け付けない．
ブラウザ (colab.research.google.com) で起動したランタイムは `colab sessions` に
名前 `?` で見えるだけで，CLI から触る手段が公式には無い．

しかし `colab sessions` が使う assignments 一覧には endpoint に加えて
runtime proxy の token / url が含まれており，`colab new` が保存しているのも
同じ 3 つなので，ここから `SessionState` を組み立てて登録すれば同等に扱える．
keep-alive デーモンは立てない (ブラウザのタブが keep-alive を担う前提)．

使い方 (colab CLI と同じ仮想環境の python で実行する):

    PY="$(uv tool dir)/google-colab-cli/bin/python"
    "$PY" scripts/colab_adopt_session.py gpu                  # 未登録が 1 つならそれを gpu として登録
    "$PY" scripts/colab_adopt_session.py gpu --endpoint m-s-… # 複数あるとき
    "$PY" scripts/colab_adopt_session.py gpu --force          # 既存の gpu を上書き (token 更新にも使う)
    "$PY" scripts/colab_adopt_session.py gpu --forget         # ローカル登録だけ消す (VM は止めない)

登録後は `colab status -s gpu` / `colab exec -s gpu` がそのまま使える．
**`colab stop -s gpu` は VM を unassign する** (ブラウザ側のランタイムも消える) ので，
ローカル登録だけ外したいときは `--forget` を使う．

`colab_cli` は 0.7.1 (GitHub 版) で検証．PyPI 0.6.0 は upstream の
jupyter-kernel-client 1.0.2 を拾って `exec` 自体が壊れるので使わない．
"""

from __future__ import annotations

import argparse
import sys

from colab_cli.common import state
from colab_cli.state import SessionState


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """コマンドライン引数を解釈する．"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "name", help="ローカルのセッション名 (`-s NAME` で使う)"
    )
    parser.add_argument(
        "--endpoint",
        help="登録する assignment の endpoint (`colab sessions` の 2 列目)．"
        "省略時は未登録の assignment が 1 つだけならそれを選ぶ",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="同名のローカル登録があっても上書きする (token の更新にも使う)",
    )
    parser.add_argument(
        "--forget",
        action="store_true",
        help="ローカル登録だけ削除して終了する (VM は unassign しない)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """エントリポイント．成功 0，失敗 1 を返す．"""
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if args.forget:
        if state.store.get(args.name) is None:
            print(
                f"[adopt] local session '{args.name}' not found",
                file=sys.stderr,
            )
            return 1
        state.store.remove(args.name)
        print(
            f"[adopt] forgot local session '{args.name}' (VM left running)"
        )
        return 0

    if (
        state.store.get(args.name) is not None
        and not args.force
    ):
        print(
            f"[adopt] local session '{args.name}' already exists (use --force)",
            file=sys.stderr,
        )
        return 1

    existing = state.store.get(args.name)
    local_sessions, assignments = state.sync_sessions()
    tracked = {s.endpoint for s in local_sessions.values()}

    if args.endpoint:
        candidates = [
            a
            for a in assignments
            if a.endpoint == args.endpoint
        ]
    elif existing is not None:
        # --force で同名を更新するときは，その登録が指す endpoint を引き継ぐ
        candidates = [
            a
            for a in assignments
            if a.endpoint == existing.endpoint
        ]
    else:
        candidates = [
            a for a in assignments if a.endpoint not in tracked
        ]

    if len(candidates) != 1:
        print(
            f"[adopt] expected exactly 1 candidate, found {len(candidates)}:",
            file=sys.stderr,
        )
        for a in assignments:
            mark = (
                "tracked"
                if a.endpoint in tracked
                else "untracked"
            )
            print(
                f"  {a.endpoint} | {a.accelerator.value} | {a.variant.name} | {mark}",
                file=sys.stderr,
            )
        print(
            "[adopt] pass --endpoint to choose one",
            file=sys.stderr,
        )
        return 1

    a = candidates[0]
    session = SessionState(
        name=args.name,
        token=a.runtime_proxy_info.token,
        url=a.runtime_proxy_info.url,
        endpoint=a.endpoint,
        variant=a.variant.name,
        accelerator=a.accelerator.value,
        machine_shape=a.machine_shape.name,
    )
    state.store.add(session)
    print(
        f"[adopt] '{args.name}' -> {a.endpoint} | {a.accelerator.value} | "
        f"{a.variant.name} | {a.machine_shape.name} "
        f"(token expires in {a.runtime_proxy_info.token_expires_in_seconds}s; "
        "re-run with --force to refresh)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
