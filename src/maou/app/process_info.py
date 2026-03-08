"""プロセス情報取得ユーティリティ．"""

from __future__ import annotations

import platform

_IS_LINUX = platform.system() == "Linux"


def get_rss_mb() -> int:
    """現在のRSS(Resident Set Size)をMB単位で返す．

    Linux では /proc/self/status の VmRSS を読み取り，
    macOS では resource.getrusage の ru_maxrss(ピーク値)で代用する．

    Returns:
        現在の RSS(MB単位)．
    """
    if _IS_LINUX:
        return _read_vmrss_mb()
    # macOS: ru_maxrss はバイト単位のピーク値(現在値の取得手段がないため代用)
    import resource

    return resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss // (1024 * 1024)


def _read_vmrss_mb() -> int:
    """Linux の /proc/self/status から VmRSS を読み取る．"""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                # "VmRSS:    123456 kB" → KB 単位の数値を取得
                return int(line.split()[1]) // 1024
    return 0
