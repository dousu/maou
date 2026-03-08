"""プロセス情報取得ユーティリティ．"""

from __future__ import annotations

import platform
import resource


def get_rss_mb() -> int:
    """現在のRSS(Resident Set Size)をMB単位で返す．

    Returns:
        RSS(MB単位)．macOS ではバイト単位，Linux では KB 単位の
        ru_maxrss を正規化して返す．
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        # macOS: ru_maxrss はバイト単位
        return rss // (1024 * 1024)
    # Linux: ru_maxrss は KB 単位
    return rss // 1024
