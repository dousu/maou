import logging
from pathlib import Path

from maou.infra.app_logging import app_logger, formatter

log_file_path = Path("test.log")
# ログファイルを初期化する
# 場合によっては邪魔になりそうだが基本はデータ量減らす方針
log_file_path.write_text("")

logger = logging.getLogger("TEST")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(str(log_file_path.absolute()))
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

app_logger.addHandler(handler)
