import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

# class JsonLogFormatter(logging.Formatter):
#     def format(self, record):
#         # https://docs.python.org/ja/3/library/logging.html
#         log_record = {
#             "time": self.formatTime(record, self.datefmt),
#             "name": record.name,
#             "level": record.levelname,
#             "message": record.getMessage(),
#             "filename": record.filename,
#             "lineno": record.lineno,
#         }
#         return json.dumps(log_record)


class CustomLogFormatter(logging.Formatter):
    # 時刻はJSTを使用する
    def formatTime(
        self, record: logging.LogRecord, datefmt: Optional[str] = None
    ) -> str:
        tz_jst = timezone(timedelta(hours=+9), "JST")
        ct = datetime.fromtimestamp(record.created, tz=tz_jst)
        s = ct.isoformat(timespec="microseconds")

        return s


def get_log_level_from_env() -> int:
    """
    環境変数MAOU_LOG_LEVELからログレベルを取得する．
    環境変数が設定されていない場合，INFOレベルを返す．
    """
    log_level = os.getenv("MAOU_LOG_LEVEL", "INFO").upper()

    level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    return level_mapping.get(log_level, logging.INFO)


handler = logging.StreamHandler()
# jsonにフォーマットするとトレースバックとか見えないのでいったんコメントアウトしておく
# handler.setFormatter(JsonLogFormatter())
formatter = CustomLogFormatter(
    "%(asctime)s | "
    "%(levelname)-5s | "
    "%(filename)20s | "
    "%(funcName)20s | "
    "%(lineno)3d | "
    "%(threadName)s | "
    "%(message)s"
)
handler.setFormatter(formatter)
logging.basicConfig()

app_logger: logging.Logger = logging.getLogger("maou")
app_logger.setLevel(get_log_level_from_env())
app_logger.addHandler(handler)
app_logger.propagate = False
