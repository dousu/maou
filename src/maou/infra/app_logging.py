import logging
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


handler = logging.StreamHandler()
# jsonにフォーマットするとトレースバックとか見えないのでいったんコメントアウトしておく
# handler.setFormatter(JsonLogFormatter())
handler.setFormatter(
    CustomLogFormatter(
        "%(asctime)s | "
        "%(levelname)-5s | "
        "%(filename)20s | "
        "%(funcName)20s | "
        "%(lineno)3d | "
        "%(threadName)s | "
        "%(message)s"
    )
)
logging.basicConfig()

app_logger: logging.Logger = logging.getLogger("maou")
app_logger.setLevel(logging.DEBUG)
app_logger.addHandler(handler)
app_logger.propagate = False
