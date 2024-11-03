import logging
from datetime import datetime, timedelta, timezone

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
    def formatTime(self, record: logging.LogRecord, datefmt=None):
        tz_jst = timezone(timedelta(hours=+9), "JST")
        ct = datetime.fromtimestamp(record.created, tz=tz_jst)
        s = ct.isoformat(timespec="microseconds")

        return s


handler = logging.StreamHandler()
# jsonにフォーマットするとトレースバックとか見えないのでいったんコメントアウトしておく
# handler.setFormatter(JsonLogFormatter())
handler.setFormatter(
    CustomLogFormatter(
        "%(asctime)s | %(levelname)-5s | %(filename)15s | %(funcName)15s | %(lineno)3d | %(threadName)s | %(message)s"
    )
)
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        # logging.FileHandler("app.log"),  # ログをファイルに出力
        handler,
    ],
)

app_logger: logging.Logger = logging.getLogger("Application Log")
