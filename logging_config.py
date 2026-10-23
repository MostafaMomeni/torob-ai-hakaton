
import logging


class _ConsoleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage() or ""
        except Exception:
            msg = ""

        prefixes = ("USER_INPUT:", "LLM_INPUT:", "LLM_OUTPUT:")

        return (
            any(msg.startswith(prefix) for prefix in prefixes)
            or getattr(record, "show_in_console", False)
        )


def configure_logging(data_dir=None) -> None:
    root = logging.getLogger()

    # حذف handlerهای قبلی برای جلوگیری از تکرار خروجی
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    root.setLevel(logging.DEBUG)

    # فقط نمایش پیام‌های انتخاب‌شده در کنسول؛ بدون ذخیره در فایل
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(_ConsoleFilter())
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    root.addHandler(console_handler)


if __name__ == "__main__":
    configure_logging()
