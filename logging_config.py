import logging
from pathlib import Path


class _ConsoleFilter(logging.Filter):
    """Allow only records intended for console: those starting with explicit prefixes
    or carrying the `show_in_console` attribute (for programmatic control).

    This keeps the terminal quiet except for USER_INPUT:, LLM_INPUT:, and LLM_OUTPUT: messages
    while still recording everything to the file log for debugging.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage() or ""
        except Exception:
            msg = ""
        prefixes = ("USER_INPUT:", "LLM_INPUT:", "LLM_OUTPUT:")
        if any(msg.startswith(p) for p in prefixes):
            return True
        if getattr(record, "show_in_console", False):
            return True
        return False


def configure_logging(data_dir: Path | str | None = None) -> None:
    data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data"
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "assistant.log"

    root = logging.getLogger()
    # remove existing handlers to avoid duplicated output when imported multiple times
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(logging.DEBUG)

    # File handler: record everything for later inspection
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)

    # Console handler: only show explicit short messages
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.addFilter(_ConsoleFilter())
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)


# Optional quick configure when imported as a script (keeps behaviour deterministic)
if __name__ == "__main__":
    configure_logging()
