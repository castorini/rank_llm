"""Cleanup for model coordinators owned by one pipeline invocation."""

import logging
import sys


def close_owned_model(coordinator):
    close = getattr(coordinator, "close", None)
    if callable(close):
        failed = sys.exc_info()[0] is not None
        try:
            close()
        except Exception:
            if not failed:
                raise
            logging.getLogger(__name__).exception(
                "Model cleanup failed while handling a pipeline exception"
            )
