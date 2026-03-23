try:
    from .duot5 import DuoT5
except ImportError:
    DuoT5 = None

__all__ = ["DuoT5"]
