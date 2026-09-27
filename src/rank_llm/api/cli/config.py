from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

_APP_NAME = "rank_llm"


def _xdg_config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))


def _config_paths() -> list[Path]:
    return [
        Path(".rank-llm.toml"),
        Path(f".{_APP_NAME}.toml"),
        _xdg_config_home() / _APP_NAME / "config.toml",
    ]


def load_config() -> tuple[dict[str, Any], Path | None]:
    """Load CLI defaults from the first config file found.

    Returns (defaults_dict, resolved_path). When no file is found the dict is
    empty and the path is ``None``.
    """

    for path in _config_paths():
        if path.is_file():
            with open(path, "rb") as fh:
                return tomllib.load(fh), path
    return {}, None
