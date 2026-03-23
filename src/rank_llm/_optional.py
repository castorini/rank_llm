def install_hint(extra: str) -> str:
    return f"Please install `rank-llm[{extra}]`."


def missing_extra_error(extra: str, detail: str | None = None) -> ImportError:
    message = install_hint(extra)
    if detail:
        message = f"{detail} {message}"
    return ImportError(message)
