EXIT_CODES = {
    "success": 0,
    "invalid_arguments": 2,
    "missing_resource": 4,
    "validation_error": 5,
    "runtime_error": 6,
}

KNOWN_COMMANDS = (
    "rerank",
    "evaluate",
    "analyze",
    "retrieve-cache",
    "serve",
    "view",
    "prompt",
    "describe",
    "schema",
    "doctor",
    "validate",
)

TOP_LEVEL_EXAMPLES = (
    "rank-llm rerank --help",
    "rank-llm doctor --output json",
)
