EXIT_CODES = {
    "success": 0,
    "invalid_arguments": 2,
    "missing_prerequisite": 3,
    "missing_resource": 4,
    "validation_error": 5,
    "provider_error": 6,
    "runtime_error": 6,
    "partial_success": 7,
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
