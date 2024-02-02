from enum import Enum


class EstimationMode(Enum):
    MAX_CONTEXT_LENGTH = "max_context_length"
    CREATE_PROMPTS = "create_prompts"

    def __str__(self):
        return self.value
