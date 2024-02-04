import os
import sys
from enum import Enum

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)


class EstimationMode(Enum):
    MAX_CONTEXT_LENGTH = "max_context_length"
    CREATE_PROMPTS = "create_prompts"

    def __str__(self):
        return self.value
