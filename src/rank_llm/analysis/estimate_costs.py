import json
import os
import sys
from argparse import ArgumentParser
from enum import Enum

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.rerank.rank_gpt import SafeOpenai
from rank_llm.rerank.rankllm import PromptMode
from rank_llm.retrieve.pyserini_retriever import PyseriniRetriever, RetrievalMethod
from rank_llm.retrieve.topics_dict import TOPICS


class EstimationMode(Enum):
    MAX_CONTEXT_LENGTH = "max_context_length"
    CREATE_PROMPTS = "create_prompts"

    def __str__(self):
        return self.value
