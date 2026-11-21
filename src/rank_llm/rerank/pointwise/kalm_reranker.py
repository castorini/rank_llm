from importlib.resources import files
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from rank_llm.data import Result
from rank_llm.rerank.pointwise.pointwise_rankllm import PointwiseRankLLM
from rank_llm.utils import default_device

DEFAULT_TEMPLATE = files("rank_llm.rerank.prompt_templates") / "kalm_template.yaml"


def _load_checkpoint(model: str, **kwargs: Any) -> Any:
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    path = Path(model)
    checkpoint = str(path.resolve()) if path.is_dir() else snapshot_download(model)
    scorer = get_class_from_dynamic_module("kalm_reranker.KaLMReranker", checkpoint)
    return scorer(checkpoint, **kwargs)


class KaLMReranker(PointwiseRankLLM):
    """Pointwise KaLM reranking using the checkpoint's published Transformers scorer.

    The scorer encodes passages independently and scores decoder yes/no logits.
    Set encoder_chunk_size to the number of passage tokens per pooled vector, or
    None to disable pooling.
    """

    def __init__(
        self,
        model: str,
        prompt_template_path: str | None = None,
        context_size: int = 1024,
        device: str | None = None,
        batch_size: int = 32,
        max_passage_words: int = 300,
        encoder_chunk_size: int | None = 4,
    ) -> None:
        device = device or default_device()
        # The reference scorer uses separate input budgets for queries and passages.
        context_size = min(context_size, 1024)
        super().__init__(
            model=model,
            context_size=context_size,
            prompt_template_path=prompt_template_path or DEFAULT_TEMPLATE,
            device=device,
            batch_size=batch_size,
            max_passage_words=max_passage_words,
        )
        if self._inference_handler.template.get("message_roles") != "reranker":
            raise ValueError(
                'KaLM needs a pointwise template with message_roles: "reranker".'
            )
        self._scorer = _load_checkpoint(
            model,
            device=device,
            batch_size=batch_size,
            max_length=context_size,
            chunk_size=encoder_chunk_size,
        )

    def create_prompt(
        self, result: Result, index: int
    ) -> tuple[list[dict[str, str]], int]:
        messages = self._inference_handler.generate_chat_messages(
            result,
            index=index,
            tokenizer=self._scorer.tokenizer,
            max_doc_tokens=None,
            max_passage_words=self._max_passage_words,
        )
        return messages, self.get_num_tokens(messages)

    def run_llm_batched(
        self, prompts: list[list[dict[str, str]]]
    ) -> tuple[list[str], list[int], list[float]]:
        if not prompts:
            return [], [], []
        instruction = prompts[0][0]["content"]
        pairs = [(query["content"], doc["content"]) for _, query, doc in prompts]
        scores = self._scorer.predict(pairs, instruction=instruction)
        return (
            ["yes" if score >= 0.5 else "no" for score in scores],
            [self.num_output_tokens()] * len(scores),
            scores,
        )

    def run_llm(self, prompt: list[dict[str, str]]) -> tuple[str, int, float]:
        outputs, counts, scores = self.run_llm_batched([prompt])
        return outputs[0], counts[0], scores[0]

    def get_num_tokens(self, prompt: list[dict[str, str]]) -> int:
        # Count the message text before the scorer adds its model-specific framing.
        return sum(
            len(
                self._scorer.tokenizer.encode(
                    message["content"], add_special_tokens=False
                )
            )
            for message in prompt
        )

    def num_output_tokens(self) -> int:
        return 1

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0
