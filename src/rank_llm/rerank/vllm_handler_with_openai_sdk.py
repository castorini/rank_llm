import asyncio
from typing import Dict, List, Sequence, Tuple, Union

from openai import AsyncOpenAI, OpenAI
from transformers import AutoTokenizer, PreTrainedTokenizerBase

Message = Dict[str, str]
PromptLike = Union[str, Message, Sequence[Message]]


class VllmHandlerWithOpenAISDK:
    def __init__(
        self,
        base_url: str,
        model: str | None = None,
    ):
        # async client for inference, but no awaiting here
        self._client = AsyncOpenAI(api_key="", base_url=base_url)

        # if model isn't provided, use the SYNC client to list models
        if model is None:
            sync = OpenAI(api_key="", base_url=base_url)
            models = sync.models.list()
            if not models.data:
                raise RuntimeError("No models available from vLLM /v1/models.")
            model = models.data[0].id

        self._model = model
        self._tokenizer = AutoTokenizer.from_pretrained(model)

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    async def _one_inference(
        self, messages: list[dict[str, str]], **kwargs
    ) -> Tuple[str, int]:
        assert isinstance(messages, list)
        assert isinstance(messages[0], dict)
        response = None
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                **kwargs,
            )
            text = response.choices[0].message.content
            toks = len(self._tokenizer.encode(text))
            return text, toks
        except Exception as e:
            print(response)
            print(e)
            return "error", 0

    async def _all_inferences(
        self, prompts: list[str] | list[list[dict[str, str]]], **kwargs
    ) -> List[Tuple[str, int]]:
        tasks = [asyncio.create_task(self._one_inference(p, **kwargs)) for p in prompts]
        return await asyncio.gather(*tasks)

    # Optional sync wrapper (OK in plain scripts; avoid inside Jupyter/FastAPI)
    def chat_completions(
        self, prompts: list[str] | list[list[dict[str, str]]], **kwargs
    ) -> List[Tuple[str, int]]:
        return asyncio.run(self._all_inferences(prompts, **kwargs))
