import asyncio
from typing import Any, Dict, List, Sequence, Tuple, Union

from openai import OpenAI, AsyncOpenAI
from transformers import AutoTokenizer, PreTrainedTokenizerBase

Message = Dict[str, str]
PromptLike = Union[str, Message, Sequence[Message]]

class VllmHandlerWithOpenAISDK:
    def __init__(
        self,
        base_url: str,
        max_output_tokens: int,
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
        self._max_output_tokens = max_output_tokens

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    def _normalize_messages(self, prompt: str |list[dict[str, str]]):
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt

    async def _one_inference(self, prompt: str |list[dict[str, str]], **kwargs) -> Tuple[str, int]:
        messages = self._normalize_messages(prompt)
        #kwargs = {"reasoning": {"effort": "medium", "summary": "detailed"}}
        try: 
            response = await self._client.responses.create(
                model=self._model,
                input=messages,
                temperature=0,
                max_output_tokens=self._max_output_tokens,
                # reasoning={"effort": "medium", "summary": "detailed"},
                **kwargs,
            )
            text = response.output_text
            toks = len(self._tokenizer.encode(text))    
            return text, toks
        except Exception as e:
            print(f"heyyy exception happened, {e}")
            print(f"heyyy input prompt causing the exception: {messages}")
            return "error", 0
        

    async def _all_inferences(self, prompts: list[str]|list[list[dict[str, str]]], **kwargs) -> List[Tuple[str, int]]:
        tasks = [asyncio.create_task(self._one_inference(p, **kwargs)) for p in prompts]
        return await asyncio.gather(*tasks)

    # Optional sync wrapper (OK in plain scripts; avoid inside Jupyter/FastAPI)
    def chat_completions(self, prompts: list[str]|list[list[dict[str, str]]], **kwargs) -> List[Tuple[str, int]]:
        return asyncio.run(self._all_inferences(prompts, **kwargs))
