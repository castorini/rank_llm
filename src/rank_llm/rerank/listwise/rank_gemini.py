import time
from typing import Any

from tqdm import tqdm

from rank_llm._optional import missing_extra_error
from rank_llm.data import Request, Result
from rank_llm.rerank.rankllm import PromptMode

from .listwise_rankllm import ListwiseRankLLM

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


def populate_generation_config(**kwargs) -> dict[str, Any]:
    # TODO: complete this for the rest of the optional generation params.
    generation_config = {"response_mime_type": "text/plain"}
    if "temperature" in kwargs:
        generation_config["temperature"] = kwargs["temperature"]
    if "top_p" in kwargs:
        generation_config["top_p"] = kwargs["top_p"]
    if "top_k" in kwargs:
        generation_config["top_k"] = kwargs["top_k"]
    if "max_output_tokens" in kwargs:
        generation_config["max_output_tokens"] = kwargs["max_output_tokens"]
    return generation_config


class SafeGenai(ListwiseRankLLM):
    def __init__(
        self,
        model: str,
        context_size: int,
        prompt_mode: PromptMode | None = None,
        prompt_template_path: str | None = None,
        num_few_shot_examples: int = 0,
        few_shot_file: str | None = None,
        window_size: int = 20,
        stride: int = 10,
        batch_size: int = 32,
        keys=None,
        key_start_id=None,
        max_passage_words: int = 300,
        **kwargs,
    ):
        super().__init__(
            model=model,
            context_size=context_size,
            prompt_mode=prompt_mode,
            prompt_template_path=prompt_template_path,
            num_few_shot_examples=num_few_shot_examples,
            few_shot_file=few_shot_file,
            window_size=window_size,
            stride=stride,
            batch_size=batch_size,
            max_passage_words=max_passage_words,
        )
        if not genai or not types:
            raise missing_extra_error(
                "genai",
                "The Gemini reranker requires google-genai.",
            )
        if isinstance(keys, str):
            keys = [keys]
        if not keys:
            raise ValueError("Please provide Genai API Keys.")
        self._output_token_estimate = None
        self._keys = keys
        self._cur_key_id = key_start_id or 0
        self._cur_key_id = self._cur_key_id % len(self._keys)
        self.generation_config = populate_generation_config(**kwargs)
        self.safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_NONE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_NONE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_NONE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_NONE",
            ),
        ]
        self.system_instruction = kwargs.get(
            "system_instruction",
            "As RankGemini, your task is to evaluate and rank unique passages based on their relevance and accuracy to a given query. Prioritize passages that directly address the query and provide detailed, correct answers. Ignore factors such as length, complexity, or writing style unless they seriously hinder readability.",
        )
        self._request_config = types.GenerateContentConfig(
            **self.generation_config,
            system_instruction=self.system_instruction,
            safety_settings=self.safety_settings,
        )
        self._set_client()

    def _set_client(self) -> None:
        self.client = genai.Client(api_key=self._keys[self._cur_key_id])

    @staticmethod
    def _message_content(message: dict[str, Any]) -> str:
        content = message.get("content") or message.get("parts") or ""
        if isinstance(content, list):
            return "\n".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)

    @staticmethod
    def _message_role(message: dict[str, Any]) -> str:
        role = message.get("role", "user")
        return "model" if role in {"assistant", "model"} else "user"

    def _to_genai_history(self, messages: list[dict[str, Any]]) -> list[Any]:
        return [
            types.Content(
                role=self._message_role(message),
                parts=[types.Part.from_text(text=self._message_content(message))],
            )
            for message in messages
        ]

    def rerank_batch(
        self,
        requests: list[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        **kwargs: Any,
    ) -> list[Result]:
        top_k_retrieve: int = kwargs.get("top_k_retrieve", rank_end)
        rank_end = min(top_k_retrieve, rank_end)
        populate_invocations_history: bool = kwargs.get(
            "populate_invocations_history", False
        )
        results = []
        for request in tqdm(requests):
            result = self.sliding_windows(
                request,
                rank_start=max(rank_start, 0),
                rank_end=min(rank_end, len(request.candidates)),
                top_k_retrieve=top_k_retrieve,
                shuffle_candidates=shuffle_candidates,
                logging=logging,
                populate_invocations_history=populate_invocations_history,
            )
            results.append(result)
        return results

    def run_llm_batched(self):
        pass

    def _call_inference(self, messages, return_text=False) -> str | dict[str, Any]:
        while True:
            try:
                if isinstance(messages, list):
                    history = messages[:-1]
                    chat_message = messages[-1]
                    chat_session = self.client.chats.create(
                        model=self._model,
                        config=self._request_config,
                        history=self._to_genai_history(history),
                    )
                    completion = chat_session.send_message(
                        self._message_content(chat_message)
                    )
                else:
                    completion = self.client.models.generate_content(
                        model=self._model,
                        contents=messages,
                        config=self._request_config,
                    )
                break
            except Exception as e:
                print("Error in completion call")
                print(str(e))
                # TODO: do not retry for some of the deterministic failures.
                self._cur_key_id = (self._cur_key_id + 1) % len(self._keys)
                self._set_client()
                time.sleep(1.0)

        if return_text:
            return completion.text
        return completion

    def run_llm(
        self,
        prompt: str | list[dict[str, Any]],
        current_window_size: int | None = None,
    ) -> tuple[str, int]:
        response = self._call_inference(
            messages=prompt,
            return_text=True,
        )
        return response, self._count_tokens(response)

    # TODO (issue #256): Need to modify gemini implementation to use OpenAI's API and then add fewshot examples
    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int
    ) -> tuple[str, int]:
        max_length = self._max_passage_words
        while True:
            message = self._inference_handler.generate_prompt(
                result=result,
                rank_start=rank_start,
                rank_end=rank_end,
                max_length=max_length,
            )[-1]["content"]
            num_tokens = self.get_num_tokens(message)
            if num_tokens <= self.max_tokens() - self.num_output_tokens():
                break
            else:
                max_length -= max(
                    1,
                    (num_tokens - self.max_tokens() + self.num_output_tokens())
                    // ((rank_end - rank_start) * 4),
                )
        return message, self.get_num_tokens(message)

    def num_output_tokens(self, current_window_size: int | None = None) -> int:
        if current_window_size is None:
            current_window_size = self._window_size
        if self._output_token_estimate and self._window_size == current_window_size:
            return self._output_token_estimate
        else:
            _output_token_estimate = (
                self._count_tokens(
                    " > ".join([f"[{i + 1}]" for i in range(current_window_size)])
                )
                - 1
            )
            if (
                self._output_token_estimate is None
                and self._window_size == current_window_size
            ):
                self._output_token_estimate = _output_token_estimate
            return _output_token_estimate

    def create_prompt_batched(
        self, results: list[Result], rank_start: int, rank_end: int
    ) -> list[tuple[list[dict[str, Any]], int]]:
        return [self.create_prompt(result, rank_start, rank_end) for result in results]

    def get_num_tokens(self, prompt: str | list[dict[str, Any]]) -> int:
        """Returns the number of tokens used by a list of messages in prompt."""
        num_tokens = 0
        if isinstance(prompt, list):
            for message in prompt:
                num_tokens += self._count_tokens(self._message_content(message))
        else:
            response = self._count_tokens(prompt)
            num_tokens += response
        num_tokens += 3
        return num_tokens

    def _count_tokens(self, contents: str) -> int:
        return self.client.models.count_tokens(
            model=self._model,
            contents=contents,
        ).total_tokens

    def cost_per_1k_token(self, input_token: bool) -> float:
        # TODO: add proper costs
        return 0

    def get_name(self) -> str:
        return self._model
