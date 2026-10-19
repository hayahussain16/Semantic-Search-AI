import asyncio
import logging
import os
import time
from typing import Type, TypeVar

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

load_dotenv()

logger = logging.getLogger("llm_client")

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Raised when the LLM call cannot be completed."""


def extract_json(raw: str) -> str:
    """Models sometimes wrap JSON in extra text. Keep only the outermost {...}."""
    start, end = raw.find("{"), raw.rfind("}")
    return raw[start: end + 1] if start != -1 and end != -1 else raw


class ProductionLLMClient:
    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.model = model or os.getenv("LLM_MODEL")
        self.max_retries = max_retries

        self.client = AsyncOpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL"),
            api_key=api_key or os.getenv("LLM_API_KEY"),
            timeout=timeout,
            max_retries=0,  # we handle retries ourselves
        )

    async def chat(self, messages: list[dict], **kwargs) -> str:
        delay = 1.0

        for attempt in range(1, self.max_retries + 1):
            start = time.perf_counter()

            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs
                )

                content = resp.choices[0].message.content

                if not content:
                    raise LLMError("Empty response from model")

                logger.info(
                    "llm_ok model=%s attempt=%d latency=%.2fs tokens=%s",
                    self.model,
                    attempt,
                    time.perf_counter() - start,
                    getattr(resp.usage, "total_tokens", "n/a"),
                )

                return content

            except (APITimeoutError, RateLimitError, APIConnectionError) as e:
                retryable = True
                err = e

            except APIStatusError as e:
                retryable = e.status_code >= 500  # retry only server errors
                err = e

            if not retryable or attempt == self.max_retries:
                raise LLMError(f"LLM call failed: {err}") from err

            logger.warning(
                "llm_retry attempt=%d error=%s wait=%.1fs",
                attempt,
                err,
                delay,
            )

            await asyncio.sleep(delay)
            delay *= 2  # exponential backoff

        raise LLMError("Unreachable")

    async def generate_structured(
        self,
        messages: list[dict],
        schema: Type[T],
        max_attempts: int = 3,
    ) -> T:
        import json

        hint = json.dumps(schema.model_json_schema())

        msgs = [
            {
                "role": "system",
                "content": (
                    "Return ONLY valid JSON that matches this JSON schema. "
                    "No markdown, no explanation.\n"
                    + hint
                ),
            }
        ] + messages

        last_error = None

        for attempt in range(1, max_attempts + 1):
            raw = await self.chat(msgs, temperature=0)

            try:
                return schema.model_validate_json(extract_json(raw))

            except ValidationError as e:
                last_error = e

                logger.warning(
                    "validation_failed attempt=%d",
                    attempt,
                )

                msgs = msgs + [
                    {
                        "role": "assistant",
                        "content": raw,
                    },
                    {
                        "role": "user",
                        "content": (
                            "That JSON was invalid: "
                            + str(e)[:500]
                            + "\nFix it. Return ONLY the corrected JSON."
                        ),
                    },
                ]

        raise LLMError(
            f"Could not get valid structured output: {last_error}"
        )