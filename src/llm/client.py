"""LM Studio client (Task B5 - response quality and reliability).

Thin wrapper over the OpenAI-compatible endpoint LM Studio exposes. Everything
Task B5 asks for lives here: empty-response detection, truncation detection via
finish_reason, timeout handling, and elapsed-time measurement for the UI's
loading indicator.

Nothing above this layer should ever touch the openai SDK directly.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import openai
from openai import OpenAI

from src.config import LLM


class LLMError(RuntimeError):
    """Raised when a completion cannot be produced or is unusable.

    Carrying a user-facing message means the Streamlit layer can surface the
    problem without knowing anything about the transport.
    """

    def __init__(self, message: str, *, kind: str = "error", elapsed: float = 0.0):
        super().__init__(message)
        self.kind = kind  # connection | timeout | empty | truncated | error
        self.elapsed = elapsed


@dataclass
class ToolCall:
    """One tool the model asked to invoke."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """A successful completion plus the telemetry Task B5 wants reported."""

    text: str
    elapsed_seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    truncated: bool = False
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.completion_tokens / self.elapsed_seconds

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


class LLMClient:
    """Chat client for the local Gemma model served by LM Studio."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ):
        self.base_url = base_url or LLM.base_url
        self.model = model or LLM.model
        self.timeout = timeout or LLM.timeout_seconds
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=api_key or LLM.api_key,
            timeout=self.timeout,
            max_retries=0,  # retries are handled explicitly by the pipeline
        )

    def health_check(self) -> tuple[bool, str]:
        """Confirm the server is up and the configured model is loaded.

        Called once at app start so a misconfigured endpoint fails loudly
        instead of surfacing as a confusing error on the first question.
        """
        try:
            models = self._client.models.list()
        except openai.APIConnectionError:
            return False, (
                f"Cannot reach LM Studio at {self.base_url}. "
                "Is the server running (Developer tab -> Status: Running)?"
            )
        except Exception as exc:  # noqa: BLE001 - surface anything else verbatim
            return False, f"Unexpected error contacting LM Studio: {exc}"

        available = [m.id for m in models.data]
        if not available:
            return False, "LM Studio is running but no model is loaded."
        if self.model not in available:
            return False, (
                f"Model {self.model!r} is not loaded. Available: {', '.join(available)}"
            )
        return True, f"Connected to {self.model} at {self.base_url}"

    def complete(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
        stop: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send a chat completion and validate the result.

        Args:
            messages: OpenAI-style message list.
            temperature: Sampling temperature. Defaults to the codegen value,
                because the majority of calls in this system generate code.
            max_tokens: Completion cap.
            json_schema: If given, request structured output conforming to this
                JSON schema. LM Studio enforces it during decoding, which stops
                the model wrapping code in prose.
            stop: Optional stop strings.

        Returns:
            LLMResponse with text and telemetry.

        Raises:
            LLMError: on connection failure, timeout, empty output, or an
                otherwise unusable response.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": LLM.codegen_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or LLM.max_tokens,
        }
        if stop:
            kwargs["stop"] = stop
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": json_schema},
            }

        start = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(**kwargs)
        except openai.APITimeoutError as exc:
            elapsed = time.perf_counter() - start
            raise LLMError(
                f"Model did not respond within {self.timeout:.0f}s.",
                kind="timeout",
                elapsed=elapsed,
            ) from exc
        except openai.APIConnectionError as exc:
            elapsed = time.perf_counter() - start
            raise LLMError(
                f"Lost connection to LM Studio at {self.base_url}.",
                kind="connection",
                elapsed=elapsed,
            ) from exc
        except openai.BadRequestError as exc:
            # Most commonly: this build of LM Studio rejected the json_schema.
            # Retry once without structured output rather than failing the query.
            if json_schema:
                kwargs.pop("response_format", None)
                try:
                    completion = self._client.chat.completions.create(**kwargs)
                except Exception as inner:  # noqa: BLE001
                    raise LLMError(
                        f"Request rejected by LM Studio: {inner}", kind="error"
                    ) from inner
            else:
                raise LLMError(f"Request rejected by LM Studio: {exc}", kind="error") from exc

        elapsed = time.perf_counter() - start

        if not completion.choices:
            raise LLMError("Model returned no choices.", kind="empty", elapsed=elapsed)

        choice = completion.choices[0]
        text = (choice.message.content or "").strip()
        finish_reason = choice.finish_reason or "stop"

        parsed_calls: list[ToolCall] = []
        for call in getattr(choice.message, "tool_calls", None) or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                # A 4B model will occasionally emit malformed argument JSON.
                # Surface it as an empty-argument call so the agent loop can feed
                # the resulting error back to the model rather than crashing.
                arguments = {}
            parsed_calls.append(
                ToolCall(id=call.id, name=call.function.name, arguments=arguments)
            )

        # A tool call is a valid response with no prose in it, so the empty-text
        # check must not fire when the model chose to call a tool instead.
        if not text and not parsed_calls:
            raise LLMError(
                "Model returned an empty response. Try rephrasing the question.",
                kind="empty",
                elapsed=elapsed,
            )

        usage = completion.usage
        return LLMResponse(
            text=text,
            elapsed_seconds=elapsed,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            finish_reason=finish_reason,
            # finish_reason == "length" means max_tokens cut the model off
            # mid-sentence. Callers must treat the text as incomplete.
            truncated=finish_reason == "length",
            tool_calls=parsed_calls,
        )
