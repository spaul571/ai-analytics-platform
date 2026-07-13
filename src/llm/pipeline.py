"""The 3-phase natural-language query pipeline (Task B2).

    Phase 1  Code Generation  -  schema + history + question  ->  pandas code
    Phase 2  Execution        -  code  ->  sandboxed run  ->  result
    Phase 3  Formatting       -  question + result  ->  Markdown answer

Phase 2 failures trigger exactly one automatic retry: the exception is fed back
to the model with the failed code. Two retries were tried during development and
did not help - if Gemma cannot fix it on the first correction it does not fix it
on the second either, and the user waits twice as long for the same failure.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import pandas as pd

from src.config import LLM
from src.data.schema import DatasetSchema
from src.llm.client import LLMClient, LLMError
from src.llm.memory import ConversationMemory, Turn
from src.llm.prompts import (
    CODE_RESPONSE_SCHEMA,
    build_codegen_system_prompt,
    build_formatter_messages,
    build_retry_message,
)
from src.llm.sandbox import (
    ExecutionResult,
    SandboxTimeout,
    SandboxViolation,
    execute,
)

# Rows of the result shown to the formatter. Enough for it to spot the notable
# rows; small enough that a wide result cannot crowd out the instructions.
RESULT_PREVIEW_ROWS = 20

_FENCE = re.compile(r"^```(?:python|py)?\s*|\s*```$", re.MULTILINE)


@dataclass
class PipelineResult:
    """Everything one NL question produced, for the UI and the benchmark."""

    question: str
    code: str
    answer: str
    data: pd.DataFrame
    success: bool = True

    # Telemetry (Task B5).
    codegen_seconds: float = 0.0
    execution_ms: float = 0.0
    format_seconds: float = 0.0
    total_seconds: float = 0.0
    retried: bool = False
    truncated: bool = False
    error: str | None = None
    error_kind: str | None = None

    # The reasoning trail, surfaced in the UI's debug expander.
    trace: list[str] = field(default_factory=list)


def _strip_code(raw: str) -> str:
    """Recover a bare snippet from whatever the model actually returned.

    Two layers of wrapping have to be peeled, and both occur in practice:

    1. Structured output wraps the snippet in {"code": "..."}.
    2. The model then wraps the snippet in markdown fences *inside* that string,
       because fencing code is overwhelmingly what its training data does. The
       json_schema constrains the JSON shape, not the content of the string, so
       nothing stops it.

    Missing case 2 was the cause of a "SyntaxError: invalid syntax" on otherwise
    perfectly good code, so the fence strip is applied unconditionally at the
    end rather than only on the non-JSON path.
    """
    text = raw.strip()

    # Layer 1: structured output - a JSON object with a `code` key.
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "code" in parsed:
                text = str(parsed["code"]).strip()
        except json.JSONDecodeError:
            pass

    # Layer 2: markdown fences, wherever they came from.
    return _FENCE.sub("", text).strip()


class NLQueryPipeline:
    """Turns a natural-language question into an executed, narrated answer."""

    def __init__(
        self,
        df: pd.DataFrame,
        schema: DatasetSchema,
        client: LLMClient | None = None,
        memory: ConversationMemory | None = None,
    ):
        self.df = df
        self.schema = schema
        self.client = client or LLMClient()
        self.memory = memory or ConversationMemory()
        self.system_prompt = build_codegen_system_prompt(schema)

    # ---------------------------------------------------------------- Phase 1
    def generate_code(self, question: str, use_history: bool = True) -> tuple[str, float]:
        """Ask the model for pandas code. Returns (code, elapsed_seconds)."""
        messages = [{"role": "system", "content": self.system_prompt}]
        if use_history:
            messages.extend(self.memory.as_messages())
        messages.append({"role": "user", "content": question})

        response = self.client.complete(
            messages,
            temperature=LLM.codegen_temperature,
            json_schema=CODE_RESPONSE_SCHEMA,
        )
        return _strip_code(response.text), response.elapsed_seconds

    def _repair_code(
        self, question: str, bad_code: str, error: str
    ) -> tuple[str, float]:
        """The single auto-retry: hand the model its own error and ask for a fix."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": bad_code},
            {"role": "user", "content": build_retry_message(bad_code, error)},
        ]
        response = self.client.complete(
            messages,
            temperature=LLM.codegen_temperature,
            json_schema=CODE_RESPONSE_SCHEMA,
        )
        return _strip_code(response.text), response.elapsed_seconds

    # ---------------------------------------------------------------- Phase 3
    def format_answer(
        self, question: str, execution: ExecutionResult
    ) -> tuple[str, float, bool]:
        """Narrate the result. Returns (markdown, elapsed_seconds, truncated)."""
        frame = execution.as_frame()
        preview = frame.head(RESULT_PREVIEW_ROWS).to_string(index=False)
        if len(frame) > RESULT_PREVIEW_ROWS:
            preview += f"\n... ({len(frame) - RESULT_PREVIEW_ROWS} more rows)"

        response = self.client.complete(
            build_formatter_messages(question, preview, execution.code),
            temperature=LLM.narrative_temperature,
        )
        return response.text, response.elapsed_seconds, response.truncated

    # ------------------------------------------------------------------- run
    def ask(self, question: str, use_history: bool = True) -> PipelineResult:
        """Run all three phases for one question.

        Never raises. Every failure mode is captured into a PipelineResult with
        success=False, so the UI always has something to render and the
        benchmark always has a row to score.
        """
        trace: list[str] = []
        total = 0.0

        # ---- Phase 1: code generation
        try:
            code, codegen_s = self.generate_code(question, use_history=use_history)
            total += codegen_s
            trace.append(f"Phase 1 - generated code in {codegen_s:.1f}s:\n{code}")
        except LLMError as exc:
            return PipelineResult(
                question=question,
                code="",
                answer=f"**Could not reach the model.** {exc}",
                data=pd.DataFrame(),
                success=False,
                error=str(exc),
                error_kind=exc.kind,
                total_seconds=exc.elapsed,
                trace=[f"Phase 1 failed: {exc}"],
            )

        # ---- Phase 2: sandboxed execution, with one auto-retry
        retried = False
        execution: ExecutionResult | None = None
        last_error = ""

        for attempt in (1, 2):
            try:
                execution = execute(code, self.df)
                trace.append(
                    f"Phase 2 - executed in {execution.execution_ms:.1f}ms "
                    f"(attempt {attempt})"
                )
                break
            except (SandboxViolation, SandboxTimeout, Exception) as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                trace.append(f"Phase 2 - attempt {attempt} failed: {last_error}")

                if attempt == 2:
                    break  # one retry only

                try:
                    code, repair_s = self._repair_code(question, code, last_error)
                    total += repair_s
                    retried = True
                    trace.append(f"Auto-retry - repaired code in {repair_s:.1f}s:\n{code}")
                except LLMError as retry_exc:
                    last_error = str(retry_exc)
                    trace.append(f"Auto-retry failed: {retry_exc}")
                    break

        if execution is None:
            return PipelineResult(
                question=question,
                code=code,
                answer=(
                    "**I could not answer that.** The generated code failed even "
                    f"after one automatic correction.\n\n```\n{last_error}\n```"
                ),
                data=pd.DataFrame(),
                success=False,
                retried=retried,
                error=last_error,
                error_kind="execution",
                codegen_seconds=total,
                total_seconds=total,
                trace=trace,
            )

        # ---- Phase 3: narrative formatting
        try:
            answer, format_s, truncated = self.format_answer(question, execution)
            total += format_s
            trace.append(f"Phase 3 - formatted answer in {format_s:.1f}s")
            if truncated:
                answer += "\n\n*(Response was cut off at the token limit.)*"
        except LLMError as exc:
            # The computation succeeded, so degrade to the raw table rather than
            # discarding a correct answer over a narration failure.
            answer = (
                "*(The narrative could not be generated, but the query ran. "
                "Result below.)*"
            )
            format_s, truncated = 0.0, False
            trace.append(f"Phase 3 failed: {exc}")

        frame = execution.as_frame()

        self.memory.add(
            Turn(
                question=question,
                code=execution.code,
                answer=answer,
                row_count=len(frame),
                elapsed_seconds=total,
                retried=retried,
            )
        )

        return PipelineResult(
            question=question,
            code=execution.code,
            answer=answer,
            data=frame,
            success=True,
            codegen_seconds=codegen_s,
            execution_ms=execution.execution_ms,
            format_seconds=format_s,
            total_seconds=total,
            retried=retried,
            truncated=truncated,
            trace=trace,
        )
