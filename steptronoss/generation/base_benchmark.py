"""Minimal benchmark data model shared by generation and offline scoring.

The intended ownership split is:

- `Prompt`: request-ready model input owned by the benchmark implementation.
- `BenchmarkMeta`: benchmark-specific scoring context owned by the benchmark
  implementation and never sent to the model.
- `EvaluationMeta`: runtime metadata owned by the evaluator/generation runner.
- `Generated`: final model output plus the metadata required for offline
  analysis.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import TypeAlias, TypedDict

JsonScalar: TypeAlias = None | bool | int | float | str
"""Primitive JSON scalar values used in exported benchmark artifacts."""

JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
"""Recursive JSON value used to describe stored benchmark metadata."""

JsonObject: TypeAlias = dict[str, JsonValue]
"""JSON object with string keys and recursive JSON values."""


class GroundTruthValue(TypedDict):
    """Minimal ground-truth pointer stored on some exported assistant turns.

    Fields:
        item_id: Stable benchmark item identifier referenced by this turn.
        dataset: Dataset or benchmark family that owns the referenced item.
    """

    item_id: str
    dataset: str


class GroundTruth(TypedDict):
    """Ground-truth wrapper used by exported benchmark prompts.

    Fields:
        value: Nested ground-truth pointer payload.
    """

    value: GroundTruthValue


class ChatMessage(TypedDict, total=False):
    """Structured chat message used by tokenizer rendering and generation.

    Fields:
        role: Chat role such as `system`, `user`, or `assistant`.
        content: Plain-text content rendered into the model prompt.
        ground_truth: Optional benchmark-specific pointer attached to the turn.
    """

    role: str
    content: str
    ground_truth: GroundTruth


Messages: TypeAlias = list[ChatMessage]
"""Ordered chat messages sent to the generation endpoint."""


class CompletionMessage(TypedDict, total=False):
    """Structured message payload from chat/completions style APIs.

    Fields:
        content: Plain-text assistant response.
        reasoning_content: Optional reasoning side channel returned by some APIs.
    """

    content: str
    reasoning_content: str


class CompletionChoice(TypedDict, total=False):
    """Normalized completion choice preserved in prediction dumps.

    Fields:
        finish_reason: Backend-provided stop reason for the sampled completion.
        text: Plain text returned by completion-style APIs.
        message: Structured message returned by chat/completions APIs.
        raw: JSON object containing the original backend payload for this choice.
    """

    finish_reason: str
    text: str
    message: CompletionMessage
    raw: JsonObject


@dataclass(frozen=True)
class SamplingParams:
    """Request parameters that control how a prompt is generated."""

    temperature: float | None = None
    """Sampling temperature passed to the generation backend."""

    top_p: float | None = None
    """Nucleus-sampling threshold passed to the generation backend."""

    top_k: int | None = None
    """Top-k sampling threshold passed to the generation backend."""

    max_tokens: int | None = None
    """Maximum number of generated tokens requested from the backend."""

    seed: int | None = None
    """Random seed used to make repeated sampling runs reproducible."""

    extra_body: JsonObject = field(default_factory=dict)
    """Additional request fields not covered by the explicit attributes above."""

    def to_dict(self) -> JsonObject:
        """Serialize request parameters into a generation payload fragment."""

        payload = dict(self.extra_body)
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.seed is not None:
            payload["seed"] = self.seed
        return payload


@dataclass(frozen=True)
class Prompt:
    """Client-side request payload for one benchmark generation."""

    tokens: list[int] | None = None
    """Tokenized prompt content for token-based generation APIs."""

    messages: Messages | None = None
    """Structured chat messages for chat-style generation APIs."""

    prompt_token_count: int | None = None
    """Optional prompt length used for budget checks when only messages are sent."""

    sampling_params: SamplingParams | None = None
    """Generation parameters attached to this prompt from the client view."""

    def __post_init__(self) -> None:
        """Validate the request shape and derive prompt length when possible."""

        has_tokens = self.tokens is not None
        has_messages = self.messages is not None
        if has_tokens == has_messages:
            raise ValueError("Prompt must define exactly one of tokens or messages")
        if self.prompt_token_count is not None and self.prompt_token_count < 0:
            raise ValueError("Prompt.prompt_token_count must be non-negative")
        if self.tokens is not None and self.prompt_token_count is None:
            object.__setattr__(self, "prompt_token_count", len(self.tokens))

    def with_sampling_params(self, sampling_params: SamplingParams) -> Prompt:
        """Return a copy of this prompt with generation parameters attached."""

        return replace(self, sampling_params=sampling_params)

    def to_dict(self) -> JsonObject:
        """Serialize the prompt into a JSON-compatible structure."""

        payload: JsonObject = {}
        if self.tokens is not None:
            payload["tokens"] = list(self.tokens)
        if self.messages is not None:
            payload["messages"] = [dict(message) for message in self.messages]
        if self.prompt_token_count is not None:
            payload["prompt_token_count"] = self.prompt_token_count
        payload["sampling_params"] = None if self.sampling_params is None else self.sampling_params.to_dict()
        return payload


@dataclass(frozen=True)
class BenchmarkMeta:
    """Benchmark-owned metadata that is independent of evaluator runtime state."""

    benchmark_name: str
    """Stable benchmark identifier, usually matching the dataset family name."""

    item_id: str
    """Stable item identifier chosen by the benchmark implementation."""

    context: JsonObject = field(default_factory=dict)
    """Benchmark-specific scoring context not meant to be fed to the model."""

    def to_dict(self) -> JsonObject:
        """Serialize benchmark-owned metadata into a JSON-compatible structure."""

        return {
            "benchmark_name": self.benchmark_name,
            "item_id": self.item_id,
            "context": dict(self.context),
        }


@dataclass(frozen=True)
class EvaluationMeta:
    """Evaluator-owned runtime metadata that is benchmark-agnostic."""

    prompt_index: int
    """Index of the expanded prompt within the current evaluation run."""

    run_index: int
    """Sample index among repeated generations for the same logical item."""

    def to_dict(self) -> JsonObject:
        """Serialize evaluator-owned metadata into a JSON-compatible structure."""

        return {
            "prompt_index": self.prompt_index,
            "run_index": self.run_index,
        }


@dataclass(frozen=True)
class EvaluationCase:
    """Minimal Prompt -> Generated data-flow unit used by benchmark runners."""

    prompt: Prompt
    """Prompt to send to the generation backend."""

    benchmark: BenchmarkMeta
    """Benchmark-owned context associated with the prompt."""

    evaluation: EvaluationMeta
    """Evaluator-owned runtime metadata associated with the prompt."""

    def with_sampling_params(self, sampling_params: SamplingParams) -> EvaluationCase:
        """Return a copy of the case with request parameters attached to its prompt."""

        return EvaluationCase(
            prompt=self.prompt.with_sampling_params(sampling_params),
            benchmark=self.benchmark,
            evaluation=self.evaluation,
        )


def build_demo_generated(response: str = "3") -> Generated:
    """Build the smallest complete Prompt -> Generated example.

    This helper is intentionally simple so new benchmark implementations can see
    the expected data flow in one place without reading the full runner stack.
    """

    case = EvaluationCase(
        prompt=Prompt(
            messages=[{"role": "user", "content": "How many r are in strawberry?"}],
            prompt_token_count=3,
            sampling_params=SamplingParams(seed=0, max_tokens=8),
        ),
        benchmark=BenchmarkMeta(
            benchmark_name="demo_count_r",
            item_id="strawberry_r_count_0",
            context={"answer": "3"},
        ),
        evaluation=EvaluationMeta(prompt_index=0, run_index=0),
    )
    return Generated(case=case, response=response)


@dataclass
class Generated:
    """Model output plus benchmark and evaluator metadata for offline scoring."""

    case: EvaluationCase
    """Prompt plus benchmark/evaluator metadata that produced this output."""

    choice: CompletionChoice | None = None
    """Normalized completion choice returned by the serving backend, if any."""

    response: str = ""
    """Plain-text model response extracted from the backend payload."""

    reasoning_content: str | None = None
    """Optional side-channel reasoning content returned by compatible models."""

    error: str | None = None
    """Error text captured when generation failed instead of returning a choice."""

    @property
    def finish_reason(self) -> str:
        """Expose a normalized finish reason for downstream metrics."""

        if self.error:
            return "error"
        if not self.choice:
            return "unknown"
        finish_reason = self.choice.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            return finish_reason
        return "unknown"

    def to_dict(self) -> JsonObject:
        """Serialize the generated output into a JSON-compatible structure."""

        return {
            "prompt": self.case.prompt.to_dict(),
            "benchmark": self.case.benchmark.to_dict(),
            "evaluation": self.case.evaluation.to_dict(),
            "choice": None if self.choice is None else dict(self.choice),
            "response": self.response,
            "reasoning_content": self.reasoning_content,
            "error": self.error,
        }


@dataclass
class BaseMetric:
    """Common aggregate metric shape shared by lightweight benchmark scorers."""

    score_avg: float
    """Mean sample score across all generated outputs."""

    score_std: float
    """Population standard deviation of sample scores."""

    pass_at_k: dict[int, float] = field(default_factory=dict)
    """Unbiased HumanEval-style pass@k estimates averaged over grouped items."""

    def __repr__(self) -> str:
        """Render a stable human-readable summary string."""

        def _format(value: float) -> str:
            if math.isnan(value):
                return "nan"
            return f"{value:.4f}"

        pass_repr = ", ".join(f"{k}: {_format(v)}" for k, v in sorted(self.pass_at_k.items()))
        return (
            f"BaseMetric(score_avg={_format(self.score_avg)}, "
            f"score_std={_format(self.score_std)}, "
            f"pass_at_k={{ {pass_repr} }})"
        )

    def to_dict(self) -> JsonObject:
        """Serialize aggregate metric fields into a JSON-compatible structure."""

        return {
            "score_avg": self.score_avg,
            "score_std": self.score_std,
            "pass_at_k": {str(k): v for k, v in self.pass_at_k.items()},
            "repr": repr(self),
        }


class BaseBenchmark(ABC):
    """Minimal protocol shared by exported benchmark adapters."""

    name: str
    """Stable benchmark identifier used in prediction dumps and summaries."""

    @abstractmethod
    def get_cases(self) -> list[EvaluationCase]:
        """Load evaluation cases in the exact order expected by the scorer."""

    @abstractmethod
    def evaluate(self, results: list[Generated]) -> BaseMetric:
        """Aggregate per-sample outputs into a benchmark metric."""
