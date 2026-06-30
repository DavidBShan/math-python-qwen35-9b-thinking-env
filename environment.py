"""Freesolo wrapper for PrimeIntellect's math-python environment.

The upstream Prime environment is the `math-python` package. It loads the MATH
example dataset, prompts the model to use Python, and grades boxed answers with
math-verify. Flash requires a Freesolo SDK environment, so this file preserves
the same dataset/prompt/scoring contract in `EnvironmentSingleTurn` form.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

from freesolo.datasets.records import load_task_examples
from freesolo.datasets.types import TaskExample
from freesolo.environments import EnvironmentSingleTurn, RewardMetric, RewardResult
from math_verify import parse, verify


ROOT = Path(__file__).parent
DEFAULT_DATASET = "datasets/rl_train.jsonl"

SYSTEM_PROMPT = (
    "Use Python for all calculations. Give your answer inside \\boxed{}.\n\n"
    "In addition to the Python standard library, you have access to: numpy sympy scipy."
)
_VERIFY_LOCK = threading.Lock()


def _boxed_answer(text: str) -> str | None:
    """Extract the final \\boxed{...} payload, handling one level of nested braces."""
    last = text.rfind("\\boxed")
    if last < 0:
        return None
    brace = text.find("{", last)
    if brace < 0:
        return None
    depth = 0
    for idx in range(brace, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : idx].strip()
    return None


def _has_visible_thinking(text: str) -> bool:
    if re.search(r"<think>.*?</think>", text, flags=re.IGNORECASE | re.DOTALL):
        return True
    close = re.search(r"</think>", text, flags=re.IGNORECASE)
    if not close:
        return False
    return bool(text[: close.start()].strip())


def _candidate_answers(predicted: str | None) -> list[str]:
    if not predicted:
        return []
    candidates = [str(predicted).strip()]
    if "=" in candidates[0]:
        rhs = candidates[0].rsplit("=", 1)[-1].strip()
        if rhs:
            candidates.append(rhs)
    return list(dict.fromkeys(candidates))


def _math_correct(predicted: str | None, gold: str) -> tuple[bool, str | None]:
    candidates = _candidate_answers(predicted)
    if not candidates:
        return False, "missing_boxed_answer"
    gold_text = str(gold).strip()
    for candidate in candidates:
        if candidate == gold_text:
            return True, None
    try:
        with _VERIFY_LOCK:
            gold_parsed = parse(str(gold))
            for candidate in candidates:
                if bool(verify(gold_parsed, parse(candidate))):
                    return True, None
        return False, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _gold_answer(example: TaskExample) -> str:
    metadata = example.metadata if isinstance(example.metadata, dict) else {}
    if metadata.get("answer") is not None:
        return str(metadata["answer"])
    if example.output is not None:
        return str(example.output)
    return ""


class MathPythonEnv(EnvironmentSingleTurn):
    max_score_concurrency = 1

    def __init__(
        self,
        *,
        dataset_path: str = DEFAULT_DATASET,
        max_examples: int = 0,
        require_thinking: bool = True,
    ) -> None:
        path = Path(dataset_path)
        if not path.is_absolute():
            path = ROOT / path
        rows = load_task_examples(path)
        if max_examples:
            rows = rows[: int(max_examples)]
        self.dataset = rows
        self.require_thinking = bool(require_thinking)

    def build_prompt_messages(self, example: TaskExample, prompt_text: str):
        _ = prompt_text
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(example.input)},
        ]

    def sft_completion(self, example: TaskExample):
        metadata = example.metadata if isinstance(example.metadata, dict) else {}
        oracle = metadata.get("oracle_solution")
        if not oracle and example.output is not None and "<think" in str(example.output).lower():
            oracle = str(example.output)
        answer = _gold_answer(example)
        if oracle:
            content = str(oracle)
        else:
            content = (
                "<think>\n"
                "I will use Python/symbolic calculation as needed and verify the final value.\n"
                f"The computed answer is {answer}.\n"
                "</think>\n\n"
                f"\\boxed{{{answer}}}"
            )
        return [{"role": "assistant", "content": content}]

    def score_response(self, example: TaskExample, response_text: str) -> RewardResult:
        gold = _gold_answer(example)
        predicted = _boxed_answer(response_text)
        correct, error = _math_correct(predicted, gold)
        has_thinking = _has_visible_thinking(response_text)

        if correct and (has_thinking or not self.require_thinking):
            score = 1.0
        elif correct:
            score = 0.8
        elif has_thinking:
            score = 0.05
        else:
            score = 0.0

        success = bool(correct and (has_thinking or not self.require_thinking))
        metrics = (
            RewardMetric(
                name="math_correct",
                score=1.0 if correct else 0.0,
                success=correct,
                value={"predicted": predicted, "gold": gold},
            ),
            RewardMetric(
                name="visible_thinking",
                score=1.0 if has_thinking else 0.0,
                success=has_thinking,
            ),
        )
        reason = (
            f"correct={correct} visible_thinking={has_thinking} "
            f"predicted={predicted!r} gold={gold!r}"
        )
        return RewardResult(
            score=score,
            threshold=1.0,
            success=success,
            value={
                "correct": correct,
                "visible_thinking": has_thinking,
                "predicted": predicted,
                "gold": gold,
            },
            reason=reason,
            error=error,
            metrics=metrics,
            return_type="numeric",
        )


def load_environment(
    dataset_path: str = DEFAULT_DATASET,
    max_examples: int = 0,
    require_thinking: bool = True,
    **kwargs: Any,
) -> MathPythonEnv:
    _ = kwargs
    return MathPythonEnv(
        dataset_path=dataset_path,
        max_examples=max_examples,
        require_thinking=require_thinking,
    )
