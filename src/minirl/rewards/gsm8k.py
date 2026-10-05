"""Independent correctness, think-tag, and final-answer rewards for GSM8K."""

from __future__ import annotations

import logging
import re
from functools import lru_cache

_THINK = re.compile(r"\A\s*<think>(?P<reasoning>[\s\S]+?)</think>(?P<final>[\s\S]*)\Z")
_ANSWER = re.compile(r"^####[ \t]+([^\n]+)$", re.MULTILINE)
_END = re.compile(r"(?:<\|im_end\|>|<\|endoftext\|>)\s*$")
logger = logging.getLogger(__name__)


@lru_cache(maxsize=16384)
def parse_number(text: str):
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse

    return parse(
        text,
        extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
        fallback_mode="no_fallback",
        parsing_timeout=2,
        raise_on_error=True,
    )


def score_gsm8k(text: str, answer: str) -> dict[str, float | int | str | None]:
    """Score the model's complete generated answer; no think tag is prefixed.

    Correctness is independent of the two formatting bonuses. The final section
    may use a boxed answer instead of #### and still earn its correctness reward.
    A single #### line takes precedence over other numbers in the final section.
    """
    from math_verify import verify

    text = _END.sub("", text).strip()
    match = _THINK.fullmatch(text)
    think = bool(
        match
        and match["reasoning"].strip()
        and text.count("<think>") == 1
        and text.count("</think>") == 1
    )
    final = text.rsplit("</think>", 1)[-1].strip()
    markers = list(_ANSWER.finditer(final))
    answer_format = bool(
        len(markers) == 1
        and markers[0].end() == len(final)
        and markers[0][1].strip()
        and final.count("####") == 1
    )
    candidate = markers[0][1].strip() if len(markers) == 1 else final
    gold = parse_number(r"\boxed{" + answer.replace(",", "") + "}")
    if not gold:
        raise ValueError(f"Unparseable GSM8K reference: {answer!r}")
    parsed, correct, error = False, False, None
    try:
        predicted = parse_number(candidate) if len(markers) <= 1 else []
        parsed = bool(predicted)
        correct = bool(
            predicted
            and verify(gold, predicted, timeout_seconds=2, raise_on_error=True)
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("GSM8K verifier failed")
    return {
        "correct": int(correct),
        "think_format": int(think),
        "answer_format": int(answer_format),
        "answer_parsed": int(parsed),
        "correctness_reward": float(correct),
        "think_reward": 0.3 * think,
        "answer_reward": 0.2 * answer_format,
        "reward": float(correct) + 0.3 * think + 0.2 * answer_format,
        "candidate": candidate,
        "grader_error": error,
    }
