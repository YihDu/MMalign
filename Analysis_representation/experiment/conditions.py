"""Simple builders for experiment conditions."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List

from data.schemas import MultilingualExample

ConditionBuilder = Callable[[Iterable[MultilingualExample]], List[MultilingualExample]]


def _identity(examples: Iterable[MultilingualExample]) -> List[MultilingualExample]:
    return list(examples)


def _mismatched(examples: Iterable[MultilingualExample]) -> List[MultilingualExample]:
    examples_list = list(examples)
    if len(examples_list) < 2:
        return examples_list
    rotated = examples_list[1:] + examples_list[:1]
    return [
        MultilingualExample(image=orig.image, captions=mis.captions)
        for orig, mis in zip(examples_list, rotated, strict=False)
    ]


def build_conditions() -> Dict[str, ConditionBuilder]:
    """Return the baseline/correct/mismatched condition builders."""

    return {
        "baseline": _identity,
        "correct": _identity,
        "mismatched": _mismatched,
    }
