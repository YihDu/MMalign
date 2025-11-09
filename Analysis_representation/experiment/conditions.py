"""Simple builders for experiment conditions."""

from __future__ import annotations

import random
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

from data.schemas import MultilingualExample

ConditionBuilder = Callable[[Iterable[MultilingualExample]], List[MultilingualExample]]


def build_conditions(
    config: Mapping[str, object] | None = None,
) -> Dict[str, ConditionBuilder]:
    """Return the baseline/correct/mismatched condition builders."""

    mismatched_cfg = {}
    if isinstance(config, Mapping):
        mismatched_cfg = config.get("mismatched", {}) or {}
    hard_negative = bool(mismatched_cfg.get("hard_negative", False))

    return {
        "correct": _identity,
        "mismatched": _mismatched_builder(hard_negative=hard_negative),
    }


# 原始样本
def _identity(examples: Iterable[MultilingualExample]) -> List[MultilingualExample]:
    return list(examples)


def _mismatched_builder(*, hard_negative: bool) -> ConditionBuilder:
    def _builder(examples: Iterable[MultilingualExample]) -> List[MultilingualExample]:
        return _mismatched(examples, hard_negative=hard_negative)

    return _builder


def _mismatched(
    examples: Iterable[MultilingualExample],
    *,
    hard_negative: bool,
) -> List[MultilingualExample]:
    examples_list = list(examples)
    count = len(examples_list)
    if count < 2:
        return examples_list

    categories = [_super_category(example.image.metadata) for example in examples_list]
    perm = _derangement(count)
    if hard_negative:
        if _has_category_diversity(categories):
            perm = _enforce_hard_negative(perm, categories)
        else:
            print("[warn][condition:mismatched] hard_negative enabled but super-category metadata is missing or not diverse; falling back to plain derangement.")
    mismatched_list = [
        MultilingualExample(image=examples_list[idx].image, captions=examples_list[target].captions)
        for idx, target in enumerate(perm)
    ]
    accidental = sum(1 for idx, target in enumerate(perm) if idx == target)
    print(f"[debug][condition:mismatched] derangement_size={count} accidental_matches={accidental}")
    if hard_negative:
        remaining = _count_supercategory_matches(perm, categories)
        if remaining:
            print(
                "[warn][condition:mismatched] Unable to satisfy hard_negative for "
                f"{remaining} pairs; continuing with best-effort derangement."
            )
        else:
            print("[debug][condition:mismatched] hard_negative satisfied across all pairs (accidental_matches=0).")
    return mismatched_list


def _derangement(count: int, attempts: int = 128) -> List[int]:
    if count < 2:
        return list(range(count))
    indices = list(range(count))
    for _ in range(attempts):
        random.shuffle(indices)
        if _is_derangement(indices):
            return list(indices)
    fallback = list(range(count))
    return fallback[1:] + fallback[:1]


def _is_derangement(perm: Sequence[int]) -> bool:
    return all(idx != value for idx, value in enumerate(perm))


def _enforce_hard_negative(
    perm: List[int],
    categories: Sequence[str | None],
    max_swaps: int = 2048,
) -> List[int]:
    adjusted = perm.copy()
    if not categories:
        return adjusted
    for _ in range(max_swaps):
        violations = [idx for idx, target in enumerate(adjusted) if _same_super(categories[idx], categories[target])]
        if not violations:
            return adjusted
        pivot = random.choice(violations)
        swapped = False
        candidate_indices = list(range(len(adjusted)))
        random.shuffle(candidate_indices)
        for other in candidate_indices:
            if other == pivot:
                continue
            target_pivot = adjusted[pivot]
            target_other = adjusted[other]
            if target_other == pivot or target_pivot == other:
                continue
            if _same_super(categories[pivot], categories[target_other]):
                continue
            if _same_super(categories[other], categories[target_pivot]):
                continue
            adjusted[pivot], adjusted[other] = adjusted[other], adjusted[pivot]
            swapped = True
            break
        if not swapped:
            break
    return adjusted


def _count_supercategory_matches(
    perm: Sequence[int],
    categories: Sequence[str | None],
) -> int:
    return sum(
        1
        for idx, target in enumerate(perm)
        if _same_super(categories[idx], categories[target])
    )


def _has_category_diversity(categories: Sequence[str | None]) -> bool:
    observed = {value for value in categories if value}
    return len(observed) > 1


def _same_super(cat_a: str | None, cat_b: str | None) -> bool:
    if not cat_a or not cat_b:
        return False
    return cat_a == cat_b


def _super_category(metadata) -> str | None:
    if not metadata:
        return None
    for key in ("supercategory", "super_category", "superCategory"):
        value = metadata.get(key) if isinstance(metadata, Mapping) else None
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None
