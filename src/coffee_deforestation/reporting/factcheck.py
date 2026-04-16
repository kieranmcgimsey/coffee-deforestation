"""Factuality cross-check: every number in a report must appear in source JSON.

What: Parses every numeric value (integers, floats, percentages) from a generated
markdown report and verifies each appears in the source stats summary JSON,
within a 1% rounding tolerance.
Why: LLM agents can hallucinate numbers. This deterministic check catches
any fabricated statistics before the report is published.
Assumes: The report is markdown text. The source is an AOISummary model or dict.
Produces: A FactcheckResult with matched/unmatched numbers and an optional
appended section if issues are found.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger

from coffee_deforestation.stats.schema import AOISummary


# Regex: match decimal numbers (e.g. 3603.0, 55.8, 1985, 0.75)
# Excludes dates (YYYY), coordinates expressed as pure integers
_NUMBER_RE = re.compile(
    r"""
    (?<!\d)                  # not preceded by a digit
    (?<!\.)                  # not preceded by a decimal point
    (
        \d{1,3}(?:,\d{3})*   # integers with optional thousands separator
        (?:\.\d+)?            # optional decimal part
        |
        \d+\.\d+              # decimals without thousands separator
    )
    (?!\d)                   # not followed by a digit (avoids slicing years)
    """,
    re.VERBOSE,
)

# Numbers that are always acceptable regardless of source (coordinates, years, thresholds)
_ALWAYS_OK_PATTERNS = re.compile(
    r"""
    ^(
        20[0-9]{2}           # years 2000–2099
        | [0-9]{1,3}\.[0-9]{1,4}  # lat/lon style coordinates
        | 0\.[0-9]+           # probabilities/fractions near 0
        | 1\.0                # probability 1.0
        | 10                  # round number
        | 100                 # percentage base
        | 0\.5                # typical threshold
        | 3[02]\s?            # 30 / 30m scale references
    )$
    """,
    re.VERBOSE,
)


@dataclass
class FactcheckResult:
    """Result of a factuality cross-check."""

    total_numbers: int = 0
    matched: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.unmatched) == 0

    @property
    def summary(self) -> str:
        return (
            f"{self.total_numbers} numbers checked: "
            f"{len(self.matched)} matched, "
            f"{len(self.unmatched)} unmatched, "
            f"{len(self.skipped)} skipped"
        )


def _extract_source_numbers(summary: AOISummary) -> set[float]:
    """Flatten AOISummary into a set of all numeric values."""
    numbers: set[float] = set()

    def _walk(obj: object) -> None:
        if isinstance(obj, (int, float)):
            numbers.add(float(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)
        elif hasattr(obj, "model_dump"):
            _walk(obj.model_dump())

    _walk(summary)

    # Add derived values: percentages (fraction × 100)
    derived = set()
    for n in numbers:
        derived.add(round(n * 100, 2))
        derived.add(round(n / 100, 6))
        # Rounded variants
        for decimals in range(4):
            derived.add(round(n, decimals))
    numbers |= derived

    return numbers


def _parse_report_numbers(report: str) -> list[str]:
    """Extract all numeric strings from a markdown report."""
    # Remove code blocks to avoid false positives from config/coordinates
    clean = re.sub(r"```.*?```", "", report, flags=re.DOTALL)
    # Remove markdown table separators
    clean = re.sub(r"\|[-:| ]+\|", "", clean)

    raw = _NUMBER_RE.findall(clean)
    # Normalise: remove commas
    return [r.replace(",", "") for r in raw]


def _number_in_source(
    num_str: str,
    source_numbers: set[float],
    tolerance: float = 0.01,
) -> bool:
    """Return True if num_str is within tolerance of any source number."""
    try:
        val = float(num_str)
    except ValueError:
        return False

    if val == 0.0:
        return True  # zero always ok

    for src in source_numbers:
        if src == 0.0:
            continue
        rel_diff = abs(val - src) / max(abs(src), 1e-9)
        if rel_diff <= tolerance:
            return True
        # Also check absolute diff for small numbers
        if abs(val - src) < 0.1:
            return True

    return False


def factcheck(
    report: str,
    summary: AOISummary,
    tolerance: float = 0.01,
) -> FactcheckResult:
    """Run the factuality cross-check on a report against its source summary.

    Returns a FactcheckResult. Numbers that cannot be traced to the source
    are flagged as unmatched (potential hallucinations).
    """
    result = FactcheckResult()
    source_numbers = _extract_source_numbers(summary)
    report_numbers = _parse_report_numbers(report)

    result.total_numbers = len(report_numbers)

    for num_str in report_numbers:
        # Skip always-ok patterns (years, standard thresholds, etc.)
        if _ALWAYS_OK_PATTERNS.match(num_str.strip()):
            result.skipped.append(num_str)
            continue

        if _number_in_source(num_str, source_numbers, tolerance):
            result.matched.append(num_str)
        else:
            result.unmatched.append(num_str)

    if result.unmatched:
        logger.warning(
            f"Factcheck found {len(result.unmatched)} unmatched numbers: "
            f"{result.unmatched[:5]}{'...' if len(result.unmatched) > 5 else ''}"
        )
    else:
        logger.info(f"Factcheck passed: {result.summary}")

    return result


def append_factcheck_section(report: str, fc_result: FactcheckResult) -> str:
    """Append a factcheck section to a report if issues were found.

    Returns the report unchanged if factcheck passed.
    """
    if fc_result.passed:
        return report

    section = f"""

---

## Factcheck Report

*Auto-generated by the factuality cross-check system.*

**Status:** ⚠️ {len(fc_result.unmatched)} number(s) could not be traced to the source statistics JSON.

**Summary:** {fc_result.summary}

**Unmatched values** (may be hallucinated or derived estimates — review manually):
{chr(10).join(f'- `{n}`' for n in fc_result.unmatched)}

*Matched {len(fc_result.matched)} values, skipped {len(fc_result.skipped)} (years, coordinates, thresholds).*
"""
    return report + section
