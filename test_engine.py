"""Fast deterministic checks for routing policy and latency."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

from engine import (
    ASPECT_PROMPTS,
    Lane,
    RISK_PROMPTS,
    ReviewInput,
    TriageEngine,
    ZeroShotClassifier,
)


@dataclass(frozen=True)
class RoutingCase:
    text: str
    rating: int
    aspect: str
    risk: str
    expected_lane: Lane
    confidence: float = 0.92
    expected_aspect: str | None = None


class FixedClassifier(ZeroShotClassifier):
    def __init__(self, aspect: str, risk: str, confidence: float = 0.92) -> None:
        self.aspect = aspect
        self.risk = risk
        self.confidence = confidence

    def classify(self, text: str, labels: Sequence[str]) -> dict[str, float]:
        if not text.strip():
            raise ValueError("Classifier input must not be blank.")
        is_aspect = tuple(labels) == tuple(ASPECT_PROMPTS.values())
        prompts = ASPECT_PROMPTS if is_aspect else RISK_PROMPTS
        chosen = prompts[self.aspect] if is_aspect else prompts[self.risk]
        return {label: self.confidence if label == chosen else 0.02 for label in labels}


CASES: tuple[RoutingCase, ...] = (
    RoutingCase(
        "Great coffee and friendly staff.", 5, "general experience",
        "routine praise or neutral feedback", Lane.AUTO_PUBLISH,
    ),
    RoutingCase(
        "My order was delayed and arrived cold.", 2, "service speed",
        "branch-level service or operations complaint", Lane.ADVISOR_QUEUE,
    ),
    RoutingCase(
        "Found a cockroach near the counter.", 1, "cleanliness",
        "safety or food hygiene concern", Lane.MANAGER_HOLD, expected_aspect="food safety",
    ),
    RoutingCase(
        "Found a cockroach near the counter.", 1, "cleanliness",
        "routine praise or neutral feedback", Lane.MANAGER_HOLD, expected_aspect="food safety",
    ),
    RoutingCase(
        "A lawyer will contact you about this.", 1, "general experience",
        "legal, regulatory, or litigation threat", Lane.MANAGER_HOLD,
    ),
    RoutingCase(
        "The app charged me twice.", 2, "digital ordering and billing",
        "fraud, payment dispute, or financial misconduct", Lane.MANAGER_HOLD,
        expected_aspect="digital ordering and billing",
    ),
    RoutingCase(
        "A neutral visit with no clear feedback.", 5, "general experience",
        "routine praise or neutral feedback", Lane.ADVISOR_QUEUE, 0.42,
    ),
)


def main() -> None:
    correct = 0
    latencies_ms: list[float] = []
    for case in CASES:
        engine = TriageEngine(
            FixedClassifier(case.aspect, case.risk, case.confidence)
        )
        review = ReviewInput(text=case.text, branch="PoC sample", rating=case.rating)
        started = perf_counter()
        result = engine.analyze(review)
        latencies_ms.append((perf_counter() - started) * 1000)
        passed = (
            result.lane is case.expected_lane
            and result.publish_allowed is (case.expected_lane is Lane.AUTO_PUBLISH)
            and (
                case.expected_aspect is None
                or result.aspect == case.expected_aspect
            )
            and (
                result.reply_draft is None
                if case.expected_lane is Lane.MANAGER_HOLD
                else result.reply_draft is not None
            )
        )
        correct += int(passed)
        print(
            f"{'PASS' if passed else 'FAIL'} | expected={case.expected_lane.value} "
            f"| actual={result.lane.value} | {latencies_ms[-1]:.2f} ms"
        )

    accuracy = correct / len(CASES)
    average_latency = sum(latencies_ms) / len(latencies_ms)
    print(f"\nRouting accuracy: {correct}/{len(CASES)} ({accuracy:.0%})")
    print(f"Mean policy latency: {average_latency:.2f} ms (model excluded)")
    if correct != len(CASES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
