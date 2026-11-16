"""Zero-shot review attribution and risk-aware routing for the Revio PoC."""

from __future__ import annotations

from enum import Enum
from threading import Lock
from typing import Protocol, Sequence, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class Lane(str, Enum):
    AUTO_PUBLISH = "Lane A: Auto-Publish"
    ADVISOR_QUEUE = "Lane B: Advisor Draft Queue"
    MANAGER_HOLD = "Lane C: Manager Crisis Hold"


class RiskBand(str, Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"


AUTO_PUBLISH_CONFIDENCE_THRESHOLD = 0.55

ASPECT_PROMPTS: dict[str, str] = {
    "food quality": "food quality, including taste, freshness, temperature, and portion size",
    "service speed": "service speed, including waiting, order timing, queues, and delays",
    "staff conduct": "staff conduct, including friendliness and professionalism",
    "cleanliness": "cleanliness of dining areas, restrooms, and tables",
    "food safety": "food safety, including illness, pests, contaminants, and allergens",
    "pricing and value": "pricing and value, including prices, portions, and promotions",
    "ambience": "ambience, including noise, seating, and atmosphere",
    "digital ordering and billing": "digital ordering and billing, including app, payment, and pickup",
    "general experience": "general experience without one clear specific aspect",
}

RISK_PROMPTS: dict[str, str] = {
    "routine praise or neutral feedback": (
        "routine positive or neutral feedback about a restaurant visit"
    ),
    "branch-level service or operations complaint": (
        "a branch-level service or operations complaint, such as a late, missing, "
        "or incorrect order, or a long wait"
    ),
    "safety or food hygiene concern": (
        "a food safety or hygiene incident, such as pests, contamination, allergens, or illness"
    ),
    "fraud, payment dispute, or financial misconduct": (
        "a fraud or payment complaint, such as an unauthorized or duplicate charge"
    ),
    "legal, regulatory, or litigation threat": (
        "a legal or regulatory threat, such as a lawsuit or report to authorities"
    ),
}

CRISIS_RISKS = {
    "safety or food hygiene concern",
    "fraud, payment dispute, or financial misconduct",
    "legal, regulatory, or litigation threat",
}

_CRISIS_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "safety or food hygiene concern",
        (
            "food poisoning",
            "allergic reaction",
            "allergic attack",
            "raw chicken",
            "undercooked chicken",
            "glass in",
            "metal in",
            "mold in",
            "mould in",
            "cockroach",
            "cockroaches",
            "rat in",
            "rats in",
            "contaminated",
            "health department",
            "hospital after eating",
        ),
    ),
    (
        "fraud, payment dispute, or financial misconduct",
        (
            "fraud",
            "stole my card",
            "card was stolen",
            "unauthorized charge",
            "unauthorised charge",
            "charged me twice",
            "fake transaction",
        ),
    ),
    (
        "legal, regulatory, or litigation threat",
        (
            "lawsuit",
            "lawyer",
            "legal action",
            "sue you",
            "sue the",
            "report you to",
            "regulator",
            "regulatory complaint",
        ),
    ),
)


class ReviewInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(min_length=3, max_length=5000)
    branch: str = Field(default="Unspecified branch", min_length=1, max_length=120)
    rating: int = Field(default=3, ge=1, le=5)
    reviewer: str | None = Field(default=None, max_length=120)


class TriageResult(BaseModel):
    review: ReviewInput
    aspect: str
    aspect_confidence: float = Field(ge=0, le=1)
    risk_category: str
    risk_band: RiskBand
    risk_confidence: float = Field(ge=0, le=1)
    lane: Lane
    route_owner: str
    publish_allowed: bool
    rationale: str
    reply_draft: str | None = None
    aspect_scores: dict[str, float]
    risk_scores: dict[str, float]


class ZeroShotClassifier(Protocol):
    def classify(self, text: str, labels: Sequence[str]) -> dict[str, float]:
        """Return a confidence score for each candidate label."""


class ZeroShotOutput(TypedDict):
    labels: list[str]
    scores: list[float]


class ZeroShotPipeline(Protocol):
    def __call__(
        self,
        text: str,
        *,
        candidate_labels: list[str],
        multi_label: bool,
    ) -> ZeroShotOutput: ...


class HuggingFaceNLIClassifier:
    """Lazy Hugging Face zero-shot classifier backed by an NLI model."""

    def __init__(self, model_name: str = "valhalla/distilbart-mnli-12-1") -> None:
        self.model_name = model_name
        self._pipeline: ZeroShotPipeline | None = None
        self._lock = Lock()

    def _load_pipeline(self) -> ZeroShotPipeline:
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    try:
                        from transformers import pipeline
                    except ImportError as exc:
                        raise RuntimeError(
                            "Hugging Face dependencies are unavailable. "
                            "Install the project's requirements.txt first."
                        ) from exc
                    self._pipeline = pipeline(
                        "zero-shot-classification",
                        model=self.model_name,
                        device=-1,
                    )
        return self._pipeline

    def classify(self, text: str, labels: Sequence[str]) -> dict[str, float]:
        output = self._load_pipeline()(
            text,
            candidate_labels=list(labels),
            multi_label=False,
        )
        return {
            label: float(score)
            for label, score in zip(output["labels"], output["scores"], strict=True)
        }


class TriageEngine:
    def __init__(self, classifier: ZeroShotClassifier | None = None) -> None:
        self.classifier = (
            classifier if classifier is not None else HuggingFaceNLIClassifier()
        )

    def analyze(self, review: ReviewInput) -> TriageResult:
        raw_aspect_scores = self.classifier.classify(
            review.text, tuple(ASPECT_PROMPTS.values())
        )
        raw_risk_scores = self.classifier.classify(review.text, tuple(RISK_PROMPTS.values()))
        aspect_scores = {
            label: raw_aspect_scores[prompt] for label, prompt in ASPECT_PROMPTS.items()
        }
        risk_scores = {
            label: raw_risk_scores[prompt] for label, prompt in RISK_PROMPTS.items()
        }

        aspect = max(aspect_scores, key=aspect_scores.__getitem__)
        risk_category = max(risk_scores, key=risk_scores.__getitem__)
        risk_confidence = risk_scores[risk_category]

        guarded_risk = _crisis_override(review.text)
        if guarded_risk is not None:
            risk_category = guarded_risk
            risk_confidence = 1.0
            risk_scores = {**risk_scores, guarded_risk: 1.0}

        if risk_category == "safety or food hygiene concern":
            aspect = "food safety"
        elif risk_category == "fraud, payment dispute, or financial misconduct":
            aspect = "digital ordering and billing"

        lane, risk_band, owner = _route(risk_category, review.rating, risk_confidence)
        reply_draft = _reply_for(review, aspect) if lane is not Lane.MANAGER_HOLD else None
        rationale = _rationale(lane, risk_category, aspect, risk_confidence)

        return TriageResult(
            review=review,
            aspect=aspect,
            aspect_confidence=aspect_scores[aspect],
            risk_category=risk_category,
            risk_band=risk_band,
            risk_confidence=risk_confidence,
            lane=lane,
            route_owner=owner,
            publish_allowed=lane is Lane.AUTO_PUBLISH,
            rationale=rationale,
            reply_draft=reply_draft,
            aspect_scores=aspect_scores,
            risk_scores=risk_scores,
        )


def _crisis_override(text: str) -> str | None:
    normalized = " ".join(text.casefold().split())
    for risk, cues in _CRISIS_CUES:
        if any(cue in normalized for cue in cues):
            return risk
    return None


def _route(risk: str, rating: int, confidence: float) -> tuple[Lane, RiskBand, str]:
    if risk in CRISIS_RISKS:
        band = (
            RiskBand.CRITICAL
            if risk == "legal, regulatory, or litigation threat"
            else RiskBand.HIGH
        )
        return Lane.MANAGER_HOLD, band, "Manager"
    if (
        risk == "branch-level service or operations complaint"
        or rating <= 3
        or confidence < AUTO_PUBLISH_CONFIDENCE_THRESHOLD
    ):
        return Lane.ADVISOR_QUEUE, RiskBand.MODERATE, "Advisor"
    return Lane.AUTO_PUBLISH, RiskBand.LOW, "Admin policy"


def _reply_for(review: ReviewInput, aspect: str) -> str:
    if aspect in {"food quality", "food safety"}:
        acknowledgement = "We’re glad you enjoyed the food"
    elif aspect in {"service speed", "staff conduct"}:
        acknowledgement = "We’re glad our team made your visit"
    elif aspect == "pricing and value":
        acknowledgement = "We appreciate you sharing your feedback"
    else:
        acknowledgement = "We’re glad you shared your experience"
    return (
        f"Thank you for visiting our {review.branch} location. "
        f"{acknowledgement}. We hope to welcome you back soon."
    )


def _rationale(lane: Lane, risk: str, aspect: str, confidence: float) -> str:
    if lane is Lane.MANAGER_HOLD:
        return (
            f"Potential {risk} signal detected. Block public auto-reply and escalate "
            "to regional management for immediate review."
        )
    if lane is Lane.ADVISOR_QUEUE:
        if confidence < AUTO_PUBLISH_CONFIDENCE_THRESHOLD:
            return (
                "The model is uncertain about the review's risk. Hold it for Advisor "
                "verification rather than auto-publishing."
            )
        return (
            f"Review concerns {aspect} at a specific branch. Advisor verification "
            "is required before publishing a response."
        )
    return (
        f"Routine, low-risk feedback about {aspect}; eligible for the approved "
        "AI reply policy."
    )
