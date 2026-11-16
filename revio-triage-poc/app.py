"""Streamlit interface for Revio AI Triage."""

from __future__ import annotations

import csv
import io
from html import escape
from typing import Any

import altair as alt
import streamlit as st
from requests.exceptions import RequestException

from engine import HuggingFaceNLIClassifier, Lane, ReviewInput, TriageEngine, TriageResult

MODEL_NAME = "valhalla/distilbart-mnli-12-1"

SAMPLE_REVIEWS = [
    ReviewInput(
        branch="Cafe A",
        reviewer="Asha K.",
        rating=5,
        text="Lovely cappuccino and a fresh croissant. The barista was warm and quick.",
    ),
    ReviewInput(
        branch="Cafe A",
        reviewer="Rahul M.",
        rating=2,
        text="Waited 35 minutes for my takeaway order and it was still missing a sandwich.",
    ),
    ReviewInput(
        branch="Kitchen B",
        reviewer="Guest 104",
        rating=1,
        text="Found a cockroach near the serving counter. I told staff but nobody responded.",
    ),
    ReviewInput(
        branch="Kitchen A",
        reviewer="Mira P.",
        rating=4,
        text="Great brunch, cheerful staff, and a comfortable patio. Will come again.",
    ),
    ReviewInput(
        branch="Cafe B",
        reviewer="Guest 882",
        rating=2,
        text="The app charged me twice for one order and the refund is still missing.",
    ),
    ReviewInput(
        branch="Kitchen B",
        reviewer="Dev S.",
        rating=3,
        text="The food was good, but the dining area and tables needed attention.",
    ),
    ReviewInput(
        branch="Cafe C",
        reviewer="Nina R.",
        rating=5,
        text="Excellent flat white and a peaceful place to work for an hour.",
    ),
    ReviewInput(
        branch="Kitchen C",
        reviewer="Guest 226",
        rating=1,
        text="I have contacted a lawyer and will take legal action over this incident.",
    ),
]

st.set_page_config(
    page_title="Revio AI Triage | Cynbit",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root { --ink:#18312f; --muted:#5e7471; --teal:#16756d; --paper:#f4f7f6; --line:#dce7e3; }
    .stApp, [data-testid="stAppViewContainer"] { background:var(--paper); color:var(--ink); }
    [data-testid="stMain"] { color:var(--ink); }
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li, label, [data-testid="stCaptionContainer"] {
      color:var(--ink);
    }
    h1,h2,h3,h4 { color:var(--ink) !important; letter-spacing:-.025em; }
    [data-testid="stSidebar"] { background:#eaf1ef; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] * { color:var(--ink); }
    .hero { background:linear-gradient(115deg,#103c3b,#126e67 66%,#20827a); color:#fff;
      border-radius:18px; padding:26px 30px; margin:2px 0 24px; }
    .hero-kicker { color:#d3ed9b; font-size:11px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
    .hero-title { color:#fff; font-size:clamp(23px,3vw,34px); line-height:1.15;
      font-weight:700; margin:9px 0 8px; }
    .hero-subtitle { color:#fff; font-size:16px; font-weight:600; margin-bottom:5px; }
    .hero-copy { color:#e1eeeb; font-size:14px; max-width:760px; line-height:1.6; }
    .eyebrow { color:var(--muted); font-size:11px; font-weight:700; letter-spacing:.1em;
      text-transform:uppercase; margin-bottom:5px; }
    .lane-card { min-height:135px; background:#fff; border:1px solid var(--line);
      border-top:4px solid var(--teal); border-radius:12px; padding:14px 16px;
      color:var(--ink); line-height:1.5; }
    .lane-card b { display:block; font-size:14px; margin-bottom:6px; color:var(--ink); }
    .lane-card span { display:block; color:#4d625f; font-size:13px; }
    .lane-card.advisor { border-top-color:#bd7810; }
    .lane-card.crisis { border-top-color:#bd3e35; }
    .summary-field { min-height:74px; padding:11px 13px; border:1px solid;
      border-radius:11px; display:flex; flex-direction:column; justify-content:center;
      gap:5px; overflow:hidden; }
    .summary-label { font-size:10px; font-weight:700; letter-spacing:.09em;
      line-height:1.2; text-transform:uppercase; opacity:.78; }
    .summary-value { font-size:13px; font-weight:700; line-height:1.28;
      overflow-wrap:anywhere; }
    .route-auto, .risk-low { background:#e9f4ec; border-color:#b8d7c0; color:#205c36; }
    .route-advisor, .risk-moderate { background:#fff4dc; border-color:#ead29a; color:#80580a; }
    .route-manager, .risk-high, .risk-critical { background:#fdebea; border-color:#e9b9b6; color:#942f29; }
    .aspect-field { background:#e9f1fb; border-color:#bed0e8; color:#294e78; }
    div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line);
      border-radius:12px; padding:12px 15px; }
    div[data-testid="stMetricLabel"] { color:var(--muted); }
    div[data-testid="stForm"] { background:#fff; border:1px solid var(--line);
      border-radius:14px; padding:18px 20px; }
    [data-testid="stTextInput"] [data-baseweb="input"],
    [data-testid="stTextArea"] [data-baseweb="textarea"] {
      background:#eaf3f0 !important; border:1px solid #9bb5ae !important;
      border-radius:9px !important; box-shadow:inset 0 1px 2px #18312f0a;
    }
    [data-testid="stTextInput"] [data-baseweb="base-input"],
    [data-testid="stTextArea"] [data-baseweb="base-input"] {
      background:transparent !important;
    }
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
      background:transparent !important; color:var(--ink) !important;
    }
    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder { color:#647a75 !important; opacity:1; }
    [data-testid="stTextInput"] [data-baseweb="input"]:focus-within,
    [data-testid="stTextArea"] [data-baseweb="textarea"]:focus-within {
      background:#fff !important; border-color:var(--teal) !important;
      box-shadow:0 0 0 3px #16756d24 !important;
    }
    div.stButton > button[kind="primary"], button[kind="primaryFormSubmit"] {
      background:#16756d; border-color:#16756d; color:#fff; font-weight:700;
      border-radius:9px; min-height:42px;
    }
    div.stButton > button[kind="primary"] p,
    button[kind="primaryFormSubmit"] p { color:#fff !important; }
    div.stButton > button[kind="primary"]:hover,
    button[kind="primaryFormSubmit"]:hover { background:#105b55; border-color:#105b55; color:#fff; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; overflow:hidden; }
    @media (max-width: 700px) {
      .hero { padding:21px 20px; }
      [data-testid="stForm"] { padding:14px; }
      .lane-card { min-height:0; margin-bottom:8px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_engine(model_name: str) -> TriageEngine:
    return TriageEngine(HuggingFaceNLIClassifier(model_name))


@st.cache_data(show_spinner=False)
def cached_triage(
    model_name: str, text: str, branch: str, rating: int, reviewer: str | None
) -> dict[str, Any]:
    review = ReviewInput(text=text, branch=branch, rating=rating, reviewer=reviewer)
    return get_engine(model_name).analyze(review).model_dump(mode="json")


def triage(review: ReviewInput) -> TriageResult:
    result = cached_triage(
        MODEL_NAME,
        review.text,
        review.branch,
        review.rating,
        review.reviewer,
    )
    return TriageResult.model_validate(result)


def show_model_error(error: Exception) -> None:
    st.error(f"Triage could not run: {error}")
    st.info(
        "Confirm the requirements are installed and that this environment can "
        f"download the Hugging Face model `{MODEL_NAME}`."
    )


def render_result(result: TriageResult, show_review: bool = True) -> None:
    with st.container(border=True):
        if show_review:
            st.markdown(f"**{result.review.branch}** · {result.review.rating}/5 stars")
            st.write(result.review.text)
        route_style = {
            Lane.AUTO_PUBLISH: "route-auto",
            Lane.ADVISOR_QUEUE: "route-advisor",
            Lane.MANAGER_HOLD: "route-manager",
        }[result.lane]
        risk_style = f"risk-{result.risk_band.value.casefold()}"
        summary_fields = (
            ("Route", result.lane.value, route_style),
            ("Aspect", result.aspect.title(), "aspect-field"),
            ("Risk", f"{result.risk_band.value} · {result.risk_confidence:.0%}", risk_style),
        )
        columns = st.columns(3)
        for column, (label, value, style) in zip(columns, summary_fields, strict=True):
            with column:
                st.markdown(
                    f'<div class="summary-field {style}">'
                    f'<span class="summary-label">{escape(label)}</span>'
                    f'<span class="summary-value">{escape(value)}</span></div>',
                    unsafe_allow_html=True,
                )
        st.caption(f"Assigned to: {result.route_owner}")
        st.write(result.rationale)
        if result.lane is Lane.MANAGER_HOLD:
            st.error("Auto-reply blocked. Escalate to regional management.")
        elif result.reply_draft:
            label = "Auto-reply eligible" if result.publish_allowed else "Advisor draft · verify before publishing"
            st.markdown(f"**{label}**")
            st.info(result.reply_draft)


def render_inspector() -> None:
    st.markdown('<div class="eyebrow">Review inspector</div>', unsafe_allow_html=True)
    st.subheader("Analyze a customer review")
    st.write("Add the branch, star rating, and review text to get an aspect, risk level, and next step.")
    with st.form("review_inspector"):
        branch_col, reviewer_col = st.columns([2, 1])
        branch = branch_col.text_input("Branch", value="Cafe A")
        reviewer = reviewer_col.text_input("Reviewer (optional)")
        rating = st.slider("Star rating", min_value=1, max_value=5, value=3)
        text = st.text_area(
            "Review text",
            value="The latte was excellent, but I waited 25 minutes for my order.",
            height=120,
            placeholder="Paste the customer review here...",
        )
        submitted = st.form_submit_button("Analyze review", type="primary", use_container_width=True)
    if submitted:
        try:
            review = ReviewInput(
                branch=branch,
                reviewer=reviewer or None,
                rating=rating,
                text=text,
            )
            with st.spinner("Scoring aspect and risk with the NLI model..."):
                result = triage(review)
            st.session_state["last_result"] = result.model_dump(mode="json")
        except (OSError, RequestException, RuntimeError, ValueError) as error:
            show_model_error(error)
    prior_result = st.session_state.get("last_result")
    if prior_result:
        st.subheader("Triage result")
        render_result(TriageResult.model_validate(prior_result))

    st.markdown('<div class="eyebrow">Three routing lanes</div>', unsafe_allow_html=True)
    lane_columns = st.columns(3)
    lane_cards = (
        ("lane-card", "Lane A · Auto-Publish", "Routine praise or low-risk feedback. AI reply can publish immediately."),
        ("lane-card advisor", "Lane B · Advisor Queue", "Branch-level operations issue. Verify the draft before publishing."),
        ("lane-card crisis", "Lane C · Manager Hold", "Safety, hygiene, fraud, or legal risk. Auto-reply is blocked."),
    )
    for column, (style, title, description) in zip(lane_columns, lane_cards, strict=True):
        with column:
            st.markdown(
                f'<div class="{style}"><b>{title}</b><span>{description}</span></div>',
                unsafe_allow_html=True,
            )


def parse_csv(contents: bytes) -> list[ReviewInput]:
    try:
        decoded = contents.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(decoded))
        if not reader.fieldnames or "review" not in reader.fieldnames:
            raise ValueError("CSV must include a `review` column.")
        reviews: list[ReviewInput] = []
        for line_number, row in enumerate(reader, start=2):
            text = (row.get("review") or "").strip()
            if not text:
                continue
            rating_value = (row.get("rating") or "3").strip()
            try:
                rating = int(rating_value)
            except ValueError as exc:
                raise ValueError(f"Invalid rating on CSV row {line_number}.") from exc
            reviews.append(
                ReviewInput(
                    text=text,
                    branch=(row.get("branch") or "Unspecified branch").strip(),
                    rating=rating,
                    reviewer=(row.get("reviewer") or "").strip() or None,
                )
            )
        if not reviews:
            raise ValueError("CSV contains no non-empty reviews.")
        return reviews
    except UnicodeDecodeError as exc:
        raise ValueError("CSV must be UTF-8 encoded.") from exc


def render_dashboard() -> None:
    st.markdown('<div class="eyebrow">Multi-branch command view</div>', unsafe_allow_html=True)
    st.subheader("Branch review dashboard")
    st.write("Triage a sample set or upload your own review CSV.")
    upload = st.file_uploader("Upload review CSV", type=["csv"])
    if upload is not None:
        try:
            reviews = parse_csv(upload.getvalue())
        except ValueError as error:
            st.error(str(error))
            return
        source_label = f"Uploaded · {len(reviews)} reviews"
    else:
        reviews = SAMPLE_REVIEWS
        source_label = f"Demo dataset · {len(reviews)} reviews"
        st.caption("Upload a CSV with `review` and optional `branch`, `rating`, `reviewer` columns.")

    st.caption(f"{source_label}. Results are triaged when you run the dashboard.")
    if not st.button("Run branch triage", type="primary"):
        st.dataframe(
            [
                {
                    "Branch": review.branch,
                    "Rating": review.rating,
                    "Reviewer": review.reviewer or "—",
                    "Review": review.text,
                }
                for review in reviews
            ],
            use_container_width=True,
            hide_index=True,
        )
        return

    results: list[TriageResult] = []
    progress = st.progress(0, text="Preparing triage...")
    try:
        for index, review in enumerate(reviews, start=1):
            progress.progress(index / len(reviews), text=f"Analyzing {index}/{len(reviews)}")
            results.append(triage(review))
    except (OSError, RequestException, RuntimeError, ValueError) as error:
        progress.empty()
        show_model_error(error)
        return
    progress.empty()

    lane_counts = {lane: sum(result.lane is lane for result in results) for lane in Lane}
    metric_columns = st.columns(4)
    metric_columns[0].metric("Reviews analyzed", len(results))
    metric_columns[1].metric("Auto-publish", lane_counts[Lane.AUTO_PUBLISH])
    metric_columns[2].metric("Advisor queue", lane_counts[Lane.ADVISOR_QUEUE])
    metric_columns[3].metric("Manager holds", lane_counts[Lane.MANAGER_HOLD])

    st.subheader("Review volume by branch")
    branch_counts: dict[str, dict[str, int]] = {}
    for result in results:
        counts = branch_counts.setdefault(
            result.review.branch, {lane.value: 0 for lane in Lane}
        )
        counts[result.lane.value] += 1
    chart_data = [
        {
            "Branch": branch,
            "Reviews": sum(counts.values()),
            "Auto-publish": counts[Lane.AUTO_PUBLISH.value],
            "Advisor queue": counts[Lane.ADVISOR_QUEUE.value],
            "Manager hold": counts[Lane.MANAGER_HOLD.value],
        }
        for branch, counts in branch_counts.items()
    ]
    chart_data.sort(key=lambda row: (-row["Reviews"], row["Branch"]))
    branch_order = [row["Branch"] for row in chart_data]
    st.caption(
        "Each bar shows total reviews for one branch. Hover to see the Lane A/B/C split; "
        "individual routes are listed below."
    )
    chart_base = (
        alt.Chart(alt.Data(values=chart_data))
        .encode(
            x=alt.X(
                "Reviews:Q",
                title="Reviews",
                scale=alt.Scale(
                    domain=[0, max(row["Reviews"] for row in chart_data) + 0.5],
                    nice=False,
                ),
                axis=alt.Axis(format="d", tickMinStep=1),
            ),
            y=alt.Y(
                "Branch:N",
                sort=branch_order,
                title=None,
                axis=alt.Axis(labelLimit=180),
            ),
            tooltip=[
                alt.Tooltip("Branch:N"),
                alt.Tooltip("Reviews:Q", format="d"),
                alt.Tooltip("Auto-publish:Q", format="d", title="Lane A · Auto-publish"),
                alt.Tooltip("Advisor queue:Q", format="d", title="Lane B · Advisor"),
                alt.Tooltip("Manager hold:Q", format="d", title="Lane C · Manager"),
            ],
        )
    )
    bars = chart_base.mark_bar(color="#16756d", size=22, cornerRadiusEnd=4)
    labels = chart_base.mark_text(
        align="left",
        baseline="middle",
        dx=7,
        color="#18312f",
        fontWeight="bold",
    ).encode(text=alt.Text("Reviews:Q", format="d"))
    chart = (
        (bars + labels)
        .properties(height=min(460, max(200, 40 * len(chart_data))))
        .configure_view(stroke=None)
        .configure_axis(gridColor="#dce7e3", labelColor="#5e7471", titleColor="#18312f")
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Review triage")
    st.dataframe(
        [
            {
                "Branch": result.review.branch,
                "Rating": result.review.rating,
                "Aspect": result.aspect,
                "Risk": result.risk_band.value,
                "Risk confidence": f"{result.risk_confidence:.0%}",
                "Route": result.lane.value,
                "Owner": result.route_owner,
                "Publish allowed": "Yes" if result.publish_allowed else "No",
                "Review": result.review.text,
            }
            for result in results
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Escalation details")
    for result in results:
        if result.lane is Lane.MANAGER_HOLD:
            with st.expander(f"{result.review.branch} · {result.risk_category}"):
                render_result(result)


st.sidebar.markdown("## ◈ Revio")
st.sidebar.markdown("**AI Triage** · Cynbit AI/ML Proof of Concept")
st.sidebar.markdown("### Model")
st.sidebar.caption(f"Hugging Face zero-shot NLI · `{MODEL_NAME}`")
st.sidebar.info("Decision support only. Lane B requires advisor verification; Lane C is held for management.")

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Cynbit AI/ML Proof of Concept · Reputation intelligence</div>
      <div class="hero-title">Revio AI Triage</div>
      <div class="hero-subtitle">Aspect Attribution &amp; Risk-Aware Reply Routing</div>
      <div class="hero-copy">Turn restaurant reviews into a clear next step.
      Low-risk praise can move quickly; operational issues get verified; critical
      complaints are held for management.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

inspector_tab, dashboard_tab = st.tabs(["Review inspector", "Branch dashboard"])
with inspector_tab:
    render_inspector()
with dashboard_tab:
    render_dashboard()
