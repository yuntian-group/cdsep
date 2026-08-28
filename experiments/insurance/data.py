"""Synthetic underwriting dataset (v2): medical-condition based.

Each applicant has a small set of medical impairments drawn from a
12-condition synthetic underwriting taxonomy. Ground-truth ratings are
deterministic and apply the evaluation rules (floor at 0; age > 75 ->
Decline).

The generated inputs mimic the format of broker-to-underwriter quick
quotes: an applicant block (age, sex, and build), family history, and a
sequence of impairment lines.

This module is shared by ``experiments/insurance/agents.py`` and
``experiments/insurance/run.py`` and is intentionally close in shape to
``experiments/insurance_real/data.py`` so the synthetic and real
pipelines stay comparable.
"""

from __future__ import annotations

import random
from typing import Literal

# Rating bucket set, matching the real-data buckets so the two
# experiments can share aggregator logic.
RATING_BUCKETS: list[str] = [
    "0", "50", "100", "150", "175", "200", "250", "300",
    "decline", "postpone",
]

# Twelve impairment chapters with deterministic per-severity rating
# tables. The mapping is deliberately simple so the deterministic ground
# truth is checkable end-to-end.
CONDITIONS: dict[str, dict] = {
    "Diabetes": {
        "severities": ["mild", "moderate", "severe"],
        "rating_table": {"mild": 50, "moderate": 150, "severe": 250},
        "snippets": {
            "mild":     "Diabetes Type 2, HbA1c 6.7 (3/2024), no complications, on metformin only.",
            "moderate": "Diabetes Type 2 since 2018, HbA1c 8.2 (4/2024), on metformin + insulin, no neuropathy.",
            "severe":   "Diabetes Type 2 since 2010, HbA1c 9.8 (2/2024), insulin-dependent, mild retinopathy.",
        },
    },
    "Hypertension": {
        "severities": ["mild", "moderate", "severe"],
        "rating_table": {"mild": 0, "moderate": 50, "severe": 150},
        "snippets": {
            "mild":     "BP 130/82 (2/2024), well controlled on lisinopril.",
            "moderate": "BP 148/92 (1/2024), on dual therapy (lisinopril + amlodipine).",
            "severe":   "BP 168/104 (3/2024), poorly controlled, recent hospitalization for HTN urgency.",
        },
    },
    "Coronary Artery Disease": {
        "severities": ["mild", "moderate", "severe"],
        "rating_table": {"mild": 150, "moderate": 250, "severe": "decline"},
        "snippets": {
            "mild":     "CAD, asymptomatic, single-vessel, stable on statin (LDL 78).",
            "moderate": "CAD with PCI 2 yrs ago (LAD stent), EF 55%, asymptomatic on ASA + statin.",
            "severe":   "CAD with MI 6 months ago, EF 35%, ongoing chest pain on exertion.",
        },
    },
    "Cancer History": {
        "severities": ["mild", "moderate", "severe"],
        "rating_table": {"mild": 100, "moderate": 250, "severe": "decline"},
        "snippets": {
            "mild":     "Hx breast CA stage I, dx 2018, s/p lumpectomy + radiation, NED on 5-yr followup.",
            "moderate": "Hx colon CA stage II, dx 2021, s/p resection, NED on imaging 3/2024.",
            "severe":   "Active prostate CA, Gleason 8, on hormonal therapy, recent PSA rising.",
        },
    },
    "Obesity": {
        "severities": ["mild", "moderate", "severe"],
        "rating_table": {"mild": 0, "moderate": 50, "severe": 150},
        "snippets": {
            "mild":     "Build 5'10\" / 200 lbs (BMI 28.7), no related comorbidities.",
            "moderate": "Build 5'8\" / 240 lbs (BMI 36.5), borderline hypertension.",
            "severe":   "Build 5'6\" / 290 lbs (BMI 46.8), OSA on CPAP, mobility limited.",
        },
    },
    "Tobacco Use": {
        "severities": ["former", "current"],
        "rating_table": {"former": 50, "current": 150},
        "snippets": {
            "former":  "Former smoker, quit 5 years ago after 10 pack-years.",
            "current": "Current smoker, 1 ppd x 15 years.",
        },
    },
    "Asthma": {
        "severities": ["mild", "moderate", "severe"],
        "rating_table": {"mild": 0, "moderate": 50, "severe": 100},
        "snippets": {
            "mild":     "Mild intermittent asthma, rescue inhaler only.",
            "moderate": "Persistent asthma on ICS + LABA, last exacerbation 18 mo ago.",
            "severe":   "Severe asthma, 2 ED visits in last 12 mo, on biologic therapy.",
        },
    },
    "Sleep Apnea": {
        "severities": ["mild", "severe"],
        "rating_table": {"mild": 0, "severe": 50},
        "snippets": {
            "mild":   "OSA, compliant on CPAP for 3 yrs, AHI down to 4.",
            "severe": "Untreated OSA dx 2022, declined CPAP, daytime somnolence.",
        },
    },
    "Mental Health": {
        "severities": ["mild", "moderate", "severe"],
        "rating_table": {"mild": 0, "moderate": 50, "severe": 150},
        "snippets": {
            "mild":     "Mild GAD, on low-dose SSRI for 2 yrs, fully functional.",
            "moderate": "MDD recurrent, on sertraline + therapy, last episode 1 yr ago.",
            "severe":   "Bipolar I with hospitalization 6 mo ago, on lithium + atypical.",
        },
    },
    "Alcohol Use": {
        "severities": ["moderate", "heavy"],
        "rating_table": {"moderate": 0, "heavy": 150},
        "snippets": {
            "moderate": "Social drinking, 4-5 drinks per week.",
            "heavy":    "Hx alcohol abuse, in recovery 14 mo, attends AA weekly.",
        },
    },
    "Family History": {
        "severities": ["positive"],
        "rating_table": {"positive": 50},
        "snippets": {
            "positive": "FH: father MI @55, mother T2DM, MGM breast CA.",
        },
    },
    "Avocation": {
        "severities": ["moderate", "high"],
        "rating_table": {"moderate": 50, "high": 175},
        "snippets": {
            "moderate": "Avocation: recreational scuba diving, <60 ft, certified 8 yrs.",
            "high":     "Avocation: amateur motorcycle racing, 4 events per year.",
        },
    },
}

# Health credits that can offset debits (kept small and explicit so the
# floor-at-zero rule is exercised).
HEALTH_CREDITS: list[tuple[str, int]] = [
    ("Non-smoker, BMI 22, walks 30 min/day, drinks rarely.", -25),
    ("Marathon runner, BMI 21, vegetarian, zero medications.", -50),
    ("Strict Mediterranean diet, normal lipid panel.", -25),
]


def _occupation() -> str:
    return random.choice([
        "RN", "teacher", "software developer", "accountant", "small-business owner",
        "construction worker", "electrician", "firefighter", "real-estate agent",
        "office manager", "retired", "physician",
    ])


def _state() -> str:
    return random.choice(["PA", "NY", "CA", "TX", "FL", "ON", "QC", "BC", "IL", "WA"])


def _sample_severity(rng: random.Random, severities: list[str]) -> str:
    # Weight toward mild/moderate so the data isn't dominated by Decline.
    weights = [3, 2, 1][:len(severities)]
    if len(severities) == 2:
        weights = [3, 1]
    return rng.choices(severities, weights=weights, k=1)[0]


def _bucket_for_int(value: int) -> str:
    """Snap an integer rating to the closest bucket label."""
    if value < 0:
        return "0"
    numeric_buckets = [0, 50, 100, 150, 175, 200, 250, 300]
    nearest = min(numeric_buckets, key=lambda b: abs(b - value))
    return str(nearest)


def compute_ground_truth(
    conditions: list[tuple[str, str]],
    credits: list[tuple[str, int]],
    age: int,
) -> tuple[str, list[dict]]:
    """Apply per-condition ratings, sum debits and credits, then apply the rules.

    Returns (final_bucket, condition_workup) where condition_workup is
    the list of dicts the agent should ideally produce.
    """
    workup: list[dict] = []
    has_decline = False
    has_postpone = False
    debit_sum = 0

    for name, severity in conditions:
        rating = CONDITIONS[name]["rating_table"][severity]
        workup.append({"name": name, "severity": severity, "rating": str(rating).lower()})
        if rating == "decline":
            has_decline = True
        elif rating == "postpone":
            has_postpone = True
        elif isinstance(rating, int):
            debit_sum += rating

    for snippet, delta in credits:
        debit_sum += delta
        workup.append({"name": "Health Credit", "severity": snippet, "rating": str(delta)})

    if age > 75:
        return "decline", workup
    if has_decline:
        return "decline", workup
    if has_postpone:
        return "postpone", workup
    return _bucket_for_int(debit_sum), workup


def render_email(
    *,
    age: int,
    sex: str,
    occupation: str,
    state: str,
    build: str,
    conditions: list[tuple[str, str]],
    credits: list[tuple[str, int]],
) -> str:
    """Format a broker quick-quote email from the structured fields."""
    body = [
        f"Need a quote on a {age}YO{sex[0]} applying for $750,000 of TERM coverage with the following history. Please advise of the best rate possible.",
        "",
        f"State of Sale : {state}, Occupation : {occupation}. Build : {build}.",
        "",
    ]
    for name, severity in conditions:
        snippet = CONDITIONS[name]["snippets"][severity]
        body.append(f"* {snippet}")
    for snippet, _delta in credits:
        body.append(f"  Healthy factor: {snippet}")
    body.append("")
    body.append("STATEMENT OF HEALTH attached.")
    return "\n".join(body)


def generate_applicant(rng: random.Random) -> dict:
    """Generate one synthetic quick-quote case with a deterministic label."""
    age = rng.randint(25, 80)
    sex = rng.choice(["F", "M"])
    occupation = rng.choice([
        "RN", "teacher", "software developer", "accountant", "small-business owner",
        "construction worker", "electrician", "firefighter", "real-estate agent",
        "office manager", "retired", "physician",
    ])
    state = rng.choice(["PA", "NY", "CA", "TX", "FL", "ON", "QC", "BC", "IL", "WA"])

    height_in = rng.randint(60, 76)
    weight_lb = rng.randint(120, 280)
    bmi = (weight_lb / (height_in ** 2)) * 703
    build = f"{height_in // 12}'{height_in % 12}\" / {weight_lb} lbs (BMI {bmi:.1f})"

    n_conditions = rng.choices([1, 2, 3], weights=[4, 4, 2])[0]
    condition_names = rng.sample(list(CONDITIONS.keys()), k=n_conditions)
    conditions: list[tuple[str, str]] = []
    for name in condition_names:
        severity = _sample_severity(rng, CONDITIONS[name]["severities"])
        conditions.append((name, severity))

    credits: list[tuple[str, int]] = []
    if rng.random() < 0.4:
        credits.append(rng.choice(HEALTH_CREDITS))

    rating, workup = compute_ground_truth(conditions, credits, age)
    email = render_email(
        age=age, sex=sex, occupation=occupation, state=state, build=build,
        conditions=conditions, credits=credits,
    )

    return {
        "description": email,
        "age": age,
        "sex": sex,
        "occupation": occupation,
        "state": state,
        "build": build,
        "conditions": [
            {"name": n, "severity": s, "rating": str(CONDITIONS[n]["rating_table"][s]).lower()}
            for n, s in conditions
        ],
        "credits": credits,
        "ground_truth": rating,
        "workup": workup,
    }


def generate_dataset(
    n_samples: int = 90,
    seed: int = 42,
    n_train: int | None = None,
    n_val: int | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Train / val / test split of medical-impairment underwriting cases."""
    rng = random.Random(seed)
    data = [generate_applicant(rng) for _ in range(n_samples)]
    if n_train is None:
        n_train = int(n_samples * 2 / 3)
    if n_val is None:
        n_val = 0
    n_test = n_samples - n_train - n_val
    return data[:n_train], data[n_train : n_train + n_val], data[n_train + n_val :]


CONDITION_NAMES: list[str] = list(CONDITIONS.keys())
"""The closed Literal universe for the medical-condition extractor."""

SEVERITY_LEVELS: list[str] = ["mild", "moderate", "severe", "former", "current", "positive", "high"]
"""Union of severity labels across conditions."""


if __name__ == "__main__":
    rng = random.Random(0)
    print("=== Sample synthetic cases ===\n")
    for _ in range(3):
        case = generate_applicant(rng)
        print(case["description"])
        print(f"GT: {case['ground_truth']}  conditions: {case['conditions']}")
        print("-" * 60)
