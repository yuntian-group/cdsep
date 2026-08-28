"""Agent definitions for the medical-impairment underwriting pipeline.

We use the same three-agent sequential structure as the original
demographic version (extractor -> rater -> aggregator), but the
semantics model a medical underwriting workflow:

* ``medical_extractor`` reads the broker email and identifies the
  primary impairment chapter plus the total number of impairments. The
  per-impairment details (name, severity, clinical detail) live in the
  free-text *data flow* of the response, where the optimizer can edit.
* ``impairment_rater`` reads the extractor's structured summary and
  assigns the worst per-condition rating bucket. Again, per-condition
  rating details live in the free-text channel.
* ``final_aggregator`` applies the two evaluation rules
  (age > 75 -> Decline; never produce a negative numeric rating) and
  emits the final bucket label.

Every routing-critical field is a closed ``Literal`` type, so by the
Protocol-Stability Lemma the optimizer cannot break parsing or routing
no matter how aggressively it edits the data-flow prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cdsep.agent import Agent

from experiments.insurance.data import CONDITION_NAMES, RATING_BUCKETS

CONDITION_LIT = Literal[
    "Diabetes",
    "Hypertension",
    "Coronary Artery Disease",
    "Cancer History",
    "Obesity",
    "Tobacco Use",
    "Asthma",
    "Sleep Apnea",
    "Mental Health",
    "Alcohol Use",
    "Family History",
    "Avocation",
    "none",
]

RATING_LIT = Literal[
    "0", "50", "100", "150", "175", "200", "250", "300",
    "decline", "postpone",
]

assert set(CONDITION_NAMES) <= set(CONDITION_LIT.__args__)
assert set(RATING_BUCKETS) <= set(RATING_LIT.__args__)


@dataclass
class MedicalExtractorControl:
    primary_chapter: CONDITION_LIT
    n_impairments: int


@dataclass
class ImpairmentRaterControl:
    worst_impairment_rating: RATING_LIT
    all_impairments_standard: bool


@dataclass
class FinalAggregatorControl:
    final_rating: RATING_LIT
    age_over_75: bool


# ---------------------------------------------------------------------------
# Separated prompts (D2 from the paper: schema lives in a frozen slot the
# optimizer never sees; these prompts are pure data-flow guidance).
# ---------------------------------------------------------------------------

MEDICAL_EXTRACTOR_PROMPT = """\
You are a junior underwriter triaging a broker quick-quote email.

Your job: read the email and identify the medical or non-medical
impairments the applicant has. For each impairment, record:

- chapter (which underwriting chapter applies, e.g. Diabetes,
  Hypertension, Coronary Artery Disease, Cancer History, Obesity,
  Tobacco Use, Asthma, Sleep Apnea, Mental Health, Alcohol Use,
  Family History, Avocation),
- severity (mild / moderate / severe / current / former / positive
  / high),
- a one-sentence clinical detail copied or paraphrased from the email.

If the email also mentions favorable factors (healthy lifestyle,
clean labs, etc.) list them as "Health Credit" with a one-sentence
summary; do not invent any.

In your message, list one impairment per line. Be specific. Do not
include a rating yet -- the next agent will do that."""

IMPAIRMENT_RATER_PROMPT = """\
You are a senior underwriter rating each impairment individually.

The previous agent has produced a list of impairments and any health
credits. For each impairment, decide the appropriate rating using the
following simplified underwriting chapter set:

- Diabetes: mild -> 50, moderate -> 150, severe -> 250.
- Hypertension: mild -> 0, moderate -> 50, severe -> 150.
- Coronary Artery Disease: mild -> 150, moderate -> 250, severe -> Decline.
- Cancer History: mild -> 100, moderate -> 250, severe -> Decline.
- Obesity: mild -> 0, moderate -> 50, severe -> 150.
- Tobacco Use: former -> 50, current -> 150.
- Asthma: mild -> 0, moderate -> 50, severe -> 100.
- Sleep Apnea: mild -> 0, severe -> 50.
- Mental Health: mild -> 0, moderate -> 50, severe -> 150.
- Alcohol Use: moderate -> 0, heavy -> 150.
- Family History: positive -> 50.
- Avocation: moderate -> 50, high -> 175.

For health credits, use -25 or -50 as stated in the email.

In your message, list one line per impairment with its rating, then
report the worst per-condition rating in the structured field."""

FINAL_AGGREGATOR_PROMPT = """\
You are the final underwriting officer signing off on the case.

Combine all per-impairment ratings into one final rating:

1. If the applicant is older than 75, the final rating is "decline"
   regardless of the medical picture. Set ``age_over_75`` to true.
2. If any impairment is "decline" or "postpone", the final rating is
   that string (decline beats postpone).
3. Otherwise, sum all numeric debits and credits. If the sum is
   negative, floor it at 0. Snap the result to the closest available
   bucket among 0, 50, 100, 150, 175, 200, 250, 300.

In your message, show the arithmetic and the rule that fired."""


# ---------------------------------------------------------------------------
# Naive prompts: schema instructions are inlined into the prompt, so the
# optimizer can (and does) edit them out.
# ---------------------------------------------------------------------------

NAIVE_MEDICAL_EXTRACTOR_PROMPT = MEDICAL_EXTRACTOR_PROMPT + """

Output format: respond with a JSON object on the first line with these
fields:
- primary_chapter: one of \"Diabetes\", \"Hypertension\", \
\"Coronary Artery Disease\", \"Cancer History\", \"Obesity\", \
\"Tobacco Use\", \"Asthma\", \"Sleep Apnea\", \"Mental Health\", \
\"Alcohol Use\", \"Family History\", \"Avocation\", or \"none\"
- n_impairments: integer count
Example: {\"primary_chapter\": \"Diabetes\", \"n_impairments\": 2}

After the JSON, list the impairments."""

NAIVE_IMPAIRMENT_RATER_PROMPT = IMPAIRMENT_RATER_PROMPT + """

Output format: respond with a JSON object on the first line with these
fields:
- worst_impairment_rating: one of \"0\", \"50\", \"100\", \"150\", \"175\", \
\"200\", \"250\", \"300\", \"decline\", \"postpone\"
- all_impairments_standard: boolean (true if every impairment rates 0)
Example: {\"worst_impairment_rating\": \"150\", \"all_impairments_standard\": false}

After the JSON, list per-impairment ratings."""

NAIVE_FINAL_AGGREGATOR_PROMPT = FINAL_AGGREGATOR_PROMPT + """

Output format: respond with a JSON object on the first line with these
fields:
- final_rating: one of \"0\", \"50\", \"100\", \"150\", \"175\", \
\"200\", \"250\", \"300\", \"decline\", \"postpone\"
- age_over_75: boolean
Example: {\"final_rating\": \"150\", \"age_over_75\": false}

After the JSON, give the rationale."""


def make_insurance_agents(
    extractor_prompt: str | None = None,
    rater_prompt: str | None = None,
    aggregator_prompt: str | None = None,
    separated: bool = True,
    demo_blocks: dict[str, str] | None = None,
    json_position: str = "begin",
) -> dict[str, Agent]:
    if separated:
        defaults = (
            MEDICAL_EXTRACTOR_PROMPT,
            IMPAIRMENT_RATER_PROMPT,
            FINAL_AGGREGATOR_PROMPT,
        )
    else:
        defaults = (
            NAIVE_MEDICAL_EXTRACTOR_PROMPT,
            NAIVE_IMPAIRMENT_RATER_PROMPT,
            NAIVE_FINAL_AGGREGATOR_PROMPT,
        )

    demo_blocks = demo_blocks or {}
    jp = json_position if separated else "begin"
    return {
        "medical_extractor": Agent(
            name="medical_extractor",
            control_schema=MedicalExtractorControl,
            system_prompt=extractor_prompt or defaults[0],
            separated=separated,
            demo_block=demo_blocks.get("medical_extractor", ""),
            json_position=jp,
        ),
        "impairment_rater": Agent(
            name="impairment_rater",
            control_schema=ImpairmentRaterControl,
            system_prompt=rater_prompt or defaults[1],
            separated=separated,
            demo_block=demo_blocks.get("impairment_rater", ""),
            json_position=jp,
        ),
        "final_aggregator": Agent(
            name="final_aggregator",
            control_schema=FinalAggregatorControl,
            system_prompt=aggregator_prompt or defaults[2],
            separated=separated,
            demo_block=demo_blocks.get("final_aggregator", ""),
            json_position=jp,
        ),
    }
