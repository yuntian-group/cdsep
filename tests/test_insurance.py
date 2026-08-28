"""Smoke tests for the synthetic underwriting experiment pipeline."""

from experiments.insurance.agents import (
    FinalAggregatorControl,
    ImpairmentRaterControl,
    MedicalExtractorControl,
    make_insurance_agents,
)
from experiments.insurance.data import (
    RATING_BUCKETS,
    compute_ground_truth,
    generate_applicant,
    generate_dataset,
)


class TestInsuranceData:
    def test_generate_applicant(self):
        import random
        rng = random.Random(42)
        app = generate_applicant(rng)
        assert "description" in app
        assert app["ground_truth"] in RATING_BUCKETS

    def test_ground_truth_standard(self):
        rating, _ = compute_ground_truth([("Hypertension", "mild")], [], age=40)
        assert rating == "0"

    def test_ground_truth_decline(self):
        rating, _ = compute_ground_truth(
            [("Coronary Artery Disease", "severe")], [], age=40,
        )
        assert rating == "decline"

    def test_generate_dataset(self):
        train, val, test = generate_dataset(30, seed=0)
        assert len(train) == 20
        assert len(val) == 0
        assert len(test) == 10


class TestInsurancePipeline:
    def test_agents_created(self):
        agents = make_insurance_agents()
        assert "medical_extractor" in agents
        assert "impairment_rater" in agents
        assert "final_aggregator" in agents

    def test_routing_with_mock(self):
        from experiments.insurance.run import route_insurance

        ctrl_extractor = MedicalExtractorControl(
            primary_chapter="Diabetes", n_impairments=1,
        )
        assert route_insurance(ctrl_extractor) == "impairment_rater"

        ctrl_rater = ImpairmentRaterControl(
            worst_impairment_rating="50", all_impairments_standard=False,
        )
        assert route_insurance(ctrl_rater) == "final_aggregator"

        ctrl_final = FinalAggregatorControl(final_rating="50", age_over_75=False)
        assert route_insurance(ctrl_final) == "terminate"
