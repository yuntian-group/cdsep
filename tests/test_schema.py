"""Tests for the control schema engine."""

from dataclasses import dataclass
from typing import Literal, Optional

import pytest
from pydantic import BaseModel

from cdsep.schema import (
    generate_scaffolding,
    get_schema_fields,
    parse_response,
    validate_control,
)


# --- Test schemas ---


@dataclass
class LeaderControl:
    action: Literal["send", "stop"]
    target_agent: str
    stop: bool


@dataclass
class SyntheticControl:
    answer: int


@dataclass
class RiskControl:
    occupation: str
    location_risk: Literal["low", "medium", "high"]
    prior_claims: int
    lifestyle_risk: Literal["low", "medium", "high"]


class PydanticLeader(BaseModel):
    action: Literal["send", "stop"]
    target_agent: str
    stop: bool


@dataclass
class OptionalField:
    value: Optional[int]
    label: str


# --- Tests for get_schema_fields ---


class TestGetSchemaFields:
    def test_dataclass_fields(self):
        fields = get_schema_fields(LeaderControl)
        assert "action" in fields
        assert "target_agent" in fields
        assert "stop" in fields
        assert len(fields) == 3

    def test_pydantic_fields(self):
        fields = get_schema_fields(PydanticLeader)
        assert "action" in fields
        assert "target_agent" in fields
        assert "stop" in fields

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            get_schema_fields(dict)


# --- Tests for generate_scaffolding ---


class TestGenerateScaffolding:
    def test_contains_field_names(self):
        scaffolding = generate_scaffolding(LeaderControl)
        assert '"action"' in scaffolding
        assert '"target_agent"' in scaffolding
        assert '"stop"' in scaffolding

    def test_contains_literal_options(self):
        scaffolding = generate_scaffolding(LeaderControl)
        assert '"send"' in scaffolding
        assert '"stop"' in scaffolding

    def test_contains_json_example(self):
        scaffolding = generate_scaffolding(SyntheticControl)
        assert '"answer"' in scaffolding
        assert "integer" in scaffolding

    def test_pydantic_scaffolding(self):
        scaffolding = generate_scaffolding(PydanticLeader)
        assert '"action"' in scaffolding
        assert "JSON" in scaffolding


# --- Tests for parse_response ---


class TestParseResponse:
    def test_json_with_message(self):
        raw = '{"action": "send", "target_agent": "worker1", "stop": false}\nHere is my assignment for you.'
        control, msg = parse_response(raw)
        assert control == {"action": "send", "target_agent": "worker1", "stop": False}
        assert "assignment" in msg

    def test_fenced_json(self):
        raw = '```json\n{"answer": 42}\n```\nThe answer is 42 because...'
        control, msg = parse_response(raw)
        assert control == {"answer": 42}
        assert "42 because" in msg

    def test_no_json(self):
        raw = "I think the answer is seven."
        control, msg = parse_response(raw)
        assert control is None
        assert msg == raw

    def test_malformed_json(self):
        raw = '{action: send}\nsome message'
        control, msg = parse_response(raw)
        assert control is None

    def test_json_with_nested_braces(self):
        raw = '{"answer": 5, "note": "value is {complex}"}\nExtra text'
        control, msg = parse_response(raw)
        assert control is not None
        assert control["answer"] == 5

    def test_empty_response(self):
        control, msg = parse_response("")
        assert control is None
        assert msg == ""


# --- Tests for validate_control ---


class TestValidateControl:
    def test_valid_leader(self):
        d = {"action": "send", "target_agent": "worker1", "stop": False}
        instance, errors = validate_control(d, LeaderControl)
        assert errors == []
        assert instance is not None
        assert instance.action == "send"
        assert instance.target_agent == "worker1"
        assert instance.stop is False

    def test_missing_field(self):
        d = {"action": "send"}
        _, errors = validate_control(d, LeaderControl)
        assert len(errors) >= 1
        assert any("target_agent" in e for e in errors)

    def test_invalid_literal(self):
        d = {"action": "jump", "target_agent": "w1", "stop": False}
        _, errors = validate_control(d, LeaderControl)
        assert len(errors) == 1
        assert "action" in errors[0]

    def test_wrong_type_int(self):
        d = {"answer": "seven"}
        _, errors = validate_control(d, SyntheticControl)
        assert len(errors) == 1
        assert "integer" in errors[0]

    def test_wrong_type_bool(self):
        d = {"action": "send", "target_agent": "w1", "stop": "yes"}
        _, errors = validate_control(d, LeaderControl)
        assert any("boolean" in e for e in errors)

    def test_valid_pydantic(self):
        d = {"action": "stop", "target_agent": "none", "stop": True}
        instance, errors = validate_control(d, PydanticLeader)
        assert errors == []
        assert instance.stop is True

    def test_none_dict(self):
        _, errors = validate_control(None, LeaderControl)
        assert len(errors) == 1

    def test_optional_field_null(self):
        d = {"value": None, "label": "test"}
        instance, errors = validate_control(d, OptionalField)
        assert errors == []
        assert instance.value is None

    def test_extra_fields_ignored(self):
        d = {"answer": 5, "extra_field": "hello"}
        instance, errors = validate_control(d, SyntheticControl)
        assert errors == []
        assert instance.answer == 5

    def test_risk_schema(self):
        d = {
            "occupation": "office worker",
            "location_risk": "low",
            "prior_claims": 0,
            "lifestyle_risk": "medium",
        }
        instance, errors = validate_control(d, RiskControl)
        assert errors == []
        assert instance.occupation == "office worker"
        assert instance.prior_claims == 0
