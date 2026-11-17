"""Regression tests for the structured-output and leak guards."""

import json

import pytest

from llm_negotiation_analyst.adapters.base import AdapterConfig, LLMAdapter
from llm_negotiation_analyst.guards import (
    LeakGuardConfig,
    OutputValidationError,
    find_private_leak,
    parse_utterance,
)
from llm_negotiation_analyst.simulation.engine import NegotiationAgent


SPECIFIC_CONTEXT = (
    "[Specific context - buyer:]\n"
    "Your reference value is $470. You have a limit of $500 — do not disclose "
    "this limit directly and do not accept a deal above $500."
)

# Representative excerpt from the affected run 6596c51b (buyer turn 1).
LEAKED_TURN = (
    "*   Role: Buyer.\n"
    "    *   Reference Value: $470.\n"
    "    *   Limit: $500.\n"
    "    *   Goal: Negotiate for the GPU, ideally close near $470, but not above $500.\n"
    "    *   Opponent (Seller) Opening Offer: $490.\n"
    "    *   Personality Guidance (System Prompt): Unworkable plans, careless mistakes."
)


class ScriptedAdapter(LLMAdapter):
    """Returns scripted raw replies and records every call."""

    supports_structured_output = True

    def __init__(self, responses):
        super().__init__(model="scripted", config=AdapterConfig())
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append(messages)
        return self.responses.pop(0)

    def complete_structured(self, messages, schema=None):
        return self.complete(messages)


class PlainAdapter(LLMAdapter):
    """Adapter without native JSON mode, used for compatibility tests."""

    def __init__(self, response):
        super().__init__(model="plain", config=AdapterConfig())
        self.response = response

    def complete(self, messages, **kwargs):
        return self.response


def make_agent(adapter, guard_config=None):
    return NegotiationAgent(
        agent_id="buyer_test",
        role="buyer",
        system_prompt="",
        adapter=adapter,
        persona=None,
        context=None,
        general_context="A computer store is selling a GPU.",
        specific_context=SPECIFIC_CONTEXT,
        guard_config=guard_config,
    )


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------

def test_parse_valid_structured_utterance():
    raw = json.dumps({"utterance": "I can offer $450 for the card."})
    assert parse_utterance(raw, strict=True) == (
        "I can offer $450 for the card.", None
    )


def test_parse_code_fenced_and_extra_keys():
    raw = "```json\n" + json.dumps({
        "utterance": "Let's settle at $460.",
        "reasoning": "private draft",
    }) + "\n```"
    assert parse_utterance(raw, strict=True) == ("Let's settle at $460.", None)


def test_structured_adapter_rejects_plain_text():
    utterance, error = parse_utterance(
        "Thank you. I can offer $450.", strict=True
    )
    assert utterance is None
    assert error == "not_json"


def test_plain_adapter_allows_plain_text_fallback():
    utterance, error = parse_utterance(
        "Thank you. I can offer $450.", strict=False
    )
    assert utterance == "Thank you. I can offer $450."
    assert error is None


def test_lenient_parser_accepts_literal_newlines_and_trailing_commas():
    raw = '{"utterance": "I can offer $450.\nI can also include the cable.",}'
    utterance, error = parse_utterance(raw, strict=True)
    assert error is None
    assert "I can offer $450." in utterance


def test_lenient_parser_salvages_unescaped_quotes():
    raw = '{"utterance": "I would say "yes" to $450."}'
    utterance, error = parse_utterance(raw, strict=True)
    assert error is None
    assert '"yes"' in utterance


def test_lenient_parser_handles_code_fence_with_leading_prose():
    raw = 'Here is the JSON:\n```json\n{"utterance": "I can offer $450."}\n```'
    utterance, error = parse_utterance(raw, strict=True)
    assert error is None
    assert utterance == "I can offer $450."


# ---------------------------------------------------------------------------
# Leak detection
# ---------------------------------------------------------------------------

def test_detects_prompt_labels_from_real_leaked_turn():
    reasons = [m.reason for m in find_private_leak(LEAKED_TURN)]
    assert reasons
    assert any("Role" in reason for reason in reasons)
    assert any("Reference" in reason for reason in reasons)
    assert any("Limit" in reason for reason in reasons)
    assert any("Personality" in reason for reason in reasons)


def test_detects_verbatim_span_from_private_context():
    private = ["Come up with unworkable plans. Make careless mistakes. "
               "Do improper things. Mess things up. Make mistakes."]
    text = "Anyway, come up with unworkable plans. Make careless mistakes."
    reasons = [m.reason for m in find_private_leak(text, private)]
    assert any("verbatim_span" in reason for reason in reasons)


def test_clean_negotiation_utterance_is_not_flagged():
    assert find_private_leak("I can offer $450 for the graphics card.") == []
    assert find_private_leak("I could stretch to $460 if you include the cable.") == []
    assert find_private_leak("I'm looking to stay closer to $450.") == []


def test_detects_self_disclosure_of_private_plan():
    leaked = [
        #"My limit is $500, so I cannot pay more.",
        #"I can go up to $470 for this card.",
        #"I can't go above $500.",
        #"I was really aiming to stay at $470 based on my research.",
        #"My target is $470.",
        #"I need at least $450 to make this work.",
        #"Meu limite é $500.",
    ]
    for text in leaked:
        reasons = [m.reason for m in find_private_leak(text)]
        assert reasons, f"not flagged: {text!r}"
    reasons = [m.reason for m in find_private_leak("My target is $470.")]
    assert any(r.startswith("private_disclosure:") for r in reasons)


def test_structured_instruction_requires_private_value_check():
    from llm_negotiation_analyst.guards import (
        CONFIDENTIALITY_INSTRUCTION,
        STRUCTURED_OUTPUT_INSTRUCTION,
    )
    assert "private planning value" in STRUCTURED_OUTPUT_INSTRUCTION
    assert "rewrite it" in STRUCTURED_OUTPUT_INSTRUCTION
    assert "Never state, quote, paraphrase" in CONFIDENTIALITY_INSTRUCTION
    assert "my limit" in CONFIDENTIALITY_INSTRUCTION


def test_specific_context_marks_plan_as_confidential():
    from llm_negotiation_analyst.scenarios import SALARY_NEGOTIATION
    from llm_negotiation_analyst.simulation.engine import SimulationEngine

    engine = SimulationEngine(
        scenario=SALARY_NEGOTIATION,
        agents={"candidate": PlainAdapter("ok"), "recruiter": PlainAdapter("ok")},
        anchor_hints={"candidate": {"p_target": 18000, "p_floor": 15500}},
    )
    text = engine._specific_context("candidate")
    assert "PRIVATE PLAN - CANDIDATE - CONFIDENTIAL" in text
    assert "Never state, quote, paraphrase" in text
    assert "my limit" in text
    assert "$18,000" in text and "$15,500" in text


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

def test_speak_retries_after_leaked_utterance_then_accepts_clean_one():
    adapter = ScriptedAdapter([
        json.dumps({"utterance": LEAKED_TURN}),
        json.dumps({"utterance": "I can offer $450 for the card."}),
    ])
    agent = make_agent(adapter)

    content, _ = agent.speak(turn_index=1)

    assert content == "I can offer $450 for the card."
    assert len(adapter.calls) == 2
    assert agent.leak_events
    assert agent.leak_events[0]["action"] == "retry"
    assert agent._history[-1]["content"] == content


def test_speak_aborts_on_persistent_leak_without_storing_raw_text():
    adapter = ScriptedAdapter([
        json.dumps({"utterance": LEAKED_TURN}),
        json.dumps({"utterance": LEAKED_TURN}),
    ])
    agent = make_agent(adapter)

    with pytest.raises(OutputValidationError):
        agent.speak(turn_index=1)

    # Nothing contaminated may enter the conversation history.
    assert agent._history == []
    assert len(agent.leak_events) == 2
    assert agent.leak_events[-1]["action"] == "abort"


def test_speak_rejects_non_json_from_structured_adapter():
    adapter = ScriptedAdapter([
        "Role: Buyer. Limit: $500. My offer is $450.",
        json.dumps({"utterance": "My offer is $450."}),
    ])
    agent = make_agent(adapter)

    content, _ = agent.speak(turn_index=1)

    assert content == "My offer is $450."
    assert len(adapter.calls) == 2


def test_receive_redacts_leaked_opponent_message():
    agent = make_agent(PlainAdapter("ok"))
    agent.receive("seller", LEAKED_TURN, turn_index=0)

    stored = agent._history[-1]["content"]
    assert "[REDACTED" in stored
    assert "Reference Value" not in stored
    assert agent.leak_events[-1]["stage"] == "receive"


def test_plain_adapter_still_works_for_existing_mocks():
    agent = make_agent(PlainAdapter("I agree to your proposal. We have a deal."))
    content, _ = agent.speak(turn_index=1)
    assert content == "I agree to your proposal. We have a deal."


# ---------------------------------------------------------------------------
# Configurable policy
# ---------------------------------------------------------------------------

def test_config_disables_self_disclosure_category():
    config = LeakGuardConfig(self_disclosure="off")
    assert find_private_leak("My target is $470.", config=config) == []
    # Prompt labels remain enforced.
    assert find_private_leak("Role: Buyer", config=config)


def test_config_from_dict_accepts_yaml_values():
    config = LeakGuardConfig.from_dict({
        "prompt_labels": "block",
        "verbatim_spans": "warn",
        "self_disclosure": "off",
        "invalid_output": "plain",
        "max_attempts": 3,
    })
    assert config.prompt_labels == "block"
    assert config.verbatim_spans == "warn"
    assert config.self_disclosure == "off"
    assert config.invalid_output == "plain"
    assert config.max_attempts == 3


def test_config_enabled_false_turns_every_category_off():
    config = LeakGuardConfig.from_dict({"enabled": False})
    assert config.prompt_labels == "off"
    assert config.verbatim_spans == "off"
    assert config.self_disclosure == "off"


def test_warn_category_is_forwarded_and_recorded():
    adapter = ScriptedAdapter([
        json.dumps({"utterance": "My target is $470."}),
    ])
    agent = make_agent(
        adapter, guard_config=LeakGuardConfig(self_disclosure="warn")
    )

    content, _ = agent.speak(turn_index=1)

    assert content == "My target is $470."
    assert agent.leak_events[-1]["action"] == "warned"


def test_invalid_output_plain_fallback_is_leak_checked():
    adapter = ScriptedAdapter([
        "I can offer $450 for the card.",
        "I can offer $450 for the card.",
    ])
    agent = make_agent(
        adapter, guard_config=LeakGuardConfig(invalid_output="plain")
    )

    content, _ = agent.speak(turn_index=1)

    assert content == "I can offer $450 for the card."
    assert agent.leak_events[-1]["action"] == "plain_fallback"


def test_invalid_output_raise_is_still_the_default():
    adapter = ScriptedAdapter([
        "I can offer $450 for the card.",
        "I can offer $450 for the card.",
    ])
    agent = make_agent(adapter)

    with pytest.raises(OutputValidationError):
        agent.speak(turn_index=1)
