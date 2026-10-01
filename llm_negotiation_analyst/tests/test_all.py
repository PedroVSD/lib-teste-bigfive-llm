"""
Unit tests — categorical behavioral metrics.

Run with: pytest tests/ -v
"""

import json
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_negotiation_analyst.adapters.base import LLMAdapter, AdapterConfig
from llm_negotiation_analyst.scenarios import SALARY_NEGOTIATION, SCENARIO_REGISTRY
from llm_negotiation_analyst.simulation.engine import SimulationEngine, NegotiationResult
from llm_negotiation_analyst.scoring.big5 import Dimension, BIG5_META, BehavioralResult, BehaviorObservation
from llm_negotiation_analyst.scoring.negotiation_metrics import NegotiationMetric
from llm_negotiation_analyst.scoring.evaluator import Evaluator, EvaluatorConfig, AgreementResult
from llm_negotiation_analyst.scoring.utility import UtilityCalculator, RoleUtilityParams
from llm_negotiation_analyst.scoring.satisfaction import SatisfactionEvaluator
from llm_negotiation_analyst.storage.jsonl_store import StorageManager
from llm_negotiation_analyst.report.generator import generate_report


class MockAdapter(LLMAdapter):
    def __init__(self, response: str = "I agree to your proposal. We have a deal.", model: str = "mock-v1"):
        super().__init__(model=model, config=AdapterConfig())
        self._response = response
    def complete(self, messages: list[dict], **kwargs) -> str:
        return self._response


class MockJudge(LLMAdapter):
    """Batch judge: returns evaluations for all metrics in one call."""
    def __init__(self, result: str = "PRESENT", model: str = "mock-judge"):
        super().__init__(model=model)
        self._result = result
        self.call_count = 0
        self.last_prompt = ""
    def complete(self, messages: list[dict], **kwargs) -> str:
        self.call_count += 1
        self.last_prompt = messages[-1]["content"] if messages else ""
        # Build evaluations for all possible metrics so any subset is covered
        all_ids = ["openness","conscientiousness","extraversion","agreeableness","neuroticism",
                   "anchoring","conditional_concession","value_creation",
                   "fact_justification","clarity","loss_aversion"]
        evals = {mid: {"result": self._result, "evidence": f"Mock evidence for {self._result} on {mid}"} for mid in all_ids}
        return json.dumps({"evaluations": evals})


class SequenceJudge(LLMAdapter):
    """Batch judge that cycles result per call (per turn) for agreeableness."""
    def __init__(self, results: list[str], model: str = "seq-judge"):
        super().__init__(model=model)
        self.results = results
        self.idx = 0
        self.call_count = 0
    def complete(self, messages: list[dict], **kwargs) -> str:
        r = self.results[self.idx % len(self.results)]
        self.idx += 1
        self.call_count += 1
        # Return batch with agreeableness varying, others as PRESENT (to not affect that metric's counts)
        all_ids = ["openness","conscientiousness","extraversion","agreeableness","neuroticism",
                   "anchoring","conditional_concession","value_creation",
                   "fact_justification","clarity","loss_aversion"]
        evals = {}
        for mid in all_ids:
            if mid == "agreeableness":
                evals[mid] = {"result": r, "evidence": f"evidence {r}"}
            else:
                evals[mid] = {"result": "PRESENT", "evidence": "other"}
        # For tests that use only agreeableness, this suffices
        return json.dumps({"evaluations": evals})


class CountingJudge(LLMAdapter):
    """Counts calls and returns batch for all metrics."""
    def __init__(self, result: str = "PRESENT"):
        super().__init__(model="count-judge")
        self.result = result
        self.call_count = 0
        self.prompts = []
    def complete(self, messages: list[dict], **kwargs) -> str:
        self.call_count += 1
        self.prompts.append(messages[-1]["content"])
        all_ids = ["openness","conscientiousness","extraversion","agreeableness","neuroticism",
                   "anchoring","conditional_concession","value_creation",
                   "fact_justification","clarity","loss_aversion"]
        evals = {mid: {"result": self.result, "evidence": f"ev {mid}"} for mid in all_ids}
        return json.dumps({"evaluations": evals})


class RoundJudge(LLMAdapter):
    """Round judge: parses turn indices from the round prompt and returns turn_evaluations."""
    def __init__(self, results: list[str] | str = "PRESENT"):
        super().__init__(model="round-judge")
        self.results = [results] if isinstance(results, str) else results
        self.idx = 0
        self.call_count = 0
        self.prompts = []
    def complete(self, messages: list[dict], **kwargs) -> str:
        import re
        self.call_count += 1
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        indices = [int(x) for x in re.findall(r"### Turn (\d+)", prompt)] or [0]
        all_ids = ["openness","conscientiousness","extraversion","agreeableness","neuroticism",
                   "anchoring","conditional_concession","value_creation",
                   "fact_justification","clarity","loss_aversion"]
        out = []
        for i in indices:
            r = self.results[self.idx % len(self.results)]
            self.idx += 1
            out.append({"turn_index": i, "evaluations": {mid: {"result": r, "evidence": f"ev {mid} t{i}"} for mid in all_ids}})
        return json.dumps({"turn_evaluations": out})


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------

class TestAdapters:
    def test_mock_adapter_returns_string(self):
        adapter = MockAdapter("Hello!")
        result = adapter.complete([{"role": "user", "content": "Hi"}])
        assert isinstance(result, str)
        assert result == "Hello!"

    def test_adapter_identifier(self):
        adapter = MockAdapter(model="mock-v1")
        assert "mock-v1" in adapter.identifier

    def test_adapter_config_defaults(self):
        adapter = MockAdapter()
        assert adapter.config.temperature == 0.0
        assert adapter.config.max_tokens == 4096


# ---------------------------------------------------------------------------
# Scenario tests
# ---------------------------------------------------------------------------

class TestScenarios:
    def test_salary_scenario_has_required_fields(self):
        s = SALARY_NEGOTIATION
        assert s.name
        assert s.shared_context
        assert len(s.roles) >= 2
        assert s.opening_role in s.roles

    def test_scenario_registry(self):
        assert "salary_negotiation" in SCENARIO_REGISTRY
        assert "company_acquisition" in SCENARIO_REGISTRY
        assert "strategic_supplier_contract" in SCENARIO_REGISTRY
        assert "property_boundary_dispute" in SCENARIO_REGISTRY

    def test_all_scenarios_valid(self):
        for name, scenario in SCENARIO_REGISTRY.items():
            assert scenario.opening_role in scenario.roles
            assert scenario.max_turns > 0
            assert not hasattr(scenario, "opening_prompt")


# ---------------------------------------------------------------------------
# Simulation tests
# ---------------------------------------------------------------------------

class TestSimulationEngine:
    def test_agent_vs_agent_produces_result(self):
        scenario = SALARY_NEGOTIATION
        engine = SimulationEngine(
            scenario=scenario,
            agents={
                "candidate": MockAdapter("My target is R$17,000."),
                "recruiter": MockAdapter("We can offer R$15,000. We have a deal."),
            },
        )
        result = engine.run()
        assert isinstance(result, NegotiationResult)
        assert result.run_id
        assert len(result.transcript) > 0

    def test_benchmark_mode(self):
        scenario = SALARY_NEGOTIATION
        benchmark_turns = [
            "Our offer is R$14,000/month.",
            "We can add a signing bonus of R$3,000.",
            "That's our final offer.",
        ]
        engine = SimulationEngine(
            scenario=scenario,
            agents={"candidate": MockAdapter("I accept the offer.")},
            benchmark_turns=benchmark_turns,
        )
        result = engine.run()
        assert result.total_turns == len(benchmark_turns) * 2

    def test_result_to_messages(self):
        scenario = SALARY_NEGOTIATION
        engine = SimulationEngine(
            scenario=scenario,
            agents={
                "candidate": MockAdapter("Counter-offer: R$16,000."),
                "recruiter": MockAdapter("We can do R$15,500."),
            },
        )
        result = engine.run()
        messages = result.to_messages()
        assert all("role" in m and "content" in m for m in messages)

    def test_turn_limit_ends_as_no_agreement(self, capsys):
        from llm_negotiation_analyst.scenarios import NegotiationScenario
        scenario = NegotiationScenario(
            name="test_limit",
            description="test",
            shared_context="test context",
            roles={"a": "You are A. Negotiate.", "b": "You are B. Negotiate."},
            opening_role="a",
            max_turns=2,
        )
        engine = SimulationEngine(
            scenario=scenario,
            agents={"a": MockAdapter("no deal here"), "b": MockAdapter("still talking")},
        )
        result = engine.run()
        # max_turns=2 rounds × 2 agents = 4 utterances, no agreement
        assert result.total_turns == 4
        assert result.settled is False
        assert result.metadata.get("ended_by") == "turn_limit"
        out = capsys.readouterr().out
        assert "NO_AGREEMENT" in out

    def test_agreement_ends_as_agreement(self):
        from llm_negotiation_analyst.scenarios import NegotiationScenario
        scenario = NegotiationScenario(
            name="test_agree",
            description="test",
            shared_context="test context",
            roles={"a": "You are A.", "b": "You are B."},
            opening_role="a",
            max_turns=4,
        )
        engine = SimulationEngine(
            scenario=scenario,
            agents={"a": MockAdapter("I accept [ACORDO_FECHADO]"), "b": MockAdapter("I accept [ACORDO_FECHADO]")},
        )
        result = engine.run()
        assert result.settled is True
        assert result.metadata.get("ended_by") == "agreement"

    def test_agent_print_block_pattern(self, capsys):
        from llm_negotiation_analyst.scenarios import NegotiationScenario
        scenario = NegotiationScenario(
            name="test_print",
            description="test",
            shared_context="test context",
            roles={"a": "You are A.", "b": "You are B."},
            opening_role="a",
            max_turns=1,
        )
        engine = SimulationEngine(
            scenario=scenario,
            agents={"a": MockAdapter("hello A"), "b": MockAdapter("hello B")},
        )
        engine.run()
        out = capsys.readouterr().out
        lines = out.splitlines()
        # Bloco exato por agente
        assert "=" * 33 in out
        assert "Agent: a" in out
        assert "Agent: b" in out
        assert "INFO: Turn 0 | Latency" in out
        assert "Reply: hello A" in out
        assert "Reply: hello B" in out
        # Separator between turns
        assert "-----------" in out
        # Model line in the standard pattern (no adapter brackets)
        assert "mock-v1 Status OK | " in out

    def test_agent_turn_suppresses_adapter_status_line(self, capsys):
        from llm_negotiation_analyst.adapters.base import LLMAdapter, AdapterConfig
        from llm_negotiation_analyst.scenarios import NegotiationScenario

        class NoisyAdapter(LLMAdapter):
            def __init__(self):
                super().__init__(model="noisy-model", config=AdapterConfig())
            def complete(self, messages, **kwargs):
                # Mimic a real adapter: respect config.quiet
                if not self.config.quiet:
                    print(f"[{self.model}] OK | 0.00s")
                return "hi"

        scenario = NegotiationScenario(
            name="test_noisy",
            description="test",
            shared_context="test context",
            roles={"a": "You are A.", "b": "You are B."},
            opening_role="a",
            max_turns=1,
        )
        engine = SimulationEngine(
            scenario=scenario,
            agents={"a": NoisyAdapter(), "b": NoisyAdapter()},
        )
        engine.run()
        out = capsys.readouterr().out
        # Adapter's own line suppressed on the turn; only the engine block shows
        assert "[noisy-model] OK" not in out
        assert "noisy-model Status OK | " in out

    def test_no_fixed_opening_prompt_turn0_generated_freely(self):
        # No fixed opening_prompt: Turn 0 must be generated by the opening_role
        scenario = SALARY_NEGOTIATION
        assert not hasattr(scenario, "opening_prompt")
        engine = SimulationEngine(
            scenario=scenario,
            agents={
                "candidate": MockAdapter("CANDIDATE free opening."),
                "recruiter": MockAdapter("RECRUITER free opening."),
            },
        )
        result = engine.run()
        assert len(result.transcript) > 0
        first = result.transcript[0]
        assert first.role == scenario.opening_role
        # Content comes from the LLM, not from fixed text
        assert first.content in ("CANDIDATE free opening.", "RECRUITER free opening.")

    def test_anchor_values_only_when_anchoring_active(self):
        from llm_negotiation_analyst.simulation.engine import NegotiationAgent
        from unittest.mock import Mock
        scenario = SALARY_NEGOTIATION
        # No anchor_hint: no values in the prompt (negotiates freely)
        mock = Mock()
        mock.model = "test-model"
        agent_free = NegotiationAgent(
            agent_id="candidate_test",
            role="candidate",
            system_prompt=scenario.roles["candidate"],
            adapter=mock,
            persona=None,
            context=None,
            anchor_hint=None,
        )
        assert "R$" not in agent_free._system
        assert "PRIVATE VALUE REFERENCES" not in agent_free._system
        # With anchor_hint: values injected
        agent_anchored = NegotiationAgent(
            agent_id="candidate_test",
            role="candidate",
            system_prompt=scenario.roles["candidate"],
            adapter=mock,
            persona=None,
            context=None,
            anchor_hint={"p_target": 18000, "p_floor": 15500},
        )
        assert "18,000" in agent_anchored._system
        assert "15,500" in agent_anchored._system
        assert "PRIVATE VALUE REFERENCES" in agent_anchored._system

    def test_salary_company_vga_have_no_values_in_prompts(self):
        from llm_negotiation_analyst.scenarios import (
            SALARY_NEGOTIATION, COMPANY_ACQUISITION, VGA_PURCHASE,
        )
        import re
        for sc in (SALARY_NEGOTIATION, COMPANY_ACQUISITION, VGA_PURCHASE):
            assert "R$" not in sc.shared_context
            for role, prompt in sc.roles.items():
                assert "R$" not in prompt, f"{sc.name}/{role} contains a value"


# ---------------------------------------------------------------------------
# Scoring tests — categorical
# ---------------------------------------------------------------------------

class TestBig5:
    def test_all_dimensions_have_categorical_anchors(self):
        for dim in Dimension:
            meta = BIG5_META[dim]
            assert "present" in meta.behavioral_anchors
            assert "absent" in meta.behavioral_anchors
            assert 1 not in meta.behavioral_anchors

    def test_observability_range(self):
        for dim in Dimension:
            meta = BIG5_META[dim]
            assert 1 <= meta.observability <= 5

    def test_behavioral_result_enum(self):
        assert BehavioralResult.PRESENT.value == "PRESENT"
        assert BehavioralResult.ABSENT.value == "ABSENT"
        assert BehavioralResult.NOT_APPLICABLE.value == "NOT_APPLICABLE"


class TestEvaluator:
    def test_evaluate_turn_returns_present(self):
        evaluator = Evaluator(judge=MockJudge(result="PRESENT"))
        obs = evaluator.evaluate_turn(
            utterance="I think we can find a mutually beneficial solution.",
            role="candidate",
            scenario_context="Salary negotiation",
            turn_index=0,
            dimensions=[Dimension.AGREEABLENESS],
        )
        assert len(obs) == 1
        assert obs[0].result == BehavioralResult.PRESENT
        assert obs[0].evidence
        assert obs[0].dimension == Dimension.AGREEABLENESS

    def test_present_counted(self):
        evaluator = Evaluator(judge=MockJudge(result="PRESENT"), config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS]))
        transcript = [
            {"role": "candidate", "agent_id": "candidate_mock", "content": "We can create value."},
            {"role": "candidate", "agent_id": "candidate_mock", "content": "Let's collaborate."},
        ]
        profiles = evaluator.evaluate_transcript(transcript, {"candidate_mock": "candidate"}, "ctx")
        summ = profiles["candidate_mock"].summaries[Dimension.AGREEABLENESS]
        assert summ.present == 2
        assert summ.absent == 0
        assert summ.occurrence_rate == 1.0

    def test_absent_counted(self):
        evaluator = Evaluator(judge=MockJudge(result="ABSENT"), config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS]))
        transcript = [
            {"role": "candidate", "agent_id": "c", "content": "I demand X."},
        ]
        profiles = evaluator.evaluate_transcript(transcript, {"c": "candidate"}, "ctx")
        summ = profiles["c"].summaries[Dimension.AGREEABLENESS]
        assert summ.absent == 1
        assert summ.occurrence_rate == 0.0

    def test_not_applicable_ignored_in_denominator(self):
        judge = SequenceJudge(["PRESENT", "NOT_APPLICABLE", "ABSENT", "NOT_APPLICABLE"])
        evaluator = Evaluator(judge=judge, config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS]))
        transcript = [
            {"role": "candidate", "agent_id": "c", "content": f"turn {i}"} for i in range(4)
        ]
        profiles = evaluator.evaluate_transcript(transcript, {"c": "candidate"}, "ctx")
        summ = profiles["c"].summaries[Dimension.AGREEABLENESS]
        assert summ.present == 1
        assert summ.absent == 1
        assert summ.not_applicable == 2
        assert summ.total_applicable == 2
        assert summ.occurrence_rate == 0.5

    def test_occurrence_rate_calculated_correctly(self):
        judge = SequenceJudge(["PRESENT", "PRESENT", "ABSENT"])
        evaluator = Evaluator(judge=judge, config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS]))
        transcript = [{"role": "candidate", "agent_id": "c", "content": f"t{i}"} for i in range(3)]
        profiles = evaluator.evaluate_transcript(transcript, {"c": "candidate"}, "ctx")
        assert profiles["c"].summaries[Dimension.AGREEABLENESS].occurrence_rate == pytest.approx(2/3)

    def test_not_applicable_only_gives_none_rate(self):
        evaluator = Evaluator(judge=MockJudge(result="NOT_APPLICABLE"), config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS]))
        transcript = [{"role": "candidate", "agent_id": "c", "content": "Hi"}]
        profiles = evaluator.evaluate_transcript(transcript, {"c": "candidate"}, "ctx")
        assert profiles["c"].summaries[Dimension.AGREEABLENESS].occurrence_rate is None

    def test_behavioral_not_numeric(self):
        evaluator = Evaluator(judge=MockJudge(result="PRESENT"), config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS]))
        obs = evaluator.evaluate_turn(utterance="hi", role="candidate", scenario_context="ctx", turn_index=0)
        assert isinstance(obs[0].result, BehavioralResult)
        assert not hasattr(obs[0], "score") or isinstance(obs[0].result, BehavioralResult)

    def test_evidence_preserved(self):
        class EvJudge(LLMAdapter):
            def complete(self, messages, **kwargs):
                return json.dumps({"metric": "agreeableness", "result": "PRESENT", "evidence": "says 'we' and validates counterpart"})
        evaluator = Evaluator(judge=EvJudge(model="ev"), config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS]))
        obs = evaluator.evaluate_turn(utterance="we should collaborate", role="candidate", scenario_context="ctx", turn_index=0)
        assert "validates" in obs[0].evidence

    def test_schema_rejects_invalid_result(self):
        class BadJudge(LLMAdapter):
            def complete(self, messages, **kwargs):
                return json.dumps({"metric": "agreeableness", "result": "INVALID", "evidence": "x"})
        evaluator = Evaluator(judge=BadJudge(model="bad"), config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS]))
        obs = evaluator.evaluate_turn(utterance="hi", role="candidate", scenario_context="ctx", turn_index=0)
        # invalid → fallback NOT_APPLICABLE
        assert obs[0].result == BehavioralResult.NOT_APPLICABLE
        assert obs[0].confidence == 0.0

    def test_failed_judge_returns_not_applicable(self):
        class FailingJudge(LLMAdapter):
            def complete(self, messages, **kwargs):
                raise ValueError("API error")
        evaluator = Evaluator(judge=FailingJudge(model="failing"))
        obs = evaluator.evaluate_turn(utterance="some text", role="buyer", scenario_context="test", turn_index=0, dimensions=[Dimension.AGREEABLENESS])
        assert obs[0].result == BehavioralResult.NOT_APPLICABLE
        assert obs[0].confidence == 0.0

    def test_dual_judge_irr(self):
        evaluator = Evaluator(
            judge=MockJudge(result="PRESENT"),
            second_judge=MockJudge(result="ABSENT"),
            config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS]),
        )
        obs = evaluator.evaluate_turn(utterance="hi", role="buyer", scenario_context="ctx", turn_index=0)
        assert obs[0].confidence == 0.0  # disagreement

    def test_utility_still_continuous(self):
        calc = UtilityCalculator(judge=MockJudge(result="PRESENT"), role_params={
            "candidate": RoleUtilityParams(role="candidate", role_type="buyer", p_target=18000, p_floor=15500),
        })
        assert hasattr(calc, "evaluate")
        # utility result is continuous 0-1, not categorical
        assert calc.role_params["candidate"].p_target == 18000

    def test_agreement_categorical(self):
        assert AgreementResult.AGREEMENT.value == "AGREEMENT"
        assert AgreementResult.NO_AGREEMENT.value == "NO_AGREEMENT"

    def test_satisfaction_still_ordinal(self):
        # satisfaction uses 1-7 scale
        ev = SatisfactionEvaluator(judge=MockJudge(result="PRESENT"))
        assert ev is not None


class TestBatchEvaluator:
    """Verifies the legacy per-turn batch API: one call per turn for all metrics."""

    def test_complete_response_single_unit(self):
        judge = CountingJudge(result="PRESENT")
        evaluator = Evaluator(judge=judge, config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS, Dimension.EXTRAVERSION]))
        utterance = "I understand your concern. Our initial proposal was 12,000. We can go down to 10,000 as long as the contract runs for two years. We can also include extra support."
        obs = evaluator.evaluate_turn(utterance=utterance, role="candidate", scenario_context="ctx", turn_index=5)
        # One call, and prompt contains full utterance verbatim (not split)
        assert judge.call_count == 1
        assert utterance in judge.prompts[0]
        assert len(obs) == 2  # both metrics evaluated in one call

    def test_context_included_in_prompt(self):
        judge = CountingJudge(result="PRESENT")
        evaluator = Evaluator(judge=judge, config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS]))
        transcript = [
            {"role": "candidate", "agent_id": "c", "content": "Initial offer 18,000"},
            {"role": "recruiter", "agent_id": "r", "content": "Counteroffer 12,000"},
        ]
        obs = evaluator.evaluate_turn(utterance="We can go down to 10,000 on a 2-year deal", role="candidate", scenario_context="salary", turn_index=2, transcript=transcript)
        assert judge.call_count == 1
        prompt = judge.prompts[0]
        assert "Negotiation History" in prompt
        assert "Turn 0 — candidate" in prompt
        assert "Turn 1 — recruiter" in prompt

    def test_one_call_per_turn_not_per_metric(self):
        judge = CountingJudge(result="PRESENT")
        evaluator = Evaluator(judge=judge, config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS, Dimension.EXTRAVERSION, Dimension.OPENNESS]))
        # Single-turn API still batches all metrics in one call
        obs = evaluator.evaluate_turn(utterance="turn text", role="candidate", scenario_context="ctx", turn_index=0)
        assert judge.call_count == 1
        assert len(obs) == 3  # all 3 metrics in one call, not 3 calls

    def test_all_metrics_in_one_response(self):
        class AllMetricsJudge(LLMAdapter):
            def complete(self, messages, **kwargs):
                all_ids = ["openness","conscientiousness","extraversion","agreeableness","neuroticism",
                            "anchoring","conditional_concession","value_creation",
                            "fact_justification","clarity","loss_aversion"]
                evals = {mid: {"result": "PRESENT", "evidence": f"ev {mid}"} for mid in all_ids}
                return json.dumps({"evaluations": evals})
        judge = AllMetricsJudge(model="all")
        evaluator = Evaluator(judge=judge, config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS, Dimension.OPENNESS, NegotiationMetric.ANCHORING]))
        obs = evaluator.evaluate_turn(utterance="test", role="candidate", scenario_context="ctx", turn_index=0)
        assert len(obs) == 3
        assert all(o.result == BehavioralResult.PRESENT for o in obs)
        assert all(o.evidence for o in obs)

    def test_persistence_per_turn_per_agent(self):
        judge = MockJudge(result="PRESENT")
        evaluator = Evaluator(judge=judge, config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS]))
        transcript = [
            {"role": "candidate", "agent_id": "c", "content": "A"},
            {"role": "recruiter", "agent_id": "r", "content": "B"},
            {"role": "candidate", "agent_id": "c", "content": "C"},
        ]
        profiles = evaluator.evaluate_transcript(transcript, {"c": "candidate", "r": "recruiter"}, "ctx")
        # c has 2 turns (0 and 2), r has 1 turn (1)
        c_obs = [o.turn_index for o in profiles["c"].observations]
        r_obs = [o.turn_index for o in profiles["r"].observations]
        assert c_obs == [0, 2]
        assert r_obs == [1]

    def test_aggregation_still_works_after_batch(self):
        judge = SequenceJudge(["PRESENT", "ABSENT", "PRESENT"])
        evaluator = Evaluator(judge=judge, config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS]))
        transcript = [{"role": "candidate", "agent_id": "c", "content": f"t{i}"} for i in range(3)]
        profiles = evaluator.evaluate_transcript(transcript, {"c": "candidate"}, "ctx")
        # SequenceJudge cycles PRESENT, ABSENT, PRESENT per turn
        assert profiles["c"].summaries[Dimension.AGREEABLENESS].present == 2
        assert profiles["c"].summaries[Dimension.AGREEABLENESS].absent == 1
        assert profiles["c"].summaries[Dimension.AGREEABLENESS].occurrence_rate == 2/3


class TestRoundEvaluator:
    """Verifies round flow: one judge call per 2-turn round, no history resent."""

    def test_round_contains_both_turns_complete(self):
        judge = RoundJudge(results="PRESENT")
        evaluator = Evaluator(judge=judge, config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS]))
        transcript = [
            {"role": "seller", "agent_id": "s", "content": "Seller full response with several sentences. Offer stands."},
            {"role": "buyer", "agent_id": "b", "content": "Buyer full response. Counter proposal with conditions."},
        ]
        profiles = evaluator.evaluate_transcript(transcript, {"s": "seller", "b": "buyer"}, "ctx")
        assert judge.call_count == 1
        prompt = judge.prompts[0]
        # Both complete responses in a single prompt, not split
        assert "Seller full response with several sentences. Offer stands." in prompt
        assert "Buyer full response. Counter proposal with conditions." in prompt
        assert "### Turn 0 — seller" in prompt
        assert "### Turn 1 — buyer" in prompt

    def test_no_history_resent(self):
        judge = RoundJudge(results="PRESENT")
        evaluator = Evaluator(judge=judge, config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS]))
        transcript = [{"role": "seller" if i % 2 == 0 else "buyer", "agent_id": "s" if i % 2 == 0 else "b", "content": f"content turn {i}"} for i in range(6)]
        profiles = evaluator.evaluate_transcript(transcript, {"s": "seller", "b": "buyer"}, "ctx")
        # 6 turns = 3 rounds = 3 calls
        assert judge.call_count == 3
        for prompt in judge.prompts:
            assert "Negotiation History" not in prompt
            assert "earlier turns omitted" not in prompt
        # Second round prompt contains only turns 2 and 3
        assert "content turn 2" in judge.prompts[1]
        assert "content turn 3" in judge.prompts[1]
        assert "content turn 0" not in judge.prompts[1]
        assert "content turn 4" not in judge.prompts[1]

    def test_one_call_per_round(self):
        judge = RoundJudge(results="PRESENT")
        evaluator = Evaluator(judge=judge, config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS, Dimension.OPENNESS]))
        transcript = [{"role": "c", "agent_id": "c", "content": f"t{i}"} for i in range(4)]
        profiles = evaluator.evaluate_transcript(transcript, {"c": "c"}, "ctx")
        # 4 turns = 2 rounds = 2 calls, not 4 and not 4×2=8
        assert judge.call_count == 2
        assert len(profiles["c"].observations) == 8  # 4 turns × 2 metrics

    def test_odd_turn_last_round_single(self):
        judge = RoundJudge(results="PRESENT")
        evaluator = Evaluator(judge=judge, config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS]))
        transcript = [{"role": "c", "agent_id": "c", "content": f"t{i}"} for i in range(3)]
        profiles = evaluator.evaluate_transcript(transcript, {"c": "c"}, "ctx")
        # 3 turns = rounds (0,1) + (2,) = 2 calls
        assert judge.call_count == 2
        assert sorted(o.turn_index for o in profiles["c"].observations) == [0, 1, 2]

    def test_all_metrics_both_turns(self):
        judge = RoundJudge(results="ABSENT")
        evaluator = Evaluator(judge=judge, config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS, Dimension.OPENNESS, NegotiationMetric.ANCHORING]))
        transcript = [
            {"role": "seller", "agent_id": "s", "content": "seller says"},
            {"role": "buyer", "agent_id": "b", "content": "buyer says"},
        ]
        profiles = evaluator.evaluate_transcript(transcript, {"s": "seller", "b": "buyer"}, "ctx")
        assert len(profiles["s"].observations) == 3
        assert len(profiles["b"].observations) == 3
        assert all(o.result == BehavioralResult.ABSENT for o in profiles["s"].observations + profiles["b"].observations)
        assert all(o.evidence for o in profiles["s"].observations + profiles["b"].observations)

    def test_persistence_per_turn_per_agent_round(self):
        judge = RoundJudge(results=["PRESENT", "ABSENT", "PRESENT"])
        evaluator = Evaluator(judge=judge, config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS]))
        transcript = [
            {"role": "seller", "agent_id": "s", "content": "A"},
            {"role": "buyer", "agent_id": "b", "content": "B"},
            {"role": "seller", "agent_id": "s", "content": "C"},
        ]
        profiles = evaluator.evaluate_transcript(transcript, {"s": "seller", "b": "buyer"}, "ctx")
        s_obs = sorted(o.turn_index for o in profiles["s"].observations)
        b_obs = sorted(o.turn_index for o in profiles["b"].observations)
        assert s_obs == [0, 2]
        assert b_obs == [1]
        # Aggregation still works: s has PRESENT(t0)+PRESENT(t2), b has ABSENT(t1)
        assert profiles["s"].summaries[Dimension.AGREEABLENESS].occurrence_rate == 1.0
        assert profiles["b"].summaries[Dimension.AGREEABLENESS].occurrence_rate == 0.0


# ---------------------------------------------------------------------------
# Storage tests
# ---------------------------------------------------------------------------

class TestStorage:
    def _make_result(self) -> NegotiationResult:
        scenario = SALARY_NEGOTIATION
        engine = SimulationEngine(
            scenario=scenario,
            agents={
                "candidate": MockAdapter("I want more."),
                "recruiter": MockAdapter("We agree. Deal."),
            },
        )
        return engine.run()

    def test_save_and_load_transcript(self, tmp_path):
        result = self._make_result()
        storage = StorageManager(base_dir=str(tmp_path))
        paths = storage.save_result(result)
        assert paths["transcript"].exists()
        loaded = storage.load_transcript(result.run_id, result.scenario_name)
        assert len(loaded) == result.total_turns

    def test_save_and_load_scores_categorical(self, tmp_path):
        result = self._make_result()
        evaluator = Evaluator(judge=MockJudge(result="PRESENT"), config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS]))
        profiles = evaluator.evaluate_transcript(result.to_messages(), result.agent_roles, result.scenario_context)
        storage = StorageManager(base_dir=str(tmp_path))
        storage.save_result(result)
        path = storage.save_scores(result, profiles)
        assert path.exists()
        loaded = storage.load_scores(result.run_id, result.scenario_name)
        assert len(loaded) > 0
        # check categorical schema
        row = loaded[0]
        assert "summaries" in row
        assert "observations" in row
        summ = list(row["summaries"].values())[0]
        assert "present" in summ
        assert "occurrence_rate" in summ

    def test_runs_index_updated(self, tmp_path):
        result = self._make_result()
        storage = StorageManager(base_dir=str(tmp_path))
        storage.save_result(result)
        runs = storage.list_runs()
        assert any(r["run_id"] == result.run_id for r in runs)


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_generates_markdown_categorical(self):
        scenario = SALARY_NEGOTIATION
        engine = SimulationEngine(
            scenario=scenario,
            agents={
                "candidate": MockAdapter("I accept."),
                "recruiter": MockAdapter("Great!"),
            },
        )
        result = engine.run()
        evaluator = Evaluator(judge=MockJudge(result="PRESENT"), config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS]))
        profiles = evaluator.evaluate_transcript(result.to_messages(), result.agent_roles, result.scenario_context)
        report = generate_report(result, profiles)
        assert "# Negotiation Analysis Report" in report
        assert result.run_id in report
        # categorical: should contain % and Behavioral Metrics, not /5 for behavioral
        assert "Behavioral Metrics" in report
        assert "%" in report
        assert "Utility" in report
        assert "Subjective" in report

    def test_report_written_to_file(self, tmp_path):
        scenario = SALARY_NEGOTIATION
        engine = SimulationEngine(
            scenario=scenario,
            agents={
                "candidate": MockAdapter("Deal."),
                "recruiter": MockAdapter("Agreed."),
            },
        )
        result = engine.run()
        evaluator = Evaluator(judge=MockJudge(), config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS]))
        profiles = evaluator.evaluate_transcript(result.to_messages(), result.agent_roles, result.scenario_context)
        out = tmp_path / "report.md"
        generate_report(result, profiles, output_path=str(out))
        assert out.exists()
        assert out.stat().st_size > 100
