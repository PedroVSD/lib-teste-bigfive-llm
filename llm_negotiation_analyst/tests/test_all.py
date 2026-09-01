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
    """Returns categorical JSON {metric, result, evidence}."""
    def __init__(self, result: str = "PRESENT", model: str = "mock-judge"):
        super().__init__(model=model)
        self._result = result
    def complete(self, messages: list[dict], **kwargs) -> str:
        return json.dumps({
            "metric": "agreeableness",
            "result": self._result,
            "evidence": f"Mock evidence for {self._result}",
        })


class SequenceJudge(LLMAdapter):
    """Cycles through results list per call."""
    def __init__(self, results: list[str], model: str = "seq-judge"):
        super().__init__(model=model)
        self.results = results
        self.idx = 0
    def complete(self, messages: list[dict], **kwargs) -> str:
        r = self.results[self.idx % len(self.results)]
        self.idx += 1
        return json.dumps({"metric": "anchoring", "result": r, "evidence": f"evidence {r}"})


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
        assert "# Relatório de Análise de Negociação" in report
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
