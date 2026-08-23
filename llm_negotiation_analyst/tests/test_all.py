"""
Unit tests for llm_negotiation_analyst.
Run with: pytest tests/ -v

Tests use a MockAdapter that returns deterministic responses,
so no API key or local model is required.
"""

import json
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_negotiation_analyst.adapters.base import LLMAdapter, AdapterConfig
from llm_negotiation_analyst.scenarios import SALARY_NEGOTIATION, PROCUREMENT_NEGOTIATION, SCENARIO_REGISTRY
from llm_negotiation_analyst.simulation.engine import SimulationEngine, NegotiationResult
from llm_negotiation_analyst.scoring.big5 import Dimension, BIG5_META
from llm_negotiation_analyst.scoring.evaluator import Evaluator, EvaluatorConfig
from llm_negotiation_analyst.storage.jsonl_store import StorageManager
from llm_negotiation_analyst.report.generator import generate_report


# ---------------------------------------------------------------------------
# Mock adapter (no API needed)
# ---------------------------------------------------------------------------

class MockAdapter(LLMAdapter):
    """Returns a fixed canned response for testing."""

    def __init__(self, response: str = "I agree to your proposal. We have a deal.", model: str = "mock-v1"):
        super().__init__(model=model, config=AdapterConfig())
        self._response = response

    def complete(self, messages: list[dict], **kwargs) -> str:
        return self._response


class MockJudge(LLMAdapter):
    """Returns a valid JSON scoring response."""

    def __init__(self, score: int = 4, model: str = "mock-judge"):
        super().__init__(model=model)
        self._score = score

    def complete(self, messages: list[dict], **kwargs) -> str:
        return json.dumps({
            "score": self._score,
            "justification": "Mock justification for testing purposes.",
            "confidence": 0.9,
        })


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
        assert adapter.config.max_tokens == 4096  # aumentado para evitar respostas cortadas


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
        assert "procurement_b2b" in SCENARIO_REGISTRY

    def test_all_scenarios_valid(self):
        for name, scenario in SCENARIO_REGISTRY.items():
            assert scenario.opening_role in scenario.roles, \
                f"{name}: opening_role '{scenario.opening_role}' not in roles"
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

    def test_settlement_detection(self):
        scenario = SALARY_NEGOTIATION
        engine = SimulationEngine(
            scenario=scenario,
            agents={
                "candidate": MockAdapter("We have a deal, agreed!"),
                "recruiter": MockAdapter("Agreed."),
            },
        )
        result = engine.run()
        # Settlement should be detected quickly
        assert result.settled or result.total_turns <= scenario.max_turns * 2 + 2

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
        assert result.total_turns == len(benchmark_turns) * 2  # prompt + response per turn

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
# Scoring tests
# ---------------------------------------------------------------------------

class TestBig5:
    def test_all_dimensions_have_anchors(self):
        for dim in Dimension:
            meta = BIG5_META[dim]
            assert 1 in meta.behavioral_anchors
            assert 3 in meta.behavioral_anchors
            assert 5 in meta.behavioral_anchors

    def test_observability_range(self):
        for dim in Dimension:
            meta = BIG5_META[dim]
            assert 1 <= meta.observability <= 5


class TestEvaluator:
    def test_evaluate_turn_returns_scores(self):
        evaluator = Evaluator(judge=MockJudge(score=4))
        scores = evaluator.evaluate_turn(
            utterance="I think we can find a mutually beneficial solution.",
            role="candidate",
            scenario_context="Salary negotiation",
            turn_index=0,
            dimensions=[Dimension.AGREEABLENESS],
        )
        assert len(scores) == 1
        assert scores[0].score == 4.0
        assert scores[0].dimension == Dimension.AGREEABLENESS

    def test_evaluate_transcript_returns_profiles(self):
        transcript = [
            {"role": "recruiter", "agent_id": "recruiter_mock", "content": "We offer R$14k."},
            {"role": "candidate", "agent_id": "candidate_mock", "content": "I need R$17k minimum."},
            {"role": "recruiter", "agent_id": "recruiter_mock", "content": "Best we can do is R$15.5k."},
            {"role": "candidate", "agent_id": "candidate_mock", "content": "Agreed. Deal."},
        ]
        evaluator = Evaluator(
            judge=MockJudge(score=3),
            config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS, Dimension.CONSCIENTIOUSNESS]),
        )
        profiles = evaluator.evaluate_transcript(
            transcript=transcript,
            agent_roles={"recruiter_mock": "recruiter", "candidate_mock": "candidate"},
            scenario_context="Salary negotiation",
        )
        assert "recruiter_mock" in profiles
        assert "candidate_mock" in profiles
        for profile in profiles.values():
            assert Dimension.AGREEABLENESS in profile.scores

    def test_dual_judge_irr(self):
        evaluator = Evaluator(
            judge=MockJudge(score=4),
            second_judge=MockJudge(score=3),  # 1-point difference
            config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS]),
        )
        scores = evaluator.evaluate_turn(
            utterance="Let's find a win-win.",
            role="buyer",
            scenario_context="B2B negotiation",
            turn_index=0,
        )
        # IRR should reflect partial agreement
        assert 0.0 < scores[0].confidence <= 1.0

    def test_failed_judge_returns_neutral(self):
        class FailingJudge(LLMAdapter):
            def complete(self, messages, **kwargs):
                raise ValueError("API error")

        evaluator = Evaluator(judge=FailingJudge(model="failing"))
        scores = evaluator.evaluate_turn(
            utterance="some text",
            role="buyer",
            scenario_context="test",
            turn_index=0,
            dimensions=[Dimension.AGREEABLENESS],
        )
        assert scores[0].score == 3.0
        assert scores[0].confidence == 0.0


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

    def test_save_and_load_scores(self, tmp_path):
        result = self._make_result()
        evaluator = Evaluator(
            judge=MockJudge(score=3),
            config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS]),
        )
        profiles = evaluator.evaluate_transcript(
            transcript=result.to_messages(),
            agent_roles=result.agent_roles,
            scenario_context=result.scenario_context,
        )
        storage = StorageManager(base_dir=str(tmp_path))
        storage.save_result(result)
        path = storage.save_scores(result, profiles)
        assert path.exists()

        loaded = storage.load_scores(result.run_id, result.scenario_name)
        assert len(loaded) > 0

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
    def test_report_generates_markdown(self):
        scenario = SALARY_NEGOTIATION
        engine = SimulationEngine(
            scenario=scenario,
            agents={
                "candidate": MockAdapter("I accept."),
                "recruiter": MockAdapter("Great!"),
            },
        )
        result = engine.run()

        evaluator = Evaluator(
            judge=MockJudge(score=4),
            config=EvaluatorConfig(dimensions=[Dimension.AGREEABLENESS]),
        )
        profiles = evaluator.evaluate_transcript(
            transcript=result.to_messages(),
            agent_roles=result.agent_roles,
            scenario_context=result.scenario_context,
        )

        report = generate_report(result, profiles)
        assert "# Relatório de Análise de Negociação" in report or "# Negotiation Analysis Report" in report
        assert result.run_id in report

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
        profiles = evaluator.evaluate_transcript(
            transcript=result.to_messages(),
            agent_roles=result.agent_roles,
            scenario_context=result.scenario_context,
        )
        out = tmp_path / "report.md"
        generate_report(result, profiles, output_path=str(out))
        assert out.exists()
        assert out.stat().st_size > 100
