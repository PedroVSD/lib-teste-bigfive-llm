import os
import tempfile
import json

from llm_negotiation_analyst import run_negotiation
from llm_negotiation_analyst.adapters.base import LLMAdapter
from llm_negotiation_analyst.scenarios import SALARY_NEGOTIATION
from llm_negotiation_analyst.persona import Big5Persona
from llm_negotiation_analyst.context import SituationalContext, InflationLevel
from llm_negotiation_analyst.scoring.evaluator import EvaluatorConfig
from llm_negotiation_analyst.scoring.big5 import Dimension
from llm_negotiation_analyst.scoring.negotiation_metrics import NegotiationMetric

class MockAdapter(LLMAdapter):
    def __init__(self, model="mock-model", response_text="I agree with the proposal."):
        super().__init__(model)
        self.response_text = response_text

    def complete(self, messages, **kwargs):
        # judge prompts contain "turn_evaluations"/"evaluations"; agents get negotiation text
        content = messages[0].get("content", "").lower()
        if "turn_evaluations" in content or "evaluations" in content or "result" in content:
            import re
            # Round prompt: one entry per turn found in the prompt
            full = messages[-1].get("content", "")
            indices = [int(x) for x in re.findall(r"### Turn (\d+)", full)] or [0]
            return json.dumps({
                "turn_evaluations": [
                    {"turn_index": i, "evaluations": {
                        "anchoring": {"result": "PRESENT", "evidence": "Mock anchoring evaluation."},
                        "value_creation": {"result": "PRESENT", "evidence": "Mock value evaluation."}
                    }} for i in indices
                ]
            })
        return self.response_text

    @property
    def identifier(self):
        return f"Mock:{self.model}"

def test_full_pipeline_integration():
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_agent = MockAdapter()
        mock_judge = MockAdapter()

        persona_candidate = Big5Persona(agreeableness="positive", extraversion="positive")
        ctx = SituationalContext(inflation=InflationLevel.HIGH)

        config_teste = EvaluatorConfig(
            dimensions=[NegotiationMetric.ANCHORING, NegotiationMetric.VALUE_CREATION]
        )

        result, profiles, report_md = run_negotiation(
            scenario=SALARY_NEGOTIATION,
            agents={"candidate": mock_agent, "recruiter": mock_agent},
            judge=mock_judge,
            evaluator_config=config_teste,
            personas={"candidate": persona_candidate},
            context=ctx,
            output_dir=tmp_dir,
            verbose=False,
            turn_delay_seconds=0.0
        )

        assert result is not None
        assert len(result.to_messages()) > 0
        assert any("candidate" in key for key in profiles.keys())
        assert any("recruiter" in key for key in profiles.keys())

        # verify categorical occurrence_rate
        for p in profiles.values():
            for summ in p.summaries.values():
                assert summ.occurrence_rate is None or 0.0 <= summ.occurrence_rate <= 1.0
                assert summ.present + summ.absent == summ.total_applicable

        assert "Negotiation Analysis Report" in report_md

        saved_files = os.listdir(tmp_dir)
        assert any(f.endswith(".jsonl") for f in saved_files)
        assert any(f.endswith(".md") for f in saved_files)
