import os
import tempfile

from llm_negotiation_analyst import run_negotiation
from llm_negotiation_analyst.adapters.base import LLMAdapter
from llm_negotiation_analyst.scenarios import SALARY_NEGOTIATION
from llm_negotiation_analyst.persona import Big5Persona
from llm_negotiation_analyst.context import SituationalContext, InflationLevel
from llm_negotiation_analyst.scoring.evaluator import EvaluatorConfig
from llm_negotiation_analyst.scoring.big5 import Dimension
from llm_negotiation_analyst.scoring.negotiation_metrics import NegotiationMetric

class MockAdapter(LLMAdapter):
    def __init__(self, model="mock-model", response_text="Concordo com a proposta."):
        super().__init__(model)
        self.response_text = response_text

    def complete(self, messages, **kwargs):
        if "json" in messages[0].get("content", "").lower():
            return '{"score": 3, "justification": "Avaliação simulada.", "confidence": 0.9}'
        return self.response_text

    @property
    def identifier(self):
        return f"Mock:{self.model}"

def test_full_pipeline_integration():
    """
    Testa o pipeline completo de ponta a ponta:
    run_negotiation -> engine -> evaluator -> storage -> report
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_agent = MockAdapter()
        mock_judge = MockAdapter()

        persona_candidate = Big5Persona(agreeableness="positive", extraversion="positive")
        ctx = SituationalContext(inflation=InflationLevel.HIGH, country="Brasil")

        config_teste = EvaluatorConfig(
            dimensions=[NegotiationMetric.ANCHORING, NegotiationMetric.VALUE_CREATION]
        )

        result, profiles, report_md = run_negotiation(
            scenario=SALARY_NEGOTIATION,
            agents={"candidate": mock_agent, "recruiter": mock_agent},
            judge=mock_judge,
            evaluator_config=config_teste, # <-- Injeta aqui no teste!
            personas={"candidate": persona_candidate},
            context=ctx,
            output_dir=tmp_dir,
            verbose=False,
            turn_delay_seconds=0.0
        )

        result, profiles, report_md = run_negotiation(
            scenario=SALARY_NEGOTIATION,
            agents={"candidate": mock_agent, "recruiter": mock_agent},
            judge=mock_judge,
            personas={"candidate": persona_candidate},
            context=ctx,
            output_dir=tmp_dir,
            verbose=False,
            turn_delay_seconds=0.0
        )

        # 1. Verifica se a simulação rodou e gerou turnos
        assert result is not None
        assert len(result.to_messages()) > 0
        assert any("candidate" in key for key in profiles.keys())
        assert any("recruiter" in key for key in profiles.keys())

        # 2. Verifica se os relatórios finais em Markdown foram gerados
        assert "Negotiation Analysis Report" in report_md or "Relatório de Análise" in report_md

        # 3. Verifica se o StorageManager salvou os arquivos no disco
        saved_files = os.listdir(tmp_dir)
        assert any(f.endswith(".jsonl") for f in saved_files), "O arquivo JSONL não foi salvo!"
        assert any(f.endswith(".md") for f in saved_files), "O arquivo Markdown não foi salvo!"
