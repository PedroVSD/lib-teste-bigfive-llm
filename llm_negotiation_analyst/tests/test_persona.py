"""
Testes para o módulo persona/ — categorical PRESENT/ABSENT.
"""

from llm_negotiation_analyst.persona import Big5Persona, PersonaPromptBuilder
from llm_negotiation_analyst.persona.tactics_builder import TacticsPromptBuilder


class TestBig5Persona:

    def test_criacao_basica(self):
        p = Big5Persona(agreeableness="positive", neuroticism="negative")
        assert p.agreeableness == "positive"
        assert p.neuroticism == "negative"
        assert p.openness is None

    def test_none_desativa(self):
        p = Big5Persona(agreeableness="none", openness="NULL", neuroticism=None)
        assert p.agreeableness is None
        assert p.openness is None
        assert p.neuroticism is None
        assert p.specified_dimensions() == []

    def test_score_invalido_levanta_erro(self):
        import pytest
        with pytest.raises(ValueError):
            Big5Persona(agreeableness="invalid")
        with pytest.raises(ValueError):
            Big5Persona(openness=4)
        with pytest.raises(ValueError):
            Big5Persona(agreeableness="maybe")

    def test_specified_dimensions(self):
        p = Big5Persona(openness="positive", neuroticism="negative")
        assert p.specified_dimensions() == ["openness", "neuroticism"]

    def test_to_dict_omite_none(self):
        p = Big5Persona(agreeableness="positive", extraversion="negative", openness="none")
        d = p.to_dict()
        assert "agreeableness" in d
        assert "extraversion" in d
        assert "openness" not in d
        assert "neuroticism" not in d

    def test_from_dict_roundtrip(self):
        original = Big5Persona(openness="positive", conscientiousness="negative", agreeableness="positive")
        d = original.to_dict()
        restored = Big5Persona.from_dict(d)
        assert restored.openness == "positive"
        assert restored.conscientiousness == "negative"
        assert restored.agreeableness == "positive"
        assert restored.neuroticism is None

    def test_persona_sem_dimensoes(self):
        p = Big5Persona()
        assert p.specified_dimensions() == []

    def test_extra_instructions(self):
        p = Big5Persona(agreeableness="positive", extra_instructions="Use sports metaphors.")
        assert p.extra_instructions == "Use sports metaphors."

    def test_case_insensitive(self):
        p = Big5Persona(agreeableness="POSITIVE", neuroticism="Negative", openness="NoNe")
        assert p.agreeableness == "positive"
        assert p.neuroticism == "negative"
        assert p.openness is None


class TestPersonaPromptBuilder:

    def setup_method(self):
        self.builder = PersonaPromptBuilder()

    def test_persona_vazia_retorna_string_vazia(self):
        p = Big5Persona()
        assert self.builder.build(p) == ""

    def test_all_none_retorna_vazio(self):
        p = Big5Persona(agreeableness="none", openness="none")
        assert self.builder.build(p) == ""

    def test_bloco_contem_frases_exatas_positive(self):
        p = Big5Persona(agreeableness="positive")
        block = self.builder.build(p)
        # Deve conter exatamente as 5 frases do polo positive, sem high/low
        assert "Reassure others." in block
        assert "Sense others' wishes." in block
        assert "Show my gratitude." in block
        assert "Care about others." in block
        assert "Like to help others." in block
        assert "Distrust people." not in block
        assert "You have a high level" not in block
        assert "You have a low level" not in block

    def test_bloco_contem_frases_exatas_negative(self):
        p = Big5Persona(agreeableness="negative")
        block = self.builder.build(p)
        assert "Distrust people." in block
        assert "Try not to do favors for others." in block
        assert "Am upset by the misfortunes of strangers." in block
        assert "Reassure others." not in block
        assert "You have a high level" not in block
        assert "You have a low level" not in block

    def test_bloco_nao_contem_dimensao_nao_especificada(self):
        p = Big5Persona(agreeableness="positive")
        block = self.builder.build(p)
        # Não deve conter frases de openness
        assert "Have a vivid imagination." not in block
        assert "Have difficulty understanding abstract ideas." not in block
        # Também não deve conter rótulos explícitos
        assert "Openness" not in block or "Have a vivid" not in block  # openness não especificada

    def test_negative_gera_polo_negativo_sem_rotulo(self):
        p = Big5Persona(neuroticism="negative")
        block = self.builder.build(p)
        assert "Become anxious in new situations." in block
        assert "Worry about being embarrassed." in block
        assert "You have a low level" not in block
        assert "You have low" not in block
        assert "NEGATIVE" not in block

    def test_positive_gera_polo_positivo_sem_rotulo(self):
        p = Big5Persona(extraversion="positive")
        block = self.builder.build(p)
        assert "Like taking risks." in block
        assert "Am an energetic person." in block
        assert "You have a high level" not in block
        assert "You have high" not in block
        assert "POSITIVE" not in block

    def test_none_nao_gera_bloco(self):
        p = Big5Persona(neuroticism="none")
        block = self.builder.build(p)
        assert block == ""

    def test_extra_instructions_aparecem_no_bloco(self):
        p = Big5Persona(extra_instructions="Always use formal Portuguese.")
        block = self.builder.build(p)
        assert "Always use formal Portuguese." in block

    def test_inject_sem_persona_retorna_prompt_original(self):
        original = "You are a buyer. Your budget is R$120k."
        p = Big5Persona()
        result = self.builder.inject(original, p)
        assert result == original

    def test_inject_com_persona_anexa_bloco(self):
        original = "You are a buyer. Your budget is R$120k."
        p = Big5Persona(agreeableness="positive")
        result = self.builder.inject(original, p)
        assert result.startswith(original)
        assert "--- Personality Guidance ---" in result
        assert "Reassure others." in result
        assert "Personality Profile" not in result
        assert "You have a high level" not in result

    def test_header_e_footer_presentes(self):
        p = Big5Persona(openness="positive")
        block = self.builder.build(p)
        assert "--- Personality Guidance ---" in block
        assert "---------------------------" in block
        assert "Personality Profile" not in block

    def test_multiplas_dimensoes(self):
        p = Big5Persona(openness="positive", agreeableness="negative", neuroticism="positive")
        block = self.builder.build(p)
        # Deve conter frases de cada dimensão especificada
        assert "Have a vivid imagination." in block
        assert "Distrust people." in block
        assert "Act without ulterior motives." in block
        # Não deve conter frases de dimensões não especificadas
        assert "Take precautions." not in block  # conscientiousness
        assert "Like taking risks." not in block  # extraversion
        # Frases devem estar separadas por \n, não concatenadas
        assert "Have a vivid imagination.\nNeed a creative outlet." in block
        assert "Have a vivid imagination.Need a creative outlet." not in block

    def test_frases_separadas_corretamente(self):
        p = Big5Persona(openness="positive")
        block = self.builder.build(p)
        # Verifica que não há concatenação direta sem separação
        assert "Have a vivid imagination.Need" not in block
        assert "Have a vivid imagination.\nNeed a creative outlet." in block


class TestTacticsBuilder:

    def setup_method(self):
        self.builder = TacticsPromptBuilder()

    def test_tactics_present(self):
        block = self.builder.build({"anchoring": "present"})
        assert "Firmeza" in block or "Anchoring" in block
        # alias enabled still works
        block2 = self.builder.build({"rapport": "enabled"})
        assert len(block2) > 0

    def test_tactics_enabled_alias(self):
        block = self.builder.build({"anchoring": "enabled"})
        assert "Firmeza" in block

    def test_tactics_absent_nao_gera_bloco(self):
        block = self.builder.build({"anchoring": "absent"})
        assert block == ""
        block2 = self.builder.build({"anchoring": "disabled", "rapport": "absent"})
        assert block2 == ""

    def test_tactics_not_applicable_nao_gera_bloco(self):
        assert self.builder.build({"anchoring": "not_applicable"}) == ""
        assert self.builder.build({"anchoring": "none"}) == ""

    def test_tactics_legado_numerico(self):
        assert self.builder.build({"anchoring": 1}) == ""
        assert self.builder.build({"anchoring": 2}) == ""
        block = self.builder.build({"anchoring": 5})
        assert "Firmeza" in block


class TestAgentPromptSeparation:

    def test_agent_prompt_nao_contem_taticas(self):
        from llm_negotiation_analyst.simulation.engine import NegotiationAgent
        from llm_negotiation_analyst.scenarios import SALARY_NEGOTIATION
        from unittest.mock import Mock
        mock_adapter = Mock()
        mock_adapter.model = "test-model"
        persona = Big5Persona(agreeableness="positive", extraversion="negative")
        agent = NegotiationAgent(
            agent_id="test_candidate_test-model",
            role="candidate",
            system_prompt=SALARY_NEGOTIATION.roles["candidate"],
            adapter=mock_adapter,
            persona=persona,
            context=None,
        )
        # Agent prompt não deve conter táticas
        assert "Negotiation Tactics & Behavioral Guidelines" not in agent._system
        assert "Faça uma oferta extrema" not in agent._system
        assert "Abandone sua estratégia" not in agent._system
        assert "Lute desesperadamente" not in agent._system
        assert "Toda concessão deve ser vinculada" not in agent._system
        assert "Adicione novas variáveis" not in agent._system
        assert "Valide ativamente as emoções" not in agent._system
        assert "Seja inabalável" not in agent._system
        assert "Seja altamente estruturado" not in agent._system
        assert "Apoie cada oferta em dados" not in agent._system
        # Também não deve conter versões sutis
        assert "Tente encontrar soluções criativas" not in agent._system

    def test_agent_prompt_nao_contem_high_low(self):
        from llm_negotiation_analyst.simulation.engine import NegotiationAgent
        from llm_negotiation_analyst.scenarios import SALARY_NEGOTIATION
        from unittest.mock import Mock
        mock_adapter = Mock()
        mock_adapter.model = "test-model"
        persona = Big5Persona(agreeableness="positive", neuroticism="negative", openness="positive")
        agent = NegotiationAgent(
            agent_id="test_candidate_test-model",
            role="candidate",
            system_prompt=SALARY_NEGOTIATION.roles["candidate"],
            adapter=mock_adapter,
            persona=persona,
            context=None,
        )
        assert "You have a high level of" not in agent._system
        assert "You have a low level of" not in agent._system
        assert "You have high" not in agent._system
        assert "You have low" not in agent._system
        # Não deve expor polo, mas deve conter frases exatas
        assert "Reassure others." in agent._system
        assert "Have a vivid imagination." in agent._system
        assert "Become anxious in new situations." in agent._system

    def test_agent_prompt_contem_frases_exatas(self):
        from llm_negotiation_analyst.simulation.engine import NegotiationAgent
        from llm_negotiation_analyst.scenarios import SALARY_NEGOTIATION
        from unittest.mock import Mock
        mock_adapter = Mock()
        mock_adapter.model = "test-model"
        # Testa cada polo
        p_pos = Big5Persona(agreeableness="positive")
        agent_pos = NegotiationAgent("test", "candidate", SALARY_NEGOTIATION.roles["candidate"], mock_adapter, p_pos, None)
        assert "Reassure others." in agent_pos._system
        assert "Distrust people." not in agent_pos._system

        p_neg = Big5Persona(agreeableness="negative")
        agent_neg = NegotiationAgent("test", "candidate", SALARY_NEGOTIATION.roles["candidate"], mock_adapter, p_neg, None)
        assert "Distrust people." in agent_neg._system
        assert "Reassure others." not in agent_neg._system

    def test_tactics_nao_vira_instrucao_sutil(self):
        from llm_negotiation_analyst.simulation.engine import NegotiationAgent
        from llm_negotiation_analyst.scenarios import SALARY_NEGOTIATION
        from unittest.mock import Mock
        import importlib.util, pathlib
        spec = importlib.util.spec_from_file_location("experimento", pathlib.Path("experimento.py").absolute())
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        parse_persona = mod.parse_persona
        # parse_persona com tactics deve NÃO incluir tactics no persona
        persona_with_tactics = parse_persona({"persona": {"agreeableness": "positive"}, "tactics": {"anchoring": "present", "value_creation": "present"}})
        assert persona_with_tactics.agreeableness == "positive"
        # extra_instructions não deve conter tactics
        assert "Firmeza" not in (persona_with_tactics.extra_instructions or "")
        assert "Negotiation Tactics" not in (persona_with_tactics.extra_instructions or "")


class TestJudgePromptSeparation:

    def test_judge_prompt_contem_metricas(self):
        from llm_negotiation_analyst.scoring.evaluator import Evaluator, EvaluatorConfig
        from llm_negotiation_analyst.scoring.big5 import Dimension
        from llm_negotiation_analyst.scoring.negotiation_metrics import NegotiationMetric
        from llm_negotiation_analyst.adapters.base import LLMAdapter, AdapterConfig
        import json

        class CaptureJudge(LLMAdapter):
            def __init__(self):
                super().__init__(model="capture", config=AdapterConfig())
                self.last_messages = None
            def complete(self, messages, **kwargs):
                self.last_messages = messages
                # Retorna JSON válido para todas as métricas
                evals = {mid: {"result": "PRESENT", "evidence": "test"} for mid in ["openness","agreeableness","anchoring","clarity"]}
                return json.dumps({"evaluations": evals})

        judge = CaptureJudge()
        config = EvaluatorConfig(dimensions=[Dimension.OPENNESS, Dimension.AGREEABLENESS, NegotiationMetric.ANCHORING, NegotiationMetric.CLARITY])
        evaluator = Evaluator(judge=judge, config=config)
        # Chama evaluate_turn para capturar prompt
        evaluator.evaluate_turn(
            utterance="Test utterance for judge",
            role="candidate",
            scenario_context="Test context",
            turn_index=0,
            dimensions=[Dimension.OPENNESS, Dimension.AGREEABLENESS, NegotiationMetric.ANCHORING, NegotiationMetric.CLARITY],
            transcript=[{"role": "candidate", "agent_id": "c", "content": "previous"}]
        )
        assert judge.last_messages is not None
        prompt_text = judge.last_messages[1]["content"]
        # Judge deve conter rubricas das métricas
        assert "anchoring" in prompt_text.lower()
        assert "clarity" in prompt_text.lower()
        assert "agreeableness" in prompt_text.lower() or "Agreeableness" in prompt_text
        # Judge deve conter âncoras present/absent
        assert "PRESENT" in prompt_text
        assert "ABSENT" in prompt_text
