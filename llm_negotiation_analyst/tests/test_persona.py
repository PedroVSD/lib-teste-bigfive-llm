"""
Testes para o módulo persona/ (atualizado para API positive/negative/none).
Execute com: pytest test_persona.py -v
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
            Big5Persona(openness=4)  # numérico não permitido
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

    def test_bloco_contem_dimensao_especificada(self):
        p = Big5Persona(agreeableness="positive")
        block = self.builder.build(p)
        assert "Agreeableness" in block
        assert "high level" in block
        assert "POSITIVE" in block

    def test_bloco_nao_contem_dimensao_nao_especificada(self):
        p = Big5Persona(agreeableness="positive")
        block = self.builder.build(p)
        assert "Openness" not in block
        assert "Neuroticism" not in block

    def test_negative_gera_polo_baixo(self):
        p = Big5Persona(agreeableness="negative")
        block = self.builder.build(p)
        assert "low level" in block
        assert "NEGATIVE" in block

    def test_positive_gera_polo_alto(self):
        p = Big5Persona(agreeableness="positive")
        block = self.builder.build(p)
        assert "high level" in block
        assert "POSITIVE" in block

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
        assert "Personality Profile" in result
        assert "Agreeableness" in result

    def test_header_e_footer_presentes(self):
        p = Big5Persona(openness="positive")
        block = self.builder.build(p)
        assert "--- Personality Profile ---" in block
        assert "---------------------------" in block

    def test_multiplas_dimensoes(self):
        p = Big5Persona(openness="positive", agreeableness="negative", neuroticism="positive")
        block = self.builder.build(p)
        assert "Openness" in block
        assert "Agreeableness" in block
        assert "Neuroticism" in block
        assert "Conscientiousness" not in block
        assert "Extraversion" not in block


class TestTacticsBuilder:

    def setup_method(self):
        self.builder = TacticsPromptBuilder()

    def test_tactics_1_a_5(self):
        for score in [1, 2, 3, 4, 5]:
            block = self.builder.build({"anchoring": score})
            assert "Firmeza" in block or "Anchoring" in block

    def test_tactics_3_usa_ancora_meio(self):
        block = self.builder.build({"anchoring": 3})
        # âncora 3 é moderada
        assert len(block) > 0
