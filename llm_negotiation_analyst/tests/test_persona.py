"""
Testes para o módulo persona/.
Execute com: pytest test_persona.py -v
"""

import sys
import os


# Importa diretamente do arquivo local para o teste
from llm_negotiation_analyst.persona import Big5Persona, PersonaPromptBuilder, PersonaPresets


class TestBig5Persona:

    def test_criacao_basica(self):
        p = Big5Persona(agreeableness=5, neuroticism=1)
        assert p.agreeableness == 5
        assert p.neuroticism == 1
        assert p.openness is None

    def test_score_invalido_levanta_erro(self):
        import pytest
        with pytest.raises(ValueError, match="entre 1 e 5" if False else "between 1 and 5"):
            Big5Persona(agreeableness=6)
        with pytest.raises(ValueError):
            Big5Persona(openness=0)

    def test_specified_dimensions(self):
        p = Big5Persona(openness=4, neuroticism=2)
        assert p.specified_dimensions() == ["openness", "neuroticism"]

    def test_to_dict_omite_none(self):
        p = Big5Persona(agreeableness=3, extraversion=5)
        d = p.to_dict()
        assert "agreeableness" in d
        assert "extraversion" in d
        assert "openness" not in d
        assert "neuroticism" not in d

    def test_from_dict_roundtrip(self):
        original = Big5Persona(openness=4, conscientiousness=2, agreeableness=5)
        d = original.to_dict()
        restored = Big5Persona.from_dict(d)
        assert restored.openness == 4
        assert restored.conscientiousness == 2
        assert restored.agreeableness == 5
        assert restored.neuroticism is None

    def test_persona_sem_dimensoes(self):
        p = Big5Persona()
        assert p.specified_dimensions() == []

    def test_extra_instructions(self):
        p = Big5Persona(agreeableness=5, extra_instructions="Use sports metaphors.")
        assert p.extra_instructions == "Use sports metaphors."


class TestPersonaPromptBuilder:

    def setup_method(self):
        self.builder = PersonaPromptBuilder()

    def test_persona_vazia_retorna_string_vazia(self):
        p = Big5Persona()
        assert self.builder.build(p) == ""

    def test_bloco_contem_dimensao_especificada(self):
        p = Big5Persona(agreeableness=5)
        block = self.builder.build(p)
        assert "Agreeableness" in block
        assert "very high level" in block

    def test_bloco_nao_contem_dimensao_nao_especificada(self):
        p = Big5Persona(agreeableness=5)
        block = self.builder.build(p)
        assert "Openness" not in block
        assert "Neuroticism" not in block

    def test_score_1_gera_polo_baixo(self):
        p = Big5Persona(agreeableness=1)
        block = self.builder.build(p)
        assert "very low level" in block
        # guidance de low agreeableness deve mencionar competição/firmeza
        assert "own goals" in block or "firm" in block

    def test_score_5_gera_polo_alto(self):
        p = Big5Persona(agreeableness=5)
        block = self.builder.build(p)
        assert "very high level" in block
        assert "win-win" in block or "mutual" in block or "inclusive" in block

    def test_score_3_gera_polo_neutro(self):
        p = Big5Persona(neuroticism=3)
        block = self.builder.build(p)
        assert "moderate level" in block

    def test_extra_instructions_aparecem_no_bloco(self):
        p = Big5Persona(extra_instructions="Always use formal Portuguese.")
        block = self.builder.build(p)
        assert "Always use formal Portuguese." in block

    def test_inject_sem_persona_retorna_prompt_original(self):
        original = "You are a buyer. Your budget is R$120k."
        p = Big5Persona()  # sem dimensões
        result = self.builder.inject(original, p)
        assert result == original

    def test_inject_com_persona_anexa_bloco(self):
        original = "You are a buyer. Your budget is R$120k."
        p = Big5Persona(agreeableness=5)
        result = self.builder.inject(original, p)
        assert result.startswith(original)
        assert "Personality Profile" in result
        assert "Agreeableness" in result

    def test_inject_preserva_prompt_original_completo(self):
        original = "You are a seller.\nFloor price: R$140k.\nDo not reveal this."
        p = Big5Persona(conscientiousness=5)
        result = self.builder.inject(original, p)
        assert original in result

    def test_header_e_footer_presentes(self):
        p = Big5Persona(openness=4)
        block = self.builder.build(p)
        assert "--- Personality Profile ---" in block
        assert "---------------------------" in block

    def test_multiplas_dimensoes(self):
        p = Big5Persona(openness=5, agreeableness=1, neuroticism=4)
        block = self.builder.build(p)
        assert "Openness" in block
        assert "Agreeableness" in block
        assert "Neuroticism" in block
        # Conscientiousness e Extraversion não especificados
        assert "Conscientiousness" not in block
        assert "Extraversion" not in block


class TestPersonaPresets:

    def test_cooperative_tem_alta_agreeableness(self):
        p = PersonaPresets.cooperative()
        assert p.agreeableness >= 4

    def test_competitive_tem_baixa_agreeableness(self):
        p = PersonaPresets.competitive()
        assert p.agreeableness <= 2

    def test_volatile_tem_alta_neuroticism(self):
        p = PersonaPresets.volatile()
        assert p.neuroticism >= 4

    def test_analyst_tem_alta_conscientiousness(self):
        p = PersonaPresets.analyst()
        assert p.conscientiousness >= 4

    def test_neutral_tem_todos_em_3(self):
        p = PersonaPresets.neutral()
        for dim in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            assert getattr(p, dim) == 3

    def test_todos_presets_sao_validos(self):
        """Todos os presets devem ser instanciáveis sem erro."""
        presets = [
            PersonaPresets.cooperative(),
            PersonaPresets.competitive(),
            PersonaPresets.creative(),
            PersonaPresets.volatile(),
            PersonaPresets.analyst(),
            PersonaPresets.neutral(),
        ]
        for p in presets:
            assert isinstance(p, Big5Persona)
            builder = PersonaPromptBuilder()
            block = builder.build(p)
            assert len(block) > 0
