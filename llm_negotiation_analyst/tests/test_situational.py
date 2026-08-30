"""
Testes para context/situational.py
Execute com: pytest test_situational.py -v
"""

import sys, os


import pytest
from llm_negotiation_analyst.context import (
    SituationalContext, ContextPromptBuilder, ContextPresets,
    InflationLevel, InterestRateLevel, GovernmentOrientation, CrisisType,
)


class TestSituationalContext:

    def test_criacao_vazia(self):
        ctx = SituationalContext()
        assert ctx.enabled is True
        assert ctx.inflation is None
        assert ctx.crises == []

    def test_is_active_false_sem_condicoes(self):
        ctx = SituationalContext()
        assert ctx.is_active() is False

    def test_is_active_false_quando_disabled(self):
        ctx = SituationalContext(
            enabled=False,
            inflation=InflationLevel.HIGH,
        )
        assert ctx.is_active() is False

    def test_is_active_true_com_inflacao(self):
        ctx = SituationalContext(inflation=InflationLevel.HIGH)
        assert ctx.is_active() is True

    def test_is_active_true_com_crise(self):
        ctx = SituationalContext(crises=[CrisisType.ECONOMIC_RECESSION])
        assert ctx.is_active() is True

    def test_is_active_true_com_custom(self):
        ctx = SituationalContext(custom_conditions=["Some condition."])
        assert ctx.is_active() is True

    def test_disable_retorna_copia(self):
        ctx = SituationalContext(inflation=InflationLevel.HIGH)
        disabled = ctx.disable()
        assert disabled.enabled is False
        assert ctx.enabled is True  # original não muda
        assert disabled.inflation == InflationLevel.HIGH

    def test_disabled_classmethod(self):
        ctx = SituationalContext.disabled()
        assert ctx.enabled is False
        assert ctx.is_active() is False

    def test_to_dict_serializa_enums(self):
        ctx = SituationalContext(
            inflation=InflationLevel.HIGH,
            crises=[CrisisType.ECONOMIC_RECESSION],
        )
        d = ctx.to_dict()
        assert d["inflation"] == "high"
        assert d["crises"] == ["economic_recession"]
        assert d["enabled"] is True

    def test_to_dict_none_vira_none(self):
        ctx = SituationalContext(inflation=None)
        d = ctx.to_dict()
        assert d["inflation"] is None

    def test_from_dict_roundtrip(self):
        ctx = SituationalContext(
            inflation=InflationLevel.VERY_HIGH,
            interest_rates=InterestRateLevel.HIGH,
            government=GovernmentOrientation.POPULIST,
            crises=[CrisisType.CURRENCY_CRISIS, CrisisType.POLITICAL_INSTABILITY],
            custom_conditions=["Some custom note."],
        )
        restored = SituationalContext.from_dict(ctx.to_dict())
        assert restored.inflation == InflationLevel.VERY_HIGH
        assert restored.interest_rates == InterestRateLevel.HIGH
        assert restored.government == GovernmentOrientation.POPULIST
        assert CrisisType.CURRENCY_CRISIS in restored.crises
        assert CrisisType.POLITICAL_INSTABILITY in restored.crises
        assert "Some custom note." in restored.custom_conditions

    def test_multiplas_crises(self):
        ctx = SituationalContext(crises=[
            CrisisType.ECONOMIC_RECESSION,
            CrisisType.ENERGY_CRISIS,
            CrisisType.GEOPOLITICAL,
        ])
        assert len(ctx.crises) == 3
        assert ctx.is_active() is True


class TestContextPromptBuilder:

    def setup_method(self):
        self.builder = ContextPromptBuilder()

    def test_contexto_inativo_retorna_vazio(self):
        ctx = SituationalContext()  # sem condições
        assert self.builder.build(ctx) == ""

    def test_contexto_disabled_retorna_vazio(self):
        ctx = SituationalContext(inflation=InflationLevel.HIGH, enabled=False)
        assert self.builder.build(ctx) == ""

    def test_bloco_contem_inflacao(self):
        ctx = SituationalContext(inflation=InflationLevel.VERY_HIGH)
        block = self.builder.build(ctx)
        assert "Macroeconomic environment" in block
        assert "very high" in block.lower() or "above 10%" in block

    def test_bloco_contem_juros(self):
        ctx = SituationalContext(interest_rates=InterestRateLevel.HIGH)
        block = self.builder.build(ctx)
        assert "Interest rates" in block
        assert "expensive" in block or "high" in block.lower()

    def test_bloco_contem_governo(self):
        ctx = SituationalContext(government=GovernmentOrientation.POPULIST)
        block = self.builder.build(ctx)
        assert "Political" in block
        assert "populist" in block.lower() or "unpredictable" in block.lower()

    def test_bloco_contem_crise(self):
        ctx = SituationalContext(crises=[CrisisType.ECONOMIC_RECESSION])
        block = self.builder.build(ctx)
        assert "Active crisis" in block
        assert "recession" in block.lower()

    def test_multiplas_crises_aparecem(self):
        ctx = SituationalContext(crises=[
            CrisisType.ECONOMIC_RECESSION,
            CrisisType.ENERGY_CRISIS,
        ])
        block = self.builder.build(ctx)
        assert block.count("Active crisis") == 2

    def test_custom_conditions_aparecem(self):
        ctx = SituationalContext(custom_conditions=[
            "The company just went through a merger.",
            "There is a hiring freeze in effect.",
        ])
        block = self.builder.build(ctx)
        assert "The company just went through a merger." in block
        assert "There is a hiring freeze in effect." in block

    def test_grounding_sem_pais_e_ano(self):
        ctx = SituationalContext(inflation=InflationLevel.LOW)
        block = self.builder.build(ctx)
        assert "external conditions" in block

    def test_header_e_footer_presentes(self):
        ctx = SituationalContext(inflation=InflationLevel.MODERATE)
        block = self.builder.build(ctx)
        assert "--- Economic & Situational Context ---" in block
        assert "--------------------------------------" in block

    def test_inject_sem_contexto_retorna_original(self):
        original = "You are a buyer."
        ctx = SituationalContext()
        result = self.builder.inject(original, ctx)
        assert result == original

    def test_inject_disabled_retorna_original(self):
        original = "You are a buyer."
        ctx = SituationalContext(inflation=InflationLevel.HIGH, enabled=False)
        result = self.builder.inject(original, ctx)
        assert result == original

    def test_inject_com_contexto_anexa_bloco(self):
        original = "You are a buyer."
        ctx = SituationalContext(inflation=InflationLevel.HIGH)
        result = self.builder.inject(original, ctx)
        assert result.startswith(original)
        assert "Economic & Situational Context" in result

    def test_inject_preserva_prompt_original(self):
        original = "You are a seller.\nFloor price: R$140k.\nPrivate info."
        ctx = SituationalContext(crises=[CrisisType.FINANCIAL_CRISIS])
        result = self.builder.inject(original, ctx)
        assert original in result

    def test_gdp_e_unemployment_aparecem(self):
        ctx = SituationalContext(
            gdp_growth="GDP contracted 3.5%.",
            unemployment="Unemployment at 12%.",
        )
        block = self.builder.build(ctx)
        assert "GDP contracted 3.5%." in block
        assert "Unemployment at 12%." in block


class TestContextPresets:

    def test_brazil_2015_e_ativo(self):
        ctx = ContextPresets.brazil_2015_crisis()
        assert ctx.is_active()
        assert CrisisType.ECONOMIC_RECESSION in ctx.crises
        assert CrisisType.POLITICAL_INSTABILITY in ctx.crises

    def test_stable_growth_sem_crises(self):
        ctx = ContextPresets.stable_growth()
        assert ctx.crises == []
        assert ctx.inflation == InflationLevel.LOW

    def test_financial_crisis_e_ativo(self):
        ctx = ContextPresets.financial_crisis()
        assert ctx.is_active()
        assert CrisisType.FINANCIAL_CRISIS in ctx.crises

    def test_disabled_preset_nao_ativo(self):
        ctx = ContextPresets.disabled()
        assert ctx.is_active() is False

    def test_todos_presets_geram_bloco(self):
        builder = ContextPromptBuilder()
        presets = [
            ContextPresets.brazil_2015_crisis(),
            ContextPresets.post_pandemic_recovery(),
            ContextPresets.stable_growth(),
            ContextPresets.financial_crisis(),
            ContextPresets.geopolitical_tension(),
        ]
        for preset in presets:
            block = builder.build(preset)
            assert len(block) > 0, f"Preset {preset} gerou bloco vazio"

    def test_disabled_preset_gera_bloco_vazio(self):
        builder = ContextPromptBuilder()
        block = builder.build(ContextPresets.disabled())
        assert block == ""
