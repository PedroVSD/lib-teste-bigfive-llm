import os
import yaml
from dotenv import load_dotenv

from llm_negotiation_analyst import run_negotiation
from llm_negotiation_analyst.adapters.deepseek_adapter import DeepSeekAdapter
from llm_negotiation_analyst.adapters.gemini_adapter import GeminiAdapter
from llm_negotiation_analyst.adapters.openai_adapter import OpenAIAdapter
from llm_negotiation_analyst.adapters.lmstudio_adapter import LMStudioAdapter
from llm_negotiation_analyst.adapters.ollama_adapter import OllamaAdapter
from llm_negotiation_analyst.adapters.ollama_local_adapter import OllamaLocalAdapter
from llm_negotiation_analyst.scenarios import SCENARIO_REGISTRY
from llm_negotiation_analyst.adapters.base import AdapterConfig
from llm_negotiation_analyst.scoring import EvaluatorConfig
from llm_negotiation_analyst.persona import Big5Persona, TacticsPromptBuilder
from llm_negotiation_analyst.context import (
    SituationalContext, InflationLevel, InterestRateLevel,
    GovernmentOrientation, CrisisType,
)
from llm_negotiation_analyst.scoring.evaluator import EvaluatorConfig
from llm_negotiation_analyst.scoring.utility import UtilityCalculator, RoleUtilityParams
from llm_negotiation_analyst.scoring.satisfaction import SatisfactionEvaluator
from llm_negotiation_analyst.report.generator import generate_report

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Funções auxiliares
# ─────────────────────────────────────────────────────────────────────────────

def load_config(filepath: str):
    with open(filepath, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_adapter(config_dict: dict):
    provider   = config_dict["provider"].lower()
    model_name = config_dict["name"]
    temp       = config_dict.get("temperature", 0.5)
    max_tokens = config_dict.get("max_tokens", 1024)
    config_obj = AdapterConfig(temperature=temp, max_tokens=max_tokens)

    if provider == "gemini":
        return GeminiAdapter(model=model_name, config=config_obj)
    elif provider == "openai":
        return OpenAIAdapter(model=model_name, config=config_obj)
    elif provider == "lmstudio":
        return LMStudioAdapter(model=model_name, config=config_obj)
    elif provider == "ollama_local":
        return OllamaLocalAdapter(model=model_name, config=config_obj)
    elif provider == "ollama":
        return OllamaAdapter(model=model_name, base_url=config_dict.get("base_url"), config=config_obj)
    elif provider == "deepseek":
        return DeepSeekAdapter(model=model_name, base_url=config_dict.get("base_url"), config=config_obj)
    raise ValueError(f"Provedor desconhecido: {provider}")


def parse_persona(agent_config: dict) -> Big5Persona | None:
    persona_dict = agent_config.get("persona") or {}
    tactics_dict = agent_config.get("tactics") or {}

    if not persona_dict and not tactics_dict:
        return None

    chaves_big5 = {"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"}
    filtered_persona = {k: v for k, v in persona_dict.items() if k in chaves_big5 and v is not None}

    instrucoes_originais = persona_dict.get("extra_instructions", "")
    builder      = TacticsPromptBuilder()
    texto_taticas = builder.build(tactics_dict)
    texto_final  = f"{instrucoes_originais}\n\n{texto_taticas}".strip() if texto_taticas else instrucoes_originais

    if texto_final:
        filtered_persona["extra_instructions"] = texto_final

    return Big5Persona(**filtered_persona)


def parse_context(context_dict: dict) -> SituationalContext:
    if not context_dict or not context_dict.get("enabled", True):
        return SituationalContext.disabled()

    return SituationalContext(
        enabled=True,
        inflation=getattr(InflationLevel, context_dict["inflation"]) if context_dict.get("inflation") else None,
        interest_rates=getattr(InterestRateLevel, context_dict["interest_rates"]) if context_dict.get("interest_rates") else None,
        government=getattr(GovernmentOrientation, context_dict["government"]) if context_dict.get("government") else None,
        crises=[getattr(CrisisType, c) for c in context_dict.get("crises", []) if c],
        country=context_dict.get("country"),
        year=str(context_dict["year"]) if context_dict.get("year") else None,
        gdp_growth=context_dict.get("gdp_growth"),
        unemployment=context_dict.get("unemployment"),
        custom_conditions=[c for c in context_dict.get("custom_conditions") or [] if c],
    )


def parse_utility_params(utility_cfg: dict, chaves_agentes: list[str], papeis: list[str],) -> dict[str, RoleUtilityParams]:
    """
    Lê o bloco 'utility' do config.yaml e constrói um dict role → RoleUtilityParams.

    Exemplo no config.yaml:
        utility:
          buyer:
            role_type: "buyer"
            p_target: 150000
            p_floor: 180000
            currency: "R$"
            unit: "/ano"
          seller:
            role_type: "seller"
            p_target: 180000
            p_floor: 114000
            currency: "R$"
            unit: "/ano"
    """
    if not utility_cfg:
        return {}

    agent_to_role = {chave: role for chave, role in zip(chaves_agentes, papeis)}

    params = {}
    for key, cfg in utility_cfg.items():
        role = agent_to_role.get(key, key)
        params[role] = RoleUtilityParams(
            role=role,
            role_type=cfg.get("role_type", "buyer"),
            p_target=float(cfg["p_target"]),
            p_floor=float(cfg["p_floor"]),
            currency=cfg.get("currency", "R$"),
            unit=cfg.get("unit", ""),
        )
    return params


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = load_config("config.yaml")
    exp    = config["experiment"]
    print(f"Iniciando experimento: {exp['name']}...")

    # Cenário
    scenario_name = exp["scenario"]
    if scenario_name not in SCENARIO_REGISTRY:
        raise ValueError(f"Cenário '{scenario_name}' não encontrado!")
    scenario = SCENARIO_REGISTRY[scenario_name]

    if "max_turns" in exp:
        scenario.max_turns = exp["max_turns"]
        print(f"⚙️  Limite de turnos ajustado para: {scenario.max_turns}")

    # 1. Juiz
    ag_judge = create_adapter(config["models"]["judge"])

    # 2. Agentes e personas
    papeis_do_cenario    = list(scenario.roles.keys())
    chaves_agentes_yaml  = [k for k in config["models"].keys() if k != "judge"]

    if len(chaves_agentes_yaml) < len(papeis_do_cenario):
        raise ValueError("Agentes insuficientes no YAML para este cenário!")

    agents_dict  = {}
    personas_dict = {}

    print("-" * 40)
    for i, role_name in enumerate(papeis_do_cenario):
        chave = chaves_agentes_yaml[i]
        agents_dict[role_name]  = create_adapter(config["models"][chave])
        personas_dict[role_name] = parse_persona(config["models"][chave])
        print(f"Papel '{role_name.upper()}' → '{chave}'")
    print("-" * 40)

    # 3. Contexto macroeconômico
    macro_context = parse_context(config.get("context", {}))

    # 4. Configuração do avaliador (métricas Big Five + negociação)
    metricas_textos = config["models"]["judge"].get("metrics", [])
    config_juiz = EvaluatorConfig.from_strings(metricas_textos) if metricas_textos else None

    # 5. Simulação principal
    result, profiles, _ = run_negotiation(
        scenario=scenario,
        agents=agents_dict,
        judge=ag_judge,
        evaluator_config=config_juiz,
        turn_delay_seconds=exp.get("turn_delay_seconds", 0.0),
        personas=personas_dict,
        context=macro_context,
        output_dir="results/",
        use_system_reminder=exp.get("use_system_reminder", False),
    )
    print("✅ Simulação concluída.")

    # 6. Utilidade econômica (opcional — só roda se 'utility' estiver no config.yaml)
    utility_results = None
    utility_cfg = config.get("utility", {})
    if utility_cfg:
        print("📊 Calculando utilidade econômica...")
        utility_params = parse_utility_params(utility_cfg, chaves_agentes_yaml, papeis_do_cenario)
        utility_calc    = UtilityCalculator(judge=ag_judge, role_params=utility_params)
        utility_results = utility_calc.evaluate(result)
        print("✅ Utilidade calculada.")

    # 7. Satisfação pós-negociação — IPC (sempre roda)
    print("📋 Avaliando satisfação (IPC — 16 questões por agente)...")
    sat_evaluator        = SatisfactionEvaluator(judge=ag_judge)
    satisfaction_results = sat_evaluator.evaluate_all(result)
    print("✅ Satisfação avaliada.")

    # 8. Regera o relatório com as novas seções
    report_path = f"results/{result.scenario_name}_{result.run_id}_report.md"
    generate_report(
        result=result,
        profiles=profiles,
        output_path=report_path,
        utility_results=utility_results,
        satisfaction_results=satisfaction_results,
    )

    print(f"\n🎉 Experimento concluído! Relatório: {report_path}")
