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
        data = yaml.safe_load(file)
    # Nome do experimento = nome do arquivo YAML (sem extensão)
    # Ex: `estagflacao.yaml` → experiment.name = "estagflacao" (sempre sobrescreve o YAML)
    # display_name/title com espaços (ex: "teste com X variável") é preservado em display_name/yaml_name
    try:
        from pathlib import Path
        file_stem = Path(filepath).stem
        exp = data.get("experiment") or {}
        yaml_name = exp.get("name")
        # title/display_name/label explícito tem prioridade como nome humano
        display_raw = exp.get("title") or exp.get("display_name") or exp.get("label") or yaml_name
        exp["name"] = file_stem
        if display_raw and str(display_raw).strip() and str(display_raw).strip() != file_stem:
            exp["display_name"] = str(display_raw).strip()
            exp["yaml_name"] = str(display_raw).strip()  # compat
        elif yaml_name and yaml_name != file_stem:
            exp["yaml_name"] = yaml_name
            exp["display_name"] = yaml_name
        exp["config_file"] = filepath
        data["experiment"] = exp
    except Exception:
        pass
    return data


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
    elif provider == "openrouter":
        from llm_negotiation_analyst.adapters.openrouter_adapter import OpenRouterAdapter
        return OpenRouterAdapter(
            model=model_name,
            api_key=config_dict.get("api_key"),
            base_url=config_dict.get("base_url"),
            referer=config_dict.get("referer"),
            title=config_dict.get("title"),
            config=config_obj,
        )
    raise ValueError(f"Provedor desconhecido: {provider}. Opções: gemini, openai, lmstudio, ollama, ollama_local, deepseek, openrouter")


def parse_persona(agent_config: dict) -> Big5Persona | None:
    persona_dict = agent_config.get("persona") or {}
    tactics_dict = agent_config.get("tactics") or {}

    if not persona_dict and not tactics_dict:
        return None

    chaves_big5 = {"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"}
    # 'none'/'null'/'nil' (string) também desativa; None já filtra mas deixamos
    # passar para que Big5Persona normalize para None (trait omitido)
    filtered_persona = {}
    for k, v in persona_dict.items():
        if k not in chaves_big5:
            continue
        # mantém 'none' como string para normalização -> None; None real já é desativado
        filtered_persona[k] = v

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

    # Suporte a `preset:` — carrega um dos 10 contextos prontos e permite sobrescrever campos
    from llm_negotiation_analyst.context import ContextPresets
    import unicodedata
    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode()
        return s.lower().replace("-", "_").replace(" ", "_")
    _PRESET_MAP = {
        "crescimento_forte": ContextPresets.crescimento_forte,
        "crescimento_economico_forte": ContextPresets.crescimento_forte,
        "expansao": ContextPresets.crescimento_forte,
        "recessao": ContextPresets.recessao,
        "recessao_economica": ContextPresets.recessao,
        "estagflacao": ContextPresets.estagflacao,
        "boom_inflacionario": ContextPresets.boom_inflacionario,
        "crise_financeira": ContextPresets.crise_financeira,
        "crise_politica": ContextPresets.crise_politica,
        "governo_intervencionista": ContextPresets.governo_intervencionista,
        "intervencionista": ContextPresets.governo_intervencionista,
        "governo_liberal": ContextPresets.governo_liberal,
        "liberal": ContextPresets.governo_liberal,
        "governo_liberal_pro_mercado": ContextPresets.governo_liberal,
        "crise_desemprego": ContextPresets.crise_desemprego,
        "crise_emprego": ContextPresets.crise_desemprego,
        "anarcho_capitalist": ContextPresets.anarcho_capitalist,
        "anarco_capitalista": ContextPresets.anarcho_capitalist,
        "anarchocapitalist": ContextPresets.anarcho_capitalist,
    }
    preset_name = context_dict.get("preset")
    if preset_name:
        key = _norm(preset_name)
        factory = _PRESET_MAP.get(key)
        if not factory:
            raise ValueError(f"Preset de contexto desconhecido: '{preset_name}'. Opções: {list(_PRESET_MAP.keys())}")
        base = factory()
        # Sobrescreve com campos manuais se fornecidos
        if context_dict.get("inflation"):
            base.inflation = getattr(InflationLevel, context_dict["inflation"])
        if context_dict.get("interest_rates"):
            base.interest_rates = getattr(InterestRateLevel, context_dict["interest_rates"])
        if context_dict.get("government"):
            base.government = getattr(GovernmentOrientation, context_dict["government"])
        if context_dict.get("crises") is not None:
            base.crises = [getattr(CrisisType, c) for c in context_dict.get("crises", []) if c]
        if context_dict.get("gdp_growth") is not None:
            base.gdp_growth = context_dict.get("gdp_growth")
        if context_dict.get("unemployment") is not None:
            base.unemployment = context_dict.get("unemployment")
        if context_dict.get("custom_conditions") is not None:
            extra = [c for c in context_dict.get("custom_conditions") or [] if c]
            # Se preset já tem custom_conditions, anexa
            base.custom_conditions = list(base.custom_conditions) + extra
        return base

    return SituationalContext(
        enabled=True,
        inflation=getattr(InflationLevel, context_dict["inflation"]) if context_dict.get("inflation") else None,
        interest_rates=getattr(InterestRateLevel, context_dict["interest_rates"]) if context_dict.get("interest_rates") else None,
        government=getattr(GovernmentOrientation, context_dict["government"]) if context_dict.get("government") else None,
        crises=[getattr(CrisisType, c) for c in context_dict.get("crises", []) if c],
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
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)
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

    # 1. Juiz(es)
    ag_judge = create_adapter(config["models"]["judge"])
    second_judge = None
    if "second_judge" in config["models"]:
        second_judge = create_adapter(config["models"]["second_judge"])
        print(f"⚖️  Segundo juiz ativado: {config['models']['second_judge'].get('name')} ({config['models']['second_judge'].get('provider')}) — IRR será calculado")
    elif "judge2" in config["models"]:
        second_judge = create_adapter(config["models"]["judge2"])
        print(f"⚖️  Segundo juiz ativado: {config['models']['judge2'].get('name')}")

    # 2. Agentes e personas
    papeis_do_cenario    = list(scenario.roles.keys())
    chaves_agentes_yaml  = [k for k in config["models"].keys() if k not in ("judge", "second_judge", "judge2")]

    if len(chaves_agentes_yaml) < len(papeis_do_cenario):
        raise ValueError("Agentes insuficientes no YAML para este cenário!")

    agents_dict  = {}
    personas_dict = {}
    tactics_dict = {}

    print("-" * 40)
    for i, role_name in enumerate(papeis_do_cenario):
        chave = chaves_agentes_yaml[i]
        agents_dict[role_name]  = create_adapter(config["models"][chave])
        personas_dict[role_name] = parse_persona(config["models"][chave])
        # Guarda tactics separadamente para relatório (mesmo que persona seja None)
        raw_tactics = config["models"][chave].get("tactics") or {}
        # Filtra none/null/nil e normaliza
        tactics_dict[role_name] = {k: v for k, v in raw_tactics.items() if v is not None and str(v).lower() not in ("none","null","nil")}
        print(f"Papel '{role_name.upper()}' → '{chave}'")
    print("-" * 40)

    # 3. Contexto macroeconômico
    macro_context = parse_context(config.get("context", {}))

    # 4. Configuração do avaliador (métricas Big Five + negociação)
    metricas_textos = config["models"]["judge"].get("metrics", [])
    config_juiz = EvaluatorConfig.from_strings(metricas_textos) if metricas_textos else None

    # 5. Simulação principal
    experiment_name = exp.get("name")
    display_name = exp.get("display_name") or exp.get("title") or exp.get("yaml_name")
    result, profiles, _ = run_negotiation(
        scenario=scenario,
        agents=agents_dict,
        judge=ag_judge,
        second_judge=second_judge,
        evaluator_config=config_juiz,
        turn_delay_seconds=exp.get("turn_delay_seconds", 0.0),
        personas=personas_dict,
        tactics=tactics_dict,
        context=macro_context,
        output_dir="results/",
        use_system_reminder=exp.get("use_system_reminder", False),
        experiment_name=experiment_name,
        experiment_display_name=display_name,
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

    # 8. Regera o relatório com as novas seções (usa display_name + experiment_name no nome do arquivo)
    exp_name = result.metadata.get("experiment_name") or experiment_name or result.scenario_name
    display = result.metadata.get("experiment_display_name") or result.metadata.get("experiment_title") or result.metadata.get("yaml_name")
    if display:
        safe_display = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in str(display)).strip()
        # keep spaces for readability but filesystem-safe (spaces allowed); also keep file_stem
        prefix = f"{safe_display}_{exp_name}_{result.scenario_name}" if exp_name else f"{safe_display}_{result.scenario_name}"
    else:
        prefix = f"{exp_name}_{result.scenario_name}" if exp_name else result.scenario_name
    report_path = f"results/{prefix}_{result.run_id}_report.md"
    generate_report(
        result=result,
        profiles=profiles,
        output_path=report_path,
        utility_results=utility_results,
        satisfaction_results=satisfaction_results,
    )

    print(f"\n🎉 Experimento concluído! Relatório: {report_path}")
