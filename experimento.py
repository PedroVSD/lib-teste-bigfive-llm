import os
from pyexpat import model
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

# Importando classes de Persona e Contexto
from llm_negotiation_analyst.persona import Big5Persona
from llm_negotiation_analyst.context import (
    SituationalContext, InflationLevel, InterestRateLevel,
    GovernmentOrientation, CrisisType
)

# Importando as classes do Avaliador e das Dimensões
from llm_negotiation_analyst.scoring.evaluator import EvaluatorConfig


load_dotenv()

def load_config(filepath: str):
    with open(filepath, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def create_adapter(config_dict: dict):
    provider = config_dict["provider"].lower()
    model_name = config_dict["name"]

    temp = config_dict.get("temperature", 0.5)
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
        url = config_dict.get("base_url")
        return OllamaAdapter(model=model_name, base_url=url, config=config_obj)
    elif provider == "deepseek":
        url = config_dict.get("base_url")
        return DeepSeekAdapter(model=model_name, base_url=url, config=config_obj)
    raise ValueError(f"Provedor desconhecido: {provider}")

def parse_persona(agent_config: dict) -> Big5Persona | None:
    persona_dict = agent_config.get("persona") or {}
    tactics_dict = agent_config.get("tactics") or {}

    if not persona_dict and not tactics_dict:
        return None

    # 1. Filtra apenas as chaves válidas do Big Five
    chaves_big5 = {"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"}
    filtered_persona = {k: v for k, v in persona_dict.items() if k in chaves_big5 and v is not None}

    # 2. Pega as instruções extras normais (se houver)
    instrucoes_originais = persona_dict.get("extra_instructions", "")

    # 3. Constrói o bloco de texto das Táticas de Negociação
    builder = TacticsPromptBuilder()
    texto_taticas = builder.build(tactics_dict)

    # 4. Junta as instruções do usuário com as Táticas geradas
    texto_final = instrucoes_originais
    if texto_taticas:
        texto_final = f"{instrucoes_originais}\n\n{texto_taticas}".strip()

    if texto_final:
        filtered_persona["extra_instructions"] = texto_final

    return Big5Persona(**filtered_persona)

def parse_context(context_dict: dict) -> SituationalContext:
    if not context_dict or not context_dict.get("enabled", True):
        return SituationalContext.disabled()


    return SituationalContext(
        enabled=True,
        inflation=getattr(InflationLevel, context_dict["inflation"])
            if context_dict.get("inflation") else None,
        interest_rates=getattr(InterestRateLevel, context_dict["interest_rates"])
            if context_dict.get("interest_rates") else None,
        government=getattr(GovernmentOrientation, context_dict["government"])
            if context_dict.get("government") else None,
        crises=[getattr(CrisisType, c) for c in context_dict.get("crises", []) if c],
        country=context_dict.get("country"),
        year=str(context_dict["year"]) if context_dict.get("year") else None,
        gdp_growth=context_dict.get("gdp_growth"),
        unemployment=context_dict.get("unemployment"),
        custom_conditions=[c for c in context_dict.get("custom_conditions") or [] if c],
    )

if __name__ == "__main__":
    config = load_config("config.yaml")
    exp    = config["experiment"]
    print(f"Iniciando experimento: {exp['name']}...")

    scenario_name = exp["scenario"]
    if scenario_name not in SCENARIO_REGISTRY:
        raise ValueError(f"Cenário '{scenario_name}' não encontrado!")

    scenario = SCENARIO_REGISTRY[exp["scenario"]]

    if "max_turns" in exp:
        scenario.max_turns = exp["max_turns"]
        print(f"⚙️ Limite de turnos ajustado para: {scenario.max_turns}")

    # 1. Instancia os Agentes e o Juiz
    ag_judge = create_adapter(config["models"]["judge"])

    papeis_do_cenario = list(scenario.roles.keys())
    chaves_agentes_yaml = [k for k in config["models"].keys() if k != "judge"]

    if len(chaves_agentes_yaml) < len(papeis_do_cenario):
        raise ValueError("Você não configurou agentes suficientes no YAML para este cenário!")

    agents_dict = {}
    personas_dict = {}

    # 2. Carrega Personalidades
    print("-" * 40)
    for i, role_name in enumerate(papeis_do_cenario):
        chave_do_agente = chaves_agentes_yaml[i] # Pega agent_1, depois agent_2...

        # Cria o modelo e a persona
        agents_dict[role_name] = create_adapter(config["models"][chave_do_agente])
        personas_dict[role_name] = parse_persona(config["models"][chave_do_agente])

        print(f"Papel '{role_name.upper()}' assumido por -> '{chave_do_agente}'")
    print("-" * 40)

    # 3. Carrega Contexto Macroeconômico
    macro_context = parse_context(config.get("context"))

    # 4. Carrega Configuração do Avaliador (Métricas)
    metricas_textos = config["models"]["judge"].get("metrics", [])
    if metricas_textos:
        # Se encontrou métricas no YAML, converte elas para o Enum do Python
        #dimensoes_ativas = [Dimension(m) for m in metricas_textos]
        config_juiz = EvaluatorConfig.from_strings(metricas_textos)
    else:
        # Se não tiver, envia None (o juiz usará o Big Five padrão)
        config_juiz = None

    # 5. Roda a Simulação
    result, profiles, report = run_negotiation(
        scenario=scenario,
        agents=agents_dict,
        judge=ag_judge,
        evaluator_config=config_juiz,  # <-- Injetando a configuração do juiz aqui!
        turn_delay_seconds=config["experiment"]["turn_delay_seconds"],
        personas=personas_dict,
        context=macro_context,
        output_dir="results/",
        use_system_reminder=config["experiment"].get("use_system_reminder", False),
    )

    print("Simulação concluída com sucesso! Verifique a pasta 'results/'.")
