import os
import yaml
from dotenv import load_dotenv

from llm_negotiation_analyst import run_negotiation
from llm_negotiation_analyst.adapters.gemini_adapter import GeminiAdapter
from llm_negotiation_analyst.adapters.openai_adapter import OpenAIAdapter
from llm_negotiation_analyst.adapters.lmstudio_adapter import LMStudioAdapter
from llm_negotiation_analyst.scenarios import SALARY_NEGOTIATION

# Importando classes de Persona e Contexto
from llm_negotiation_analyst.persona import Big5Persona
from llm_negotiation_analyst.context import (
    SituationalContext, InflationLevel, InterestRateLevel,
    GovernmentOrientation, CrisisType
)

# Importando as classes do Avaliador e das Dimensões
from llm_negotiation_analyst.scoring.evaluator import EvaluatorConfig
from llm_negotiation_analyst.scoring.big5 import Dimension

load_dotenv()

def load_config(filepath: str):
    with open(filepath, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def create_adapter(config_dict: dict):
    provider = config_dict["provider"].lower()
    model_name = config_dict["name"]

    if provider == "gemini":
        return GeminiAdapter(model=model_name)
    elif provider == "openai":
        return OpenAIAdapter(model=model_name)
    elif provider == "lmstudio":
        return LMStudioAdapter(model=model_name)
    raise ValueError(f"Provedor desconhecido: {provider}")

def parse_persona(persona_dict: dict) -> Big5Persona:
    if not persona_dict:
        return None
    # Passa diretamente os valores do YAML (1 a 5) para a dataclass
    return Big5Persona(**persona_dict)

def parse_context(context_dict: dict) -> SituationalContext:
    if not context_dict or not context_dict.get("enabled", True):
        return SituationalContext.disabled()

    # Converte as strings do YAML para os Enums do Python de forma segura
    inflation_val = context_dict.get("inflation")
    interest_val = context_dict.get("interest_rates")
    gov_val = context_dict.get("government")
    crises_list = context_dict.get("crises", [])

    return SituationalContext(
        enabled=True,
        inflation=getattr(InflationLevel, inflation_val) if inflation_val else None,
        interest_rates=getattr(InterestRateLevel, interest_val) if interest_val else None,
        government=getattr(GovernmentOrientation, gov_val) if gov_val else None,
        crises=[getattr(CrisisType, c) for c in crises_list],
        country=context_dict.get("country"),
        year=str(context_dict.get("year")) if context_dict.get("year") else None,
        gdp_growth=context_dict.get("gdp_growth"),
        unemployment=context_dict.get("unemployment"),
        custom_conditions=context_dict.get("custom_conditions", [])
    )

if __name__ == "__main__":
    config = load_config("config.yaml")
    print(f"Iniciando experimento: {config['experiment']['name']}...")

    # 1. Instancia os Agentes e o Juiz
    ag_candidate = create_adapter(config["models"]["candidate"])
    ag_recruiter = create_adapter(config["models"]["recruiter"])
    ag_judge = create_adapter(config["models"]["judge"])

    # 2. Carrega Personalidades
    personas_dict = {
        "candidate": parse_persona(config["models"]["candidate"].get("persona")),
        "recruiter": parse_persona(config["models"]["recruiter"].get("persona")),
    }

    # 3. Carrega Contexto Macroeconômico
    macro_context = parse_context(config.get("context"))

    # 4. Carrega Configuração do Avaliador (Métricas)
    metricas_textos = config["models"]["judge"].get("metrics", [])
    if metricas_textos:
        # Se encontrou métricas no YAML, converte elas para o Enum do Python
        dimensoes_ativas = [Dimension(m) for m in metricas_textos]
        config_juiz = EvaluatorConfig(dimensions=dimensoes_ativas)
    else:
        # Se não tiver, envia None (o juiz usará o Big Five padrão)
        config_juiz = None

    # 5. Roda a Simulação
    result, profiles, report = run_negotiation(
        scenario=SALARY_NEGOTIATION,
        agents={
            "candidate": ag_candidate,
            "recruiter": ag_recruiter,
        },
        judge=ag_judge,
        evaluator_config=config_juiz,  # <-- Injetando a configuração do juiz aqui!
        turn_delay_seconds=config["experiment"]["turn_delay_seconds"],
        personas=personas_dict,
        context=macro_context,
        output_dir="results/",
    )

    print("Simulação concluída com sucesso! Verifique a pasta 'results/'.")
