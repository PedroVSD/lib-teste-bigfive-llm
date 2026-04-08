import os
from dotenv import load_dotenv

from llm_negotiation_analyst import run_negotiation
from llm_negotiation_analyst.adapters.gemini_adapter import GeminiAdapter
from llm_negotiation_analyst.scenarios import SALARY_NEGOTIATION

# 1. Carrega as variáveis de ambiente (GEMINI_API_KEY)
load_dotenv()

print("Iniciando a simulação da negociação com modelos Gemini...")

# 2. Roda a simulação
result, profiles, report = run_negotiation(
    scenario=SALARY_NEGOTIATION,
    agents={
        # O Flash é absurdamente rápido e barato para os agentes conversarem
        "candidate": GeminiAdapter("gemini-2.5-flash"),
        "recruiter": GeminiAdapter("gemini-2.5-flash"),
    },
    # O Pro tem maior capacidade de raciocínio lógico para atuar como Juiz do Big Five
    judge=GeminiAdapter("gemini-2.5-pro"),
    output_dir="results/",
)

print("Simulação concluída com sucesso!")
print("Verifique a pasta 'results/' para ler o relatório em Markdown e os logs em JSONL.")
