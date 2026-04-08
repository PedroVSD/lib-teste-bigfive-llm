import os
from llm_negotiation_analyst import run_negotiation
from llm_negotiation_analyst.adapters import OpenAIAdapter, GeminiAdapter
from llm_negotiation_analyst.scenarios import SALARY_NEGOTIATION

os.environ["OPENAI_API_KEY"] = "sk-sua-chave-aqui"
os.environ["GEMINI_API_KEY"] = "AIza-sua-chave-aqui"

print("Iniciando a Batalha: Gemini (Candidato) vs GPT-4o (Recrutador)...")

result, profiles, report = run_negotiation(
    scenario=SALARY_NEGOTIATION,
    agents={
        "candidate": GeminiAdapter("gemini-2.5-flash"),
        "recruiter": OpenAIAdapter("gpt-4o-mini"),
    },
    judge=OpenAIAdapter("gpt-4o"), # Usando o GPT-4o maior como juiz do Big Five
    output_dir="resultados_oficiais/",
)

print(f"Negociação finalizada em {result.total_turns} turnos!")
print(f"Relatório gerado na pasta: resultados_oficiais/")
