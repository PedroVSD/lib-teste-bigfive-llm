# Documentação Oficial: LLM Negotiation Analyst
A llm_negotiation_analyst é uma biblioteca Python avançada desenvolvida para simular, estruturar e analisar interações complexas de negociação entre Modelos de Linguagem Grande (LLMs). O grande diferencial desta ferramenta é a sua capacidade de avaliar a expressão de traços de personalidade (Modelo Big Five) nas respostas geradas pelos modelos, utilizando uma arquitetura robusta de "LLM-as-a-Judge" (LLM como juiz).

## 1. Visão Geral da Arquitetura
O fluxo de dados da biblioteca é estritamente unidirecional e modular, garantindo que a simulação (onde os agentes interagem) seja completamente desacoplada da avaliação (onde o juiz pontua os turnos).

* **Cenário** (NegotiationScenario): Define as regras do jogo (contexto, papéis, limite de turnos, palavras-chave de acordo).

* **Adapters** (LLMAdapter): Conectam os "atores" (agentes) aos modelos de IA subjacentes (OpenAI, Ollama, Anthropic).

* **Motor** (SimulationEngine): Orquestra o diálogo. Injeta prompts, alterna os turnos e detecta acordos ("settlements"). O motor gera um NegotiationResult.

* **Avaliador** (Evaluator): Recebe o NegotiationResult e invoca um modelo Juiz (diferente dos agentes) para pontuar cada fala individualmente usando rubricas comportamentais (âncoras de 1 a 5).

* **Persistência** e Relatório: O StorageManager salva tudo em formato .jsonl (fácil de ingerir no Pandas), e o Generator compila um relatório legível em Markdown.

## 2. Instalação e Dependências
A biblioteca foi desenhada para ter dependências mínimas no seu núcleo. O armazenamento usa a biblioteca padrão do Python (json, pathlib) e a comunicação local usa httpx.


### Instalação base (ideal para rodar apenas com Ollama local)
pip install ".[httpx]"

### Para usar APIs comerciais como juízes ou agentes
pip install ".[openai]"
pip install ".[anthropic]"

### Instalação completa (inclui pandas e numpy para análise do output)
pip install ".[all]"

# 3. Guia de Uso: Modos de Simulação
A biblioteca suporta dois modos fundamentais de operação, expostos pelas funções principais do __init__.py.

### Modo 1: Agent vs. Agent (Simulação Autônoma)
Neste modo, dois LLMs são colocados frente a frente. O motor gerencia o histórico de conversa de cada um isoladamente. Útil para estudos generativos ou para testar como um modelo se sai contra um "adversário" de referência.

```
from llm_negotiation_analyst import run_negotiation
from llm_negotiation_analyst.adapters import OpenAIAdapter, OllamaAdapter
from llm_negotiation_analyst.scenarios import SALARY_NEGOTIATION

result, profiles, report = run_negotiation(
    scenario=SALARY_NEGOTIATION,
    agents={
        "candidate": OllamaAdapter("llama3.1:8b"),
        "recruiter": OpenAIAdapter("gpt-4o"),
    },
    judge=OpenAIAdapter("gpt-4o"), # Recomenda-se um modelo fechado e estável como juiz
    output_dir="resultados/",
)
```
### Modo 2: Benchmark (Simulação Controlada)
Ideal para ciência rigorosa. Um único modelo é testado contra uma sequência de prompts engessados (hardcoded). Isso garante que diferentes modelos (ou quantizações do mesmo modelo) sejam avaliados sob as exatas mesmas condições de contorno.

```
from llm_negotiation_analyst import run_benchmark

BENCHMARK_PROMPTS = [
    "Nossa oferta inicial é R$ 10.000/mês, o que acha?",
    "Não podemos subir o salário base, mas posso oferecer um bônus de assinatura.",
    "Essa é a nossa última oferta. Pegar ou largar."
]

result, profiles, report = run_benchmark(
    scenario=SALARY_NEGOTIATION,
    agent_role="candidate",
    agent_adapter=OllamaAdapter("mistral:7b"),
    benchmark_turns=BENCHMARK_PROMPTS,
    judge=OpenAIAdapter("gpt-4o"),
)
```

# 4. Referência da API (Core Modules)
### 4.1. Módulo adapters
O pacote fornece uma classe base abstrata LLMAdapter que padroniza como as mensagens são enviadas.

Como criar um adaptador customizado:
Se você precisar integrar com uma API proprietária (ex: Gemini ou Vertex AI), basta estender a classe base e implementar o método complete:

```
from llm_negotiation_analyst.adapters.base import LLMAdapter, AdapterConfig

class VertexAdapter(LLMAdapter):
    def complete(self, messages: list[dict], **kwargs) -> str:
        # 1. Converta o formato de mensagens [{"role": "...", "content": "..."}] 
        #    para o formato esperado pelo provedor.
        # 2. Faça a chamada de rede.
        # 3. Retorne apenas a string gerada.
        return "Texto da resposta"
```

A AdapterConfig permite controlar temperature, max_tokens e um dicionário extra (muito útil para passar uma seed de controle de aleatoriedade no Ollama e OpenAI).

### 4.2. Módulo scenarios
Um NegotiationScenario é um objeto de dados puramente declarativo.


| Atributo | Tipo | Descrição |
| :--- | :---: | :---: |
| name | str | Identificador único do cenário. |
| shared_context | str | Contexto passado para ambos os modelos no início (turn_index <= 1). |
| roles | dict[str, str] | Mapeamento entre o nome do papel (ex: buyer) e o System Prompt exclusivo daquele papel. |
| opening_role | str | Qual agente deve falar primeiro. |
| max_turns | int | Limite de falas antes do motor abortar a simulação (evita loops infinitos). |
| settlement_keywords | list[str] | Palavras (como "agreed", "fechado") que fazem o motor parar prematuramente se detectadas na resposta. |

### 4.3. Módulo scoring (A Avaliação do Big Five)
O coração científico da biblioteca mora em big5.py e evaluator.py. O avaliador usa um LLM isolado para ler cada fala individual e atribuir uma nota de 1 a 5 baseada em âncoras comportamentais (ex: 1 = Adversarial, 5 = Altamente colaborativo para a dimensão Agreeableness).

Configurações do EvaluatorConfig:

* dimensions: Permite rodar a avaliação apenas em partes do Big 5 para economizar tokens/custos de API. Padrão: Todas as 5.

* invert_neuroticism: O padrão do NEO-PI-R é que 5 indica alta reatividade emocional (Instável). Ao ativar esta flag (True), a nota é invertida matematicamente ($6.0 - \text{score}$), para que 5 passe a significar estabilidade emocional. Isso é útil se você quiser que notas altas em todas as dimensões sejam consideradas características "positivas" ou "maduras".

Confiabilidade Inter-Avaliadores (Dual-Judge IRR):
Para estudos rigorosos, passe um second_judge na instanciação do Evaluator. A biblioteca forçará os dois juízes a pontuar o mesmo turno e calculará a concordância normalizada, armazenando-a no campo confidence.

### 4.4. Módulos storage e report
A biblioteca não usa bancos relacionais complexos, favorecendo o padrão de Data Science: arquivos JSON Lines (JSONL).
Isso significa que cada simulação gera um identificador único de 8 caracteres (run_id).

* transcripts/{scenario}_{run_id}.jsonl: O log cru, com as mensagens.

* scores/{scenario}_{run_id}_scores.jsonl: As métricas Big 5 por agente.

* runs_index.jsonl: O arquivo central estilo catálogo. Cada linha é um resumo de um experimento realizado.

Para carregar os resultados depois, a biblioteca oferece integração nativa com Pandas:

```
from llm_negotiation_analyst.storage import StorageManager
store = StorageManager("results/")
df = store.to_dataframe("scores") 
# Retorna um DataFrame onde cada linha é o perfil de um modelo em um turno
```

# 5. Metodologia Científica e Vieses (Avisos de Uso)
Ao utilizar esta ferramenta para pesquisas acadêmicas ou comparativos de arquiteturas, atente-se às seguintes diretrizes embutidas na biblioteca:

1. Self-Evaluation Bias (Viés de Autoavaliação): Um LLM não deve avaliar a própria resposta. Se os agentes são llama3.1, o juiz deve ser um modelo com outra arquitetura ou com maior capacidade de raciocínio lógico (ex: gpt-4o ou claude-3-5-sonnet).

2. Observabilidade das Dimensões: A biblioteca pontua as 5 dimensões, mas como definido nos metadados de big5.py, nem todas são igualmente visíveis em negociações por texto.

    * Agreeableness (A) e Conscientiousness (C): Altamente observáveis em trocas verbais.

    * Neuroticism (N): Moderadamente observável através da volatilidade de concessões.

    * Openness (O) e Extraversion (E): Baixa observabilidade (difícil distinguir passividade de introversão via texto plano). As conclusões sobre estas dimensões devem ser muito cautelosas.

3. Agregação: A pontuação final (profile.scores) é a média aritmética das pontuações válidas por turno. Turnos em que a avaliação falhou (ex: Juiz não retornou JSON válido) retornam score 3 (neutro) mas com confidence 0, e são processados nos logs de erro.