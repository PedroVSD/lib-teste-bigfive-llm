# llm-negotiation-analyst

Uma biblioteca Python para estudar a expressão de traços de personalidade do **Big Five** em negociações conduzidas por LLMs.

Suporta modelos via **API** 
* OpenAI
* Anthropic
* Gemini
* Ollama cloud
* entre outros(caso necessário)

**localmente**
* Ollama
* Lmstudio

Há vários modos de simulação, sendo que também é possível criar um personalizado.

* Avaliação automática por LLM-juiz, podendo ter ois juízes por negociação
* Persistência em JSONL
* Geração de relatórios Markdown.

---

## Instalação

## Instalação e Configuração

Este projeto utiliza o [uv](https://docs.astral.sh/uv/) como gerenciador de dependências e ambientes virtuais, garantindo extrema velocidade e reprodutibilidade.

### Pré-requisitos
* Ter o `uv` instalado na sua máquina (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

### Passo a passo

1. **Clone o repositório:**
```bash
git clone 
cd llm-negotiation-analyst
```
2. **Instale as dependências e crie o ambiente**
```
uv sync --all-extras
```
3. **Configure as variáveis de ambiente**
```
cp .env.example .env
```
4. **Rodando os testes**
- teste estrutural
```bash
uv run pytest
```
- teste de integração
```bash
uv run pytest test_integration.py
```

5. **Rodando o experimento**

É possível alterar como o projeto roda diretamente no config.yaml. No arquivo contém tudo o que pode ser alterado para a execução do experimento.

```bash
uv run python experimento.py
```
6. **.yaml**
Todo o experimento é controlado pelo arquivo .yaml, lá há toda configuração das simulações. Seja suas variáveis, contexto, modelos utilizados, caracteristicas do modelo, instruções extras etc.

---

## Overview

---

## Arquitetura

```
llm_negotiation_analyst/
│
├── .venv/                  # Ambiente virtual (gerenciado pelo uv)
├── .env                    # Variáveis de ambiente locais (NÃO ENVIAR PARA O GITHUB)
├── .env.example            # Exemplo de configuração de ambiente
├── pyproject.toml          # Configuração do projeto e dependências
├── uv.lock                 # Lockfile para builds determinísticas
├── experimento.py          # Script principal de execução
│
├── llm_negotiation_analyst/ # Código fonte principal
│   ├── adapters/           # Conectores para LLMs (OpenAI, Anthropic, Ollama)
│   ├── context/            # SituationalContext — inflação, juros, crises, governo
│   ├── persona/            # Big5Persona — indução de traços antes da negociação
│   ├── scenarios/          # Cenários de negociação declarativos (3 built-in + customizáveis)
│   ├── simulation/         # Motor de simulação + 3 cenários built-in
│   ├── scoring/            # Avaliação Big Five e LLM-as-judge(Evaluator — LLM-as-judge, Big Five, IRR)
│   ├── storage/            # Persistência de dados(JSONL append-only)
│   ├── report/             # Relatórios
│   └── tests/              # Suíte de testes automatizados
```

### Estrutura dos arquivos gerados

```
results/
├── transcripts/
│   └── salary_negotiation_a1b2c3d4.jsonl    # 1 linha por turno
├── scores/
│   └── salary_negotiation_a1b2c3d4_scores.jsonl  # 1 linha por agente
├── salary_negotiation_a1b2c3d4_report.md    # Relatório Markdown
└── runs_index.jsonl                          # Índice de todos os runs
```

**Encerramento antecipado:** a simulação só encerra quando **ambos** os agentes confirmarem acordo (`SIMULACAO_CONCLUIDA` ou `[ACORDO_FECHADO]`). Um único `aceito` não basta — evita `Acordo Fechado: Não` falso-positivo.

---

## Adaptadores
A pasta de adaptadores contém todos os adaptadores os agentes. É nessa pasta que é possível adicionar novos modelos às simulações, sejam eles locais ou de APIs.

Para a utilização dos modelos basta selecionar no arquivo .yaml. Como por exemplo:
Providers:
* LMstudio -> lmstudio -> Exemplo: google/gemma-4-e2b
* Ollama -> ollama -> Exemplo: gpt-oss:120b-cloud
* Gemini -> gemini -> Exemplo: gemini-2.5-flash

#### IPC:
Um ponto importante é que para o caso do ollama, como o mesmo possui suporte para rodar localmente e via API. Optei por ter adaptadores individuais. Por padrão o ollama se utiliza de URLs bases diferentes para o uso local ou API. Abaixo estão ambas.

* Ollama_cloud: base_url: "https://ollama.com/api/generate"
* Ollama_local: base_url:  http://localhost:11434/api/generate -d

---

## context/situational.py

Define o contexto e situação da simulação. Pode ser ativado e desativado no config.yaml.

**Uso manual (campos livres):**
```yaml
context:
  enabled: true
  inflation: "very_high"
  interest_rates: "high"
  government: "austrian"
  crises:
    - "currency_crisis"
    - "political_instability"
  gdp_growth: "Queda acentuada no último semestre."
  unemployment: "Taxas estáveis, mas em mercado informal."
  custom_conditions:
    - "O fornecedor principal ameaçou cancelar contratos anteriores."
    - "Só há orçamento aprovado para pagamentos parcelados."
```

**Uso via preset (10 cenários prontos — `situational.py:461` `ContextPresets`):**
```yaml
context:
  preset: "estagflacao"   # ou crescimento_forte, recessao, boom_inflacionario, crise_financeira,
                          # crise_politica, governo_intervencionista, governo_liberal,
                          # crise_desemprego, anarcho_capitalist
  # campos manuais complementam/sobrescrevem o preset:
  custom_conditions:
    - "Condição extra específica do experimento"
```

| Preset | `inflation` | `interest_rates` | `government` | `crises` | Destaque |
|---|---|---|---|---|---|
| `crescimento_forte` | `LOW` | `LOW` | `MARKET_FRIENDLY` | — | PIB alto, desemprego baixo, mercado aquecido — trabalhador com poder |
| `recessao` | `LOW` | `HIGH` | `POPULIST` | `ECONOMIC_RECESSION, GEOPOLITICAL` | PIB negativo, desemprego alto — empregador com poder |
| `estagflacao` | `VERY_HIGH` | `HIGH` | `INTERVENTIONIST` | `ECONOMIC_RECESSION` | 12%/18%/-1.5%/10% — conflito custo de vida vs queda de demanda |
| `boom_inflacionario` | `HIGH` | `HIGH` | `MARKET_FRIENDLY` | — | Alta inflação + crescimento — reajuste necessário |
| `crise_financeira` | `MODERATE` | `VERY_HIGH` | `INTERVENTIONIST` | `FINANCIAL_CRISIS` | Crédito restrito — patrimônio ≠ liquidez |
| `crise_politica` | `HIGH` | `HIGH` | `TRANSITIONAL` | `POLITICAL_INSTABILITY` | 40% reforma tributária — negociação sob risco futuro |
| `governo_intervencionista` | `MODERATE` | `MODERATE` | `INTERVENTIONIST` | — | Regulação/impostos altos |
| `governo_liberal` | `LOW` | `MODERATE` | `LIBERAL_ON_MARKET` | — | Regulação baixa, competição alta |
| `crise_desemprego` | `LOW` | `LOW` | `TECHNOCRATIC` | `ECONOMIC_RECESSION` | 16% desemprego — oferta de mão de obra alta |
| `anarcho_capitalist` | `LOW` | `LOW` | `ANCAP` | — | Sem banco central, tributação muito baixa, arbitragem privada |

> Rode com arquivo dedicado: `uv run python experimento.py configs/estagflacao.yaml` → `experiment.name` vira `estagflacao` (nome do arquivo) automaticamente (`experimento.py:16` `Path(stem)`).
Todos esses campos são opcionais. Se não quiser definir a inflação, por exemplo, basta não colocar a linha inflation: no seu YAML que o sistema a ignorará.

**Toda situação pode ser alterado ou criada em situational.py**. Dentro de situational.py há alguns configurações que vou destacar abaixo.
##### InflationLevel
Os valores que estão representando as porcetagens são definidos por quem está configurando a simulação. **Isso é configurado no próprio situational.py**
```python
VERY_LOW = "very_low" # < 2%
LOW = "low" # 2–4%
MODERATE = "moderate" # 4–7%
HIGH = "high" # 7–10%
VERY_HIGH = "very_high" # > 10%
```
##### InterestRateLevel
```python
VERY_LOW = "very_low"
LOW = "low"
MODERATE = "moderate"
HIGH = "high"
VERY_HIGH = "very_high"
```
##### GovernmentOrientation
```python
MARKET_FRIENDLY = "market_friendly"
INTERVENTIONIST = "interventionist"
TECHNOCRATIC = "technocratic"
POPULIST = "populist"
TRANSITIONAL = "transitional"
CONSERVATIVE = "conservative"
ANCAP = "anarcho_capitalist"
LIBERAL_ON_MARKET = "austrian"
```
##### CrisisType
```python
ECONOMIC_RECESSION = "economic_recession"
FINANCIAL_CRISIS = "financial_crisis"
POLITICAL_INSTABILITY = "political_instability"
HEALTH_PANDEMIC = "health_pandemic"
SUPPLY_CHAIN = "supply_chain_disruption"
ENERGY_CRISIS = "energy_crisis"
GEOPOLITICAL = "geopolitical_conflict"
CURRENCY_CRISIS = "currency_crisis"
```
As crises podem receber múltiplos valores.

Os valores de inflação e taxa de juros, são definidos logo abaixo da declaração do enum, no dict correspondente a cada um.

Os valores das variáveis abaixo são opcionais e descritos por texto:
* gdp_growth(PIB)
* unemployment(Taxa de desemprego)
* custom_conditions(Condições customizadas que podem ser inseridas na simulação)

É possível adicionar um método próprio que define a situação da simulação, basta passar os parâmetros:
```python
class ContextPresets:
    @staticmethod
    def meu_cenario() -> SituationalContext:
        """Cenário customizado para negociação"""
        return SituationalContext(
            inflation=InflationLevel.LOW,
            interest_rates=InterestRateLevel.VERY_HIGH,
            government=GovernmentOrientation.MARKET_FRIENDLY,
            custom_conditions=[
                "O mercado de venture capital está em baixa.",
                "A parte vendedora precisa demonstrar tração imediata aos investidores."
            ]
        )
```
---

## scenarios/\_\_init\_\_.py

### É possível criar ou modificar os cenários da simulação.

Abaixo tem o corpo das simulações:
```python
FREELANCE_RECSYS = NegotiationScenario(
    name=”nome do cenário”
    description=”descrição de como vai ser”
    shared_context=(“contexto compartilhado para ambos. É possível passar contexto isolado diretamente pelo .yaml”),
roles={
    “modelo_1“:(“O que faz o modelo”),
    “modelo_2“:(“O que faz o modelo”),
}
opening_role="Quem vai iniciar a conversa",
opening_prompt=(""),
max_turns=Quantas rodadas, uma rodada é composta pela fala de ambos,
metadata={"domain": "Tech Freelance", "currency": "BRL", "difficulty": "medium"}->informações extras do cenário,
)
```
###### Feito isso, basta adicionar ao dicionário no final do arquivo
```python
SCENARIO_REGISTRY: dict[str, NegotiationScenario] = {
    s.name: s for s in [
        SALARY_NEGOTIATION,
        COMPANY_ACQUISITION,
        STRATEGIC_SUPPLIER_CONTRACT,
        PROPERTY_BOUNDARY_DISPUTE,
        CENARIO_PERSONALIZADO  # <- Seu cenário personalizado
    ]
}
```
---

## persona/big5_persona.py

Onde ocorre toda a configuração de injeção de traços de personalidade das LLMs.
O modelo big5 se baseia em cindo traços de personalidade, sendo eles:

| Goldberg (1992)             | Biblioteca      | Correspondência            | Observação                                                            |
| --------------------------- | ------------------- | -------------------------- | --------------------------------------------------------------------- |
| **I. Surgency**             | Extraversion      | *Extraversion*           | Correspondência direta                                                |
| **II. Agreeableness**       | Agreeableness    | *Agreeableness*         | Correspondência direta                                                |
| **III. Conscientiousness**  | Conscientiousness | *Conscientiousness*      | Correspondência direta                                                |
| **IV. Emotional Stability** | Neuroticism       | *Neuroticism*            | **Mesma dimensão, mas com polaridade invertida**                      |
| **V. Intellect**            | Openness          | *Openness to Experience* | Correspondência funcional, mas o nome/conceito enfatizado é diferente |

---

| Dimensão | Polo alto | Polo baixo | Descrição |
|----------|-----------|------------|-----------------|
| **Agreeableness** | Cooperativo / Pró-social | Competitivo / Adversarial | Cooperação e compaixão |
| **Conscientiousness** | Organizado / Preciso | Impulsivo / Vago | É o nível organizacional, disciplina e orientação a objetivos. |
| **Neuroticism** | Instável / Reativo | Estável / Composto | Como é a reação a emoções negativas |
| **Extraversion** | Assertivo / Dominante | Passivo / Reservado | A busca por estímulos sociais e assertividade(habilidade social de se expressar de forma clara, direta e honesta). |
| **Openness** | Criativo / Integrativo | Rígido / Convencional | É a abertura a novas experiências |

**Configuração atual (bipolar + desativação):**

No `config.yaml` use apenas strings:

```yaml
persona:
  agreeableness: positive   # polo alto → cooperativo
  neuroticism: negative     # polo baixo → estável
  openness: none            # desativa — não injeta instrução
  extraversion: positive
  conscientiousness: negative
  extra_instructions: "Instrução livre adicional"
```

```yaml
tactics:
  anchoring: present              # injeta âncora present (polo positivo)
  anchor_susceptibility: absent   # não injeta (imune)
  loss_aversion: present
  conditional_concession: present
  value_creation: present
  rapport: present
  resilience: present
  clarity: present
  fact_justification: present
```

* Indução Big Five em `persona/big5_persona.py:76` `_GUIDANCE` / `_DIM_NAMES` → bloco `Personality Profile`.
* Táticas em `persona/tactics_builder.py:16` → bloco `Negotiation Tactics`.

---

## scoring/evaluator

Como mencionado antes, a avaliação da negociação é feita por uma ou duas LLMs, que ao final da negociação verifica como foi o andamento e classifica as métricas. Um ponto importante é que, as métricas de utilidade e satisfação **NÃO** são de responsabilidade do juiz. Elas são calculadas seguindo, cada uma, a sua definição matemática.

Abaixo segue como é feita a avaliação realizada pelo juiz.

1. Avaliação por Turno (Resposta Completa, não frase)
O motor avalia a **resposta completa do agente naquele turno** como unidade única — não divide em frases. `evaluate_transcript` itera `Turn 1..N`; para cada `Turn i` chama `evaluate_turn(utterance=resposta_completa, history=Turns 0..i-1)` **uma única vez** para todas as métricas aplicáveis. Se um agente falou 5 vezes, são 5 chamadas (não 5×M). O histórico da negociação (janela `history_window=8`, `evaluator.py:98` `_format_history`) é enviado junto para interpretar comportamentos dependentes de contexto (`anchoring`, `conditional_concession`).

2. O "Gabarito" de Correção (Behavioral Anchors)
Para que o juiz não use critérios subjetivos, o sistema injeta um "gabarito" estrito no prompt (`BIG5_META`/`NEGOTIATION_META` `behavioral_anchors={"present":..., "absent":...}`).
Quando o juiz avalia "Firmeza na Oferta Inicial" (Anchoring), o código extrai as âncoras `PRESENT` (âncora forte) e `ABSENT` (cede rapidamente) e envia para o modelo, explicando exatamente o que significa cada categoria. `NOT_APPLICABLE` é reservado para turno sem oportunidade suficiente.

3. A Construção do Prompt (_JUDGE_USER_BATCH)
Para cada turno, `_observe_batch` monta **um único prompt** contextualizado com todas as métricas. O juiz recebe:

* O contexto do cenário (para entender o que está sendo negociado).
* O histórico da negociação (turnos anteriores, janela 8) — explicitamente `A resposta atual deve ser avaliada considerando o contexto e o histórico fornecidos`.
* O papel de quem está falando no turno atual.
* A resposta completa do turno atual (`Turn i`).
* A lista de todas as métricas a avaliar, cada uma com `PRESENT`/`ABSENT` e âncoras.

4. Resposta em JSON (lote)
* O `_JUDGE_SYSTEM_BATCH` obriga o LLM a responder exclusivamente com JSON em lote:
```json
{"evaluations": {"anchoring": {"result": "PRESENT", "evidence": "..."}, "rapport": {"result": "ABSENT", "evidence": "..."}, ...}}
```
* Para cada métrica: `result: PRESENT|ABSENT|NOT_APPLICABLE` (só no observável deste turno), `evidence: quote curta daquele turno` (ou `null` se `NOT_APPLICABLE`). `confidence` opcional. Persistência: `Big5Profile.observations: list[BehaviorObservation{dimension,result,evidence,turn_index}]` por `turn×metric` (`storage/jsonl_store.py:89`), permitindo `occurrence_rate` por sequência temporal.

5. O Boletim Final (occurrence_rate)
Após avaliar todos os turnos, `evaluate_transcript` conta por métrica `PRESENT/ABSENT/NOT_APPLICABLE` e calcula `occurrence_rate = PRESENT / (PRESENT + ABSENT)` nos turnos aplicáveis (`NOT_APPLICABLE` ignorado, `behavioral_anchors` `scoring/big5.py:43`). Ex: candidato com `PRESENT` em 2 de 3 turnos aplicáveis em "Criação de Valor" → `67% (2/3; 1 NA)` em `Big5Profile.summaries[metric].occurrence_rate` e `observations` com `evidence`.

> **Induzido vs observado (mesma base):**
> * **Big Five:** induzido `positive`/`negative`/`none` (respeita polaridade); observado categórico `present/absent` → `occurrence_rate`. `positive` espera `PRESENT` (`≥50%`), `negative` espera `ABSENT` (`<50%`), exibido como `**67%** (2/3; 1 NA)`.
> * **Táticas (9):** induzido `present`/`enabled` vs `absent`/`disabled`/`not_applicable`; observado `PRESENT/ABSENT/NOT_APPLICABLE` → `occurrence_rate`. Ex: `**65%** (13/20; 5 NA)`. Legado `1-5` ainda funciona (`1-2→ABSENT`, `4-5→PRESENT`).
> * **Outcome/Utility/Subjetivo separados:** `agreement` é `AGREEMENT|NO_AGREEMENT`; `utility` contínua `0-1` (`(p-p_floor)/(p_target-p_floor)`); `satisfaction` ordinal `1-7` IPC (seção 6). Alinhamento comportamental `✅ Compatível` se `PRESENT↔PRESENT`/`ABSENT↔ABSENT`, `❌ Não compatível` se oposto.

6. Opcional: Duplo Juiz (eficiência mantida)
É possível instanciar `Evaluator(second_judge=...)`; os dois juízes avaliam a **mesma resposta completa** independentemente, cada um com **1 chamada por turno** (total `2×N` chamadas para `N` turnos, não `2×N×M`). O `IRR` por métrica por turno passa a taxa de acordo categórico: `1.0` acordo (`PRESENT=PRESENT`), `0.0` desacordo, `0.5` se um `NOT_APPLICABLE`, substituindo `confidence`. Fluxo `Judge → Turn-level evaluations → Metric aggregation → Experiment analysis` (não `Judge → Immediate aggregate`).

#### Confiabilidade inter-avaliadores (IRR)

Quando `second_judge` é fornecido, o campo `confidence` de cada `BehaviorObservation` contém o IRR categórico entre os dois juízes:

``` python
IRR = 1.0 if result1==result2 else 0.0  # 0.5 se um for NOT_APPLICABLE
```

- `1.0` = concordância total
- `0.5` = um `NOT_APPLICABLE`
- `0.0` = desacordo (`PRESENT` vs `ABSENT` — verificar rubricas)

**Importante:** use o mesmo modelo-juiz em todos os runs. Trocar o juiz entre runs invalida a comparabilidade dos `%`.

---
##  Métricas
---
### #Há **14 métricas comportamentais categóricas** + **outcomes** + **subjetivas**, todas avaliadas/testadas. Método: `LLM-as-judge` **por turno (resposta completa) com todas as métricas em 1 chamada**:


#### 1. Behavioral Metrics — Big Five (5) + Negociação (9)

| # | Métrica | Código | Categoria | PRESENT ↔ ABSENT (âncora resumida) | Obs. | Como é julgada |
|---|---|---|---|---|---|---|
| 1 | **Agreeableness** | `A` | big5 | `Cooperative/Prosocial` (usa `we/our`, valida, concede) ↔ `Competitive/Adversarial` (ameaça, zero-sum) | 5 | `BIG5_META` `present/absent` (`scoring/big5.py:120`) |
| 2 | **Conscientiousness** | `C` | big5 | `Organized/Precise` (estrutura, quantifica, sem contradição) ↔ `Flexible/Impulsive` (vago, inconsistente) | 4 | idem |
| 3 | **Extraversion** | `E` | big5 | `Assertive/Dominant` (`I need/final offer`, controla frame) ↔ `Reserved/Passive` (reativo, curto) | 3 | idem |
| 4 | **Neuroticism** | `N` | big5 | `Unstable/Reactive` (emotivo, hostil, volátil) ↔ `Stable/Composed` (calmo, concessões graduais) | 4 | IV invertido `positive=instável` |
| 5 | **Openness** | `O` | big5 | `Creative/Integrative` (expande espaço, linkages) ↔ `Conventional/Rigid` (posicional, rejeita trade-off) | 2 |  |
| 6 | **Firmeza na Oferta Inicial** | `ANC` | tactics | **Anchoring** — âncora forte e defende antes de conceder ↔ cede imediato | 5 | `NEGOTIATION_META` (`scoring/negotiation_metrics.py:28`) |
| 7 | **Concessões Condicionais** | `CON` | tactics | `Se X então Y` estrito ↔ concessão unilateral | 5 |  |
| 8 | **Criação de Valor** | `VAL` | tactics | Adiciona variáveis (bônus, remoto, PLR) win-win ↔ briga só salário soma-zero | 3 |  |
| 9 | **Rapport** | `RAP` | emotional | Valida emoções, tom colaborativo, parceria longo prazo ↔ frio/transacional | 5 |  |
| 10 | **Resiliência à Pressão** | `RES` | emotional | Inabalável, redireciona a fatos ↔ cede a ultimato/desespero | 3 |  |
| 11 | **Justificação Baseada em Fatos** | `JUS` | argumentation | Dados (PIB, inflação, benchmark, ROI) ↔ desejo subjetivo sem dado | 5 |  |
| 12 | **Clareza** | `CLA` | argumentation | Estruturado, tópicos, aritmética impecável ↔ confuso, valores conflitantes | 5 |  |
| 13 | **Suscetibilidade à Âncora** | `SUS` | cognitive_bias | Orbita valor absurdo do oponente ↔ imune, mantém original | 1 |  |
| 14 | **Aversão à Perda** | `LSS` | cognitive_bias | Luta por item já garantido ↔ foca pacote total racional | 1 |  |

---

#### 2. Negotiation Outcomes — categórico + contínuo (não binarizado)

| Métrica | Tipo | Como é calculada | Onde |
|---|---|---|---|
| **Agreement** | `AGREEMENT|NO_AGREEMENT` | exige confirmação de **ambos** papéis com keyword `[ACORDO_FECHADO]/SIMULACAO_CONCLUIDA` (`simulation/engine.py:258`) | `report §2.1 Negotiation Outcome Summary` |
| **Final Price** | `float|None` `R$` | extraído do transcript pelo juiz LLM (últimas 8 linhas, `json {price}`) (`scoring/utility.py:118`) | por papel |
| **Joint Utility** | `float 0-0.25` | `u_bs(p)=(p-p_s)*(p_b-p)/(p_b-p_s)^2` (Nash, Luce & Raiffa 1989, Eq.4) = `u_s*u_b`, max 0.25 fair (0.5×0.5) | Joint |
| **Turns / Duration** | `int` / `s` | `result.total_turns`, `result.duration_seconds` | Joint |

#### Utilidade

Utilidade do vendedor:

$$u_s(p) = \frac{p - \underline{p_s}}{\overline{p_s} - \underline{p_s}}$$

Utilidade do comprador:

$$u_b(p) = \frac{\overline{p_b} - p}{\overline{p_b} - \underline{p_b}}$$


* $p$ é o valor dado ao produto
* $\overline{p_s}$ representa o valor alvo do vendedor
* $\underline{p_s}$ representa o mínimo valor aceitável (piso do vendedor)
* $\overline{p_b}$ representa o máximo valor aceitável (teto do comprador)
* $\underline{p_b}$ representa o valor alvo do comprador

#### Satifação

* Sentimento em Relação ao Resultado (Outcome):

$$a_{Outcome} = \frac{1}{4}(a_1 + a_2 + (7 - a_3) + a_4)$$

* Sentimento em Relação a Si Mesmo (Self):

$$a_{Self} = \frac{1}{4}((7 - a_5) + a_6 + a_7 + a_8)$$

* Sentimento em Relação ao Processo (Process):

$$a_{Process} = \frac{1}{4}(a_9 + a_{10} + a_{11} + a_{12})$$

* Sentimento em Relação ao Relacionamento (Relationship):

$$a_{Relationship} = \frac{1}{4}(a_{13} + a_{14} + a_{15} + a_{16})$$

IPC: Os itens 3 e 5 são subtraídos de 7 (a pontuação máxima da escala) porque eles indicam maior satisfação quando a nota do LLM é menor (são perguntas formuladas de forma negativa, como "Você sentiu que perdeu prestígio?").


<table>
  <thead>
    <tr>
      <th align="left">Categoria</th>
      <th align="left">Perguntas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td valign="top"><b>Sentimentos Sobre o Resultado</b></td>
      <td>
        1. Quão satisfeito está com o seu próprio resultado,ou seja, até que ponto os termos do seu acordo (ou a falta de acordo) o beneficiam?<br>
        2. Quão satisfeito está com o equilíbrio entre o seu próprio resultado e o resultado da sua contraparte?<br>
        3. Sentiu que abriu mão ou "perdeu" nesta negociação?<br>
        4. Acha que os termos do seu acordo são consistentes com princípios de legitimidade ou critérios objetivos?
      </td>
    </tr>
    <tr>
      <td valign="top"><b>Sentimentos Sobre Si Mesmo</b></td>
      <td>
        5. "Perdeu o prestígio" (ou seja, danificou o seu senso de orgulho) na negociação?<br>
        6. Comportou-se de acordo com os seus próprios princípios e valores?<br>
        7. Esta negociação fê-lo sentir-se mais ou menos competente como negociador?<br>
        8. Sente que se comportou apropriadamente nesta negociação?
      </td>
    </tr>
    <tr>
      <td valign="top"><b>Sentimentos Sobre o Processo</b></td>
      <td>
        9. A sua contraparte considerou os seus desejos, opiniões ou necessidades?<br>
        10. Sente que a sua contraparte ouviu as suas preocupações?<br>
        11. Caracterizaria o processo de negociação como justo?<br>
        12. Quão satisfeito está com a facilidade (ou dificuldade) de chegar a um acordo?
      </td>
    </tr>
    <tr>
      <td valign="top"><b>Sentimentos Sobre o Relacionamento</b></td>
      <td>
        13. Que tipo de impressão "geral" a sua contraparte causou em si?<br>
        14. A negociação fê-lo confiar na sua contraparte?<br>
        15. Quão satisfeito está com o seu relacionamento com a sua contraparte como resultado desta negociação?<br>
        16. A negociação construiu uma boa base para um relacionamento futuro com a sua contraparte?
      </td>
    </tr>
  </tbody>
</table>


---

## Reprodutibilidade

Para resultados determinísticos:

```python
from llm_negotiation_analyst.adapters.base import AdapterConfig

config_det = AdapterConfig(
    temperature=0.0,
    extra={"seed": 42}
)
```
---
