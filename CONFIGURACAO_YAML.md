# Guia Completo de Configuração — `config.yaml`

Este documento lista **todas as variações possíveis** no `config.yaml` para configurar o experimento `llm-negotiation-analyst`. Toda a execução é controlada por este arquivo (`experimento.py:32` `load_config()`).

> Dica: use um arquivo por experimento e rode com `uv run python experimento.py configs/estagflacao.yaml` — o `experiment.name` vira automaticamente o nome do arquivo (`experimento.py:16`).

---

## 1. Bloco `experiment`

```yaml
experiment:
  name: "estagflacao"              # opcional — se vazio ou "teste_modelos_todas_as_variaveis", vira Path(stem) do YAML
  scenario: "salary_negotiation"   # obrigatório — ver §5 Cenários
  max_turns: 8                     # opcional — sobrescreve NegotiationScenario.max_turns (cada turno = 1 fala de um agente)
  turn_delay_seconds: 10           # opcional — delay entre turnos (para rate-limit)
  use_system_reminder: true        # opcional — injeta lembrete [ACORDO_FECHADO]/SIMULACAO_CONCLUIDA
```

| Chave | Tipo | Obrigatório | Valores |
|---|---|---|---|
| `name` | `str` | não | qualquer string; usado no print e `metadata` |
| `scenario` | `str` | sim | `salary_negotiation` / `company_acquisition` / `strategic_supplier_contract` / `property_boundary_dispute` (`scenarios/__init__.py:52`) |
| `max_turns` | `int` | não | `1..20` (uma rodada = fala de ambos) |
| `turn_delay_seconds` | `float` | não | `0`, `10` etc |
| `use_system_reminder` | `bool` | não | `true`/`false` |

---

## 2. Bloco `context` — Contexto macroeconômico (`context/situational.py:262`)

Pode ser via **preset** (recomendado) ou manual. Se `preset` for usado, campos manuais **complementam/sobrescrevem**.

### 2.1 Via preset (10 cenários prontos)

```yaml
context:
  preset: "estagflacao"
  # opcionais que sobrescrevem o preset:
  custom_conditions:
    - "Condição extra específica"
```

| `preset` | Descrição curta |
|---|---|
| `crescimento_forte` / `expansao` | **Expansão**: `LOW/LOW/MARKET_FRIENDLY`, PIB 4-5%, desemprego 4-5%, mercado aquecido — trabalhador com alto poder de barganha. |
| `recessao` / `recessao_economica` | **Recessão**: `LOW/HIGH/CONSERVATIVE`, `ECONOMIC_RECESSION`, PIB -2..-3%, desemprego 10-12% — empregador com poder. |
| `estagflacao` | **Estagflação** (mais interessante): `VERY_HIGH/HIGH/INTERVENTIONIST`, PIB -1.5%, desemprego 10%, 12%/18% — conflito custo de vida vs queda de demanda. |
| `boom_inflacionario` | **Boom inflacionário**: `HIGH/HIGH/MARKET_FRIENDLY`, PIB 4-6%, desemprego 3-4% — crescimento + inflação corroendo salário; bom para negociação salarial. |
| `crise_financeira` | **Crise financeira**: `MODERATE/VERY_HIGH/INTERVENTIONIST`, `FINANCIAL_CRISIS`, PIB -4..-5%, crédito restrito — patrimônio ≠ liquidez. |
| `crise_politica` | **Crise política**: `HIGH/HIGH/TRANSITIONAL`, `POLITICAL_INSTABILITY`, PIB ~1%, incerteza muito alta (40% reforma tributária) — risco futuro. |
| `governo_intervencionista` / `intervencionista` | **Intervencionista**: `MODERATE/MODERATE/INTERVENTIONIST`, regulação/impostos altos — útil empresa×governo, sindicato×empresa. |
| `governo_liberal` / `liberal` | **Liberal pró-mercado**: `LOW/MODERATE/LIBERAL_ON_MARKET(austrian)`, regulação baixa, competição alta. |
| `crise_desemprego` / `crise_emprego` | **Desemprego**: `LOW/LOW/TECHNOCRATIC`, `ECONOMIC_RECESSION`, 16% desemprego — oferta de mão de obra muito alta. |
| `anarcho_capitalist` / `anarco_capitalista` | **Anarco-capitalista**: `LOW/LOW/ANCAP`, sem banco central, tributação muito baixa, competição muito alta, arbitragem privada. |

Alias com/sem acento e com ` `, `-`, `_` são normalizados (`experimento.py:87`).

### 2.2 Manual (todos opcionais — só o preenchido é injetado)

```yaml
context:
  enabled: true
  inflation: "VERY_HIGH"         # ver §2.3
  interest_rates: "HIGH"
  government: "INTERVENTIONIST"
  crises:
    - "ECONOMIC_RECESSION"
    - "POLITICAL_INSTABILITY"
  gdp_growth: "Queda acentuada no último semestre."   # str livre
  unemployment: "Taxas estáveis, mas em mercado informal." # str livre
  custom_conditions:             # list[str] livre
    - "O fornecedor principal ameaçou cancelar contratos anteriores."
```

| Variável | Valores válidos (`situational.py`) | Injetado como |
|---|---|---|
| `enabled` | `true`/`false` (`situational.py:297`) | `false` = `ContextPromptBuilder.build()` retorna `""` |
| `inflation` | `VERY_LOW`(<2%), `LOW`(2-4%), `MODERATE`(4-7%), `HIGH`(7-10%), `VERY_HIGH`(>10%) (`situational.py:64`) | `Macroeconomic environment: ...` (`_INFLATION_DESC:106`) |
| `interest_rates` | `VERY_LOW`/`LOW`/`MODERATE`/`HIGH`/`VERY_HIGH` (`situational.py:72`) | `Interest rates: ...` (`_INTEREST_DESC:134`) |
| `government` | `MARKET_FRIENDLY`/`INTERVENTIONIST`/`TECHNOCRATIC`/`POPULIST`/`TRANSITIONAL`/`CONSERVATIVE`/`ANCAP`/`LIBERAL_ON_MARKET(austrian)` (`situational.py:80`) | `Political/institutional environment: ...` (`_GOVERNMENT_DESC:162`) |
| `crises` | `ECONOMIC_RECESSION`/`FINANCIAL_CRISIS`/`POLITICAL_INSTABILITY`/`HEALTH_PANDEMIC`/`SUPPLY_CHAIN`/`ENERGY_CRISIS`/`GEOPOLITICAL`/`CURRENCY_CRISIS` (`situational.py:91`) | `Active crisis: ...` por crise (`_CRISIS_DESC:212`) |
| `gdp_growth` | `str` livre | `GDP growth: ...` |
| `unemployment` | `str` livre | `Unemployment: ...` |
| `custom_conditions` | `list[str]` | `Custom conditions:` + cada linha |

> Remover a linha = omitir. Não duplicar chaves no mesmo mapa YAML (erro `Map keys must be unique`).

---

## 3. Bloco `models` — Agentes e Juiz

As chaves `agent_1`, `agent_2`, ... são mapeadas na ordem aos `roles` do cenário (`experimento.py:172` `papeis_do_cenario`). `judge` é separado.

```yaml
models:
  agent_1:   # → 1º role do cenário (ex: candidate em salary_negotiation)
    provider: "ollama"         # ollama / ollama_local / gemini / openai / lmstudio / deepseek
    name: "gemma4:31b-cloud"   # modelo no provider
    temperature: 0.5           # 0.0 determinístico — 1.0 criativo
    max_tokens: 2048           # limite de geração (4096 evita cortes)
    persona:                   # Big Five — bipolar
      agreeableness: positive  # positive / negative / none (ou null/~)
      neuroticism: negative
      extraversion: positive
      openness: none           # desativa
      conscientiousness: positive
      extra_instructions: "Texto livre adicional"
    tactics:                   # binário enabled/disabled (legado 1-5 ainda ok)
      anchoring: enabled
      anchor_susceptibility: enabled
      loss_aversion: enabled
      conditional_concession: enabled
      value_creation: enabled
      rapport: enabled
      resilience: enabled
      clarity: enabled
      fact_justification: enabled

  agent_2:   # → 2º role (ex: recruiter)
    provider: "gemini"
    name: "gemma-4-26b-a4b-it"
    temperature: 0.5
    max_tokens: 2048
    persona:
      agreeableness: positive
      # ... idem
    tactics:
      anchoring: 2
      # ...

  judge:
    provider: "ollama"
    name: "gpt-oss:120b-cloud"
    temperature: 0.0
    metrics:                   # quais dimensões o juiz avalia (1 chamada por turno para todas as métricas, com histórico)
      - "agreeableness"        # Big Five: openness, conscientiousness, extraversion, agreeableness, neuroticism
      - "neuroticism"
      - "extraversion"
      - "openness"
      - "conscientiousness"
      - "anchoring"            # Táticas: anchoring, conditional_concession, value_creation
      - "rapport"              # Emocional: rapport, resilience
      - "resilience"
      - "clarity"              # Argumentação: fact_justification, clarity
      - "fact_justification"
      - "anchor_susceptibility" # Vieses: anchor_susceptibility, loss_aversion
      - "loss_aversion"
      - "value_creation"
```

### 3.1 `provider` → Adapter (`adapters/`)

| `provider` | Classe (`adapters/__init__.py`) | `base_url` / API key |
|---|---|---|
| `ollama` | `OllamaAdapter` (cloud) | `OLLAMA_BASE_URL` + `OLLAMA_API_KEY` |
| `ollama_local` | `OllamaLocalAdapter` | fixo `http://localhost:11434` |
| `gemini` | `GeminiAdapter` | `GEMINI_API_KEY` |
| `openai` | `OpenAIAdapter` | `OPENAI_API_KEY` |
| `lmstudio` | `LMStudioAdapter` | `http://localhost:1234/v1` |
| `deepseek` | `DeepSeekAdapter` | `DEEPSEEK_API_KEY` |

### 3.2 `persona` — Big Five bipolar (`persona/big5_persona.py:76`)

| Dimensão | `positive` (alto) | `negative` (baixo) | `none` |
|---|---|---|---|
| `agreeableness` | Cooperativo/pró-social | Competitivo/adversarial | desativa |
| `conscientiousness` | Organizado/preciso | Impulsivo/vago | desativa |
| `neuroticism` | Instável/reativo (IV invertido: `positive`=instável) | Estável/composto | desativa |
| `extraversion` | Assertivo/dominante | Passivo/reservado | desativa |
| `openness` | Criativo/integrativo | Rígido/convencional | desativa |

Guias em `_GUIDANCE:76`; nomes em `_DIM_NAMES`. `none`/`null`/`nil`/`~`/omitir = não injeta.

### 3.3 `tactics` — `PRESENT`/`ABSENT`/`NOT_APPLICABLE` (`persona/tactics_builder.py:14`, `scoring/negotiation_metrics.py:1`)

| Métrica | `ABSENT` / `disabled` | `PRESENT` / `enabled` | `NOT_APPLICABLE` |
|---|---|---|---|
| `anchoring` | não injeta | Âncora forte | não injeta (sem oportunidade) |
| `conditional_concession` | não injeta | Trocas estritas | — |
| `value_creation` | não injeta | Integrativo/criativo | — |
| `rapport` | não injeta | Altamente empático | — |
| `resilience` | não injeta | Calmo/inabalável | — |
| `clarity` | não injeta | Estruturado/matemático | — |
| `anchor_susceptibility` | não injeta (imune) | Facilmente influenciado | — |
| `loss_aversion` | não injeta (imune) | Reativo à perda | — |
| `fact_justification` | não injeta | Altamente embasado | — |

`PRESENT` = comportamento observado no turno; `ABSENT` = oportunidade havia mas ausente; `NOT_APPLICABLE` = turno sem oportunidade suficiente (não entra no denominador) e deve ter `evidence` curta. Alias `enabled→PRESENT`, `disabled/absent→ABSENT`, `none/not_applicable→ignorado`. Legado `1-5` ainda funciona (`1-2→ABSENT`, `4-5→PRESENT`) para compatibilidade. Todos os 10 `configs/*.yaml` já com as 9 métricas `present`/`enabled` em ambos agentes e `judge.metrics` com as 14 dimensões, e `experimento.py:272` avalia **utilidade** (`utility` contínua 0-1) e **satisfação** (IPC 1-7) separadamente.

**Observação categórica em lote (mesma base):** mesmo `NEGOTIATION_META` usado para induzir é usado para julgar. O juiz (`scoring/evaluator.py:62` `_JUDGE_SYSTEM_BATCH`) recebe **por turno, em 1 chamada**, a **resposta completa** do agente + **histórico da negociação** (janela 8 turnos, `_format_history`) + `anchor_present/absent` de **todas** as métricas listadas em `judge.metrics`. Retorna em lote `{"evaluations": {"anchoring": {"result": "PRESENT", "evidence": "..."}, ...}}` (`N` turnos = `N` chamadas, antes `N×M`). Agregação posterior: `occurrence_rate = PRESENT / (PRESENT + ABSENT)` nos turnos aplicáveis (`NOT_APPLICABLE` ignorado) → `65% (13/20; 5 NA)`, permitindo `persistence_rate/first/last_occurrence` por sequência temporal `Turn 1:0, Turn 2:N/A...`. Big Five respeita polaridade: `positive→PRESENT`, `negative→ABSENT`; táticas `enabled→PRESENT`. Ex: `Induzido PRESENT` vs `Observado 65%` → `✅ Compatível` (`≥50%`), `20%` → `❌` (`report/generator.py:250`).

### 3.4 `judge.metrics` (comportamentais em lote, com contexto)

Lista livre de Big Five + `NegotiationMetric` (`scoring/negotiation_metrics.py:42`). Se omitido, avalia só Big Five. Para cada turno, **uma única chamada** avalia todas as métricas com contexto: `Turn i` (resposta completa) + `Turns 0..i-1` (histórico) + rubricas `PRESENT/ABSENT`. Cada turno retorna `{"evaluations": {metric: {result: PRESENT|ABSENT|NOT_APPLICABLE, evidence}}}` (`scoring/evaluator.py:98`). Métricas comportamentais **não** usam média imediata: `occurrence_rate` posterior (`NOT_APPLICABLE` ignorado) → `65%`. `utility` (contínua 0-1, `utility.py:147`) e `satisfaction` (ordinal 1-7, `satisfaction.py:48`) separadas; `agreement` `AGREEMENT|NO_AGREEMENT`. Relatório separa `3 Behavioral Metrics | 5 Utility | 6 Subjective`; `evidence` por `turn×metric` persiste em `scores.jsonl` (`storage/jsonl_store.py:89`).

---

## 4. Bloco `utility` — Utilidade econômica (opcional)

```yaml
utility:
  agent_1:
    role_type: "buyer"   # buyer = quer maximizar (u_b), seller = minimizar (u_s)
    p_target: 18000      # alvo (R$/mês)
    p_floor: 15500       # piso (buyer teto, seller piso)
    currency: "R$"
    unit: "/mês"
  agent_2:
    role_type: "seller"
    p_target: 14000
    p_floor: 16500
    currency: "R$"
    unit: "/mês"
```

Fórmulas `scoring/utility.py:8`: `u_s(p)=(p-p_s)/(p̄_s-p_s)`, `u_b(p)=(p̄_b-p)/(p̄_b-p_b)`. Se ausente, seção `5. Utilidade` omitida.

---

## 5. Cenários possíveis (`scenarios/__init__.py:56`)

| `scenario` | Descrição | `roles` (ordem) | `settlement_keywords` | `max_turns` |
|---|---|---|---|---|
| `salary_negotiation` | Negociação salarial: engenheiro vs empresa (R$12k vs R$16k + bônus/férias) | `candidate` → `recruiter` | `SIMULACAO_CONCLUIDA`, `ACORDO_FECHADO` | 8 |
| `company_acquisition` | Aquisição de empresa: fundador vs compradora (R$8M vs R$5-7M, earn-out, IP) | `seller` → `buyer` | `SIMULACAO_CONCLUIDA`, `ACORDO_FECHADO` | 10 |
| `strategic_supplier_contract` | Contrato fornecedor: comprador vs fornecedor (R$1.200 vs R$950, volume/prazo) | `buyer` → `supplier` | `SIMULACAO_CONCLUIDA`, `ACORDO_FECHADO` | 10 |
| `property_boundary_dispute` | Disputa propriedade: owner_a vs owner_b (12m², R$80k, muro) | `owner_a` → `owner_b` | `SIMULACAO_CONCLUIDA`, `ACORDO_FECHADO` | 10 |
| *custom* | Criar em `scenarios/__init__.py:228` `NegotiationScenario(name=..., roles={...}, opening_role=..., max_turns=...)` e registrar em `SCENARIO_REGISTRY:175` | — | — | — |

Encerramento só quando **ambos** confirmarem acordo (`simulation/engine.py:271`).

---

## 6. Exemplo mínimo por contexto

```yaml
# configs/estagflacao.yaml
experiment: {name: estagflacao, scenario: salary_negotiation, max_turns: 8}
context: {preset: estagflacao}
models:
  agent_1: {provider: ollama, name: "gemma4:31b-cloud", persona: {agreeableness: positive}}
  agent_2: {provider: gemini, name: "gemma-4-26b-a4b-it", persona: {agreeableness: negative}}
  judge: {provider: ollama, name: "gpt-oss:120b-cloud", metrics: [agreeableness, anchoring]}
```

Todos os 10 presets têm arquivo pronto em `configs/*.yaml`.
