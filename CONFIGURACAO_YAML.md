# Complete Configuration Guide — `config.yaml`

This document lists **every possible variation** in `config.yaml` for configuring the `llm-negotiation-analyst` experiment. The entire run is controlled by this file (`experimento.py:32` `load_config()`).

> Tip: use one file per experiment and run with `uv run python experimento.py configs/estagflacao.yaml` — `experiment.name` automatically becomes the filename (`experimento.py:16`).

---

## 1. `experiment` block

```yaml
experiment:
  name: "estagflacao"              # optional — when empty or "teste_modelos_todas_as_variaveis", becomes the YAML Path(stem)
  scenario: "salary_negotiation"   # required — see §5 Scenarios
  max_turns: 8                     # optional — overrides NegotiationScenario.max_turns (each round = 1 utterance per agent)
  turn_delay_seconds: 10           # optional — delay between turns (for rate-limit)
  use_system_reminder: true        # optional — injects the [ACORDO_FECHADO]/SIMULACAO_CONCLUIDA reminder
```

| Key | Type | Required | Values |
|---|---|---|---|
| `name` | `str` | no | any string; used in prints and `metadata` |
| `scenario` | `str` | yes | `salary_negotiation` / `company_acquisition` / `strategic_supplier_contract` / `property_boundary_dispute` (`scenarios/__init__.py:52`) |
| `max_turns` | `int` | no | `1..20` (one round = everyone speaks) |
| `turn_delay_seconds` | `float` | no | `0`, `10` etc |
| `use_system_reminder` | `bool` | no | `true`/`false` |

---

## 2. `context` block — Macroeconomic context (`context/situational.py:262`)

Use a **preset** (recommended) or manual fields. When `preset` is used, manual fields **complement/override** it.

### 2.1 Via preset (10 ready scenarios)

```yaml
context:
  preset: "estagflacao"
  # optionals overriding the preset:
  custom_conditions:
    - "Extra specific condition"
```

| `preset` | Short description |
|---|---|
| `crescimento_forte` / `expansao` | **Expansion**: `LOW/LOW/MARKET_FRIENDLY`, GDP 4-5%, unemployment 4-5%, hot market — worker with high bargaining power. |
| `recessao` / `recessao_economica` | **Recession**: `LOW/HIGH/CONSERVATIVE`, `ECONOMIC_RECESSION`, GDP -2..-3%, unemployment 10-12% — employer with power. |
| `estagflacao` | **Stagflation** (most interesting): `VERY_HIGH/HIGH/INTERVENTIONIST`, GDP -1.5%, unemployment 10%, 12%/18% — cost-of-living vs falling-demand conflict. |
| `boom_inflacionario` | **Inflationary boom**: `HIGH/HIGH/MARKET_FRIENDLY`, GDP 4-6%, unemployment 3-4% — growth + wage-eroding inflation; good for salary negotiation. |
| `crise_financeira` | **Financial crisis**: `MODERATE/VERY_HIGH/INTERVENTIONIST`, `FINANCIAL_CRISIS`, GDP -4..-5%, restricted credit — equity ≠ liquidity. |
| `crise_politica` | **Political crisis**: `HIGH/HIGH/TRANSITIONAL`, `POLITICAL_INSTABILITY`, GDP ~1%, very high uncertainty (40% tax reform) — future risk. |
| `governo_intervencionista` / `intervencionista` | **Interventionist**: `MODERATE/MODERATE/INTERVENTIONIST`, high regulation/taxes — useful company-vs-government, union-vs-company. |
| `governo_liberal` / `liberal` | **Pro-market liberal**: `LOW/MODERATE/LIBERAL_ON_MARKET(austrian)`, low regulation, high competition. |
| `crise_desemprego` / `crise_emprego` | **Unemployment**: `LOW/LOW/TECHNOCRATIC`, `ECONOMIC_RECESSION`, 16% unemployment — very high labor supply. |
| `anarcho_capitalist` / `anarco_capitalista` | **Anarcho-capitalist**: `LOW/LOW/ANCAP`, no central bank, very low taxation, very high competition, private arbitration. |

Aliases with/without accents and with ` `, `-`, `_` are normalized (`experimento.py:87`).

### 2.2 Manual (all optional — only filled fields are injected)

```yaml
context:
  enabled: true
  inflation: "VERY_HIGH"         # see §2.3
  interest_rates: "HIGH"
  government: "INTERVENTIONIST"
  crises:
    - "ECONOMIC_RECESSION"
    - "POLITICAL_INSTABILITY"
  gdp_growth: "Sharp decline in the last semester."   # free str
  unemployment: "Stable rates, but informal market." # free str
  custom_conditions:             # free list[str]
    - "The main supplier threatened to cancel previous contracts."
```

| Variable | Valid values (`situational.py`) | Injected as |
|---|---|---|
| `enabled` | `true`/`false` (`situational.py:297`) | `false` = `ContextPromptBuilder.build()` returns `""` |
| `inflation` | `VERY_LOW`(<2%), `LOW`(2-4%), `MODERATE`(4-7%), `HIGH`(7-10%), `VERY_HIGH`(>10%) (`situational.py:64`) | `Macroeconomic environment: ...` (`_INFLATION_DESC:106`) |
| `interest_rates` | `VERY_LOW`/`LOW`/`MODERATE`/`HIGH`/`VERY_HIGH` (`situational.py:72`) | `Interest rates: ...` (`_INTEREST_DESC:134`) |
| `government` | `MARKET_FRIENDLY`/`INTERVENTIONIST`/`TECHNOCRATIC`/`POPULIST`/`TRANSITIONAL`/`CONSERVATIVE`/`ANCAP`/`LIBERAL_ON_MARKET(austrian)` (`situational.py:80`) | `Political/institutional environment: ...` (`_GOVERNMENT_DESC:162`) |
| `crises` | `ECONOMIC_RECESSION`/`FINANCIAL_CRISIS`/`POLITICAL_INSTABILITY`/`HEALTH_PANDEMIC`/`SUPPLY_CHAIN`/`ENERGY_CRISIS`/`GEOPOLITICAL`/`CURRENCY_CRISIS` (`situational.py:91`) | `Active crisis: ...` per crisis (`_CRISIS_DESC:212`) |
| `gdp_growth` | free `str` | `GDP growth: ...` |
| `unemployment` | free `str` | `Unemployment: ...` |
| `custom_conditions` | `list[str]` | `Custom conditions:` + each line |

> Removing a line = omitting. Do not duplicate keys in the same YAML map (`Map keys must be unique` error).

---

## 3. `models` block — Agents and Judge

Keys `agent_1`, `agent_2`, ... map in order onto the scenario `roles` (`experimento.py:172` `papeis_do_cenario`). `judge` is separate.

```yaml
models:
  agent_1:   # → scenario 1st role (e.g., candidate in salary_negotiation)
    provider: "ollama"         # ollama / ollama_local / gemini / openai / lmstudio / deepseek / openrouter
    name: "gemma4:31b-cloud"   # model name on the provider
    temperature: 0.5           # 0.0 deterministic — 1.0 creative
    max_tokens: 2048           # generation limit (4096 avoids cutoffs)
    persona:                   # Big Five — bipolar
      agreeableness: positive  # positive / negative / none (or null/~)
      neuroticism: negative
      extraversion: positive
      openness: none           # disables
      conscientiousness: positive
      extra_instructions: "Extra free text"
    tactics:                   # binary enabled/disabled (legacy 1-5 still ok)
      anchoring: enabled
      loss_aversion: enabled
      conditional_concession: enabled
      value_creation: enabled
      clarity: enabled
      fact_justification: enabled

  agent_2:   # → 2nd role (e.g., recruiter)
    provider: "gemini"
    name: "gemma-4-26b-a4b-it"
    temperature: 0.5
    max_tokens: 2048
    persona:
      agreeableness: positive
      # ... same
    tactics:
      anchoring: 2
      # ...

  judge:
    provider: "ollama"
    name: "gpt-oss:120b-cloud"
    temperature: 0.0
    metrics:                   # which dimensions the judge evaluates (1 call per 2-turn round, no history)
      - "agreeableness"        # Big Five: openness, conscientiousness, extraversion, agreeableness, neuroticism
      - "neuroticism"
      - "extraversion"
      - "openness"
      - "conscientiousness"
      - "anchoring"            # Tactics: anchoring, conditional_concession, value_creation
      - "clarity"              # Argumentation: fact_justification, clarity
      - "fact_justification"
      - "loss_aversion"        # Biases: loss_aversion
      - "value_creation"
```

### 3.1 `provider` → Adapter (`adapters/`)

| `provider` | Class (`adapters/__init__.py`) | `base_url` / API key |
|---|---|---|
| `ollama` | `OllamaAdapter` (cloud) | `OLLAMA_BASE_URL` + `OLLAMA_API_KEY` |
| `ollama_local` | `OllamaLocalAdapter` | fixed `http://localhost:11434` |
| `gemini` | `GeminiAdapter` | `GEMINI_API_KEY` |
| `openai` | `OpenAIAdapter` | `OPENAI_API_KEY` |
| `lmstudio` | `LMStudioAdapter` | `http://localhost:1234/v1` |
| `deepseek` | `DeepSeekAdapter` | `DEEPSEEK_API_KEY` |
| `openrouter` | `OpenRouterAdapter` | `OPENROUTER_API_KEY` |

### 3.2 `persona` — bipolar Big Five (`persona/big5_persona.py:76`)

| Dimension | `positive` (high) | `negative` (low) | `none` |
|---|---|---|---|
| `agreeableness` | Cooperative/pro-social | Competitive/adversarial | disables |
| `conscientiousness` | Organized/precise | Impulsive/vague | disables |
| `neuroticism` | Unstable/reactive (reversed IV: `positive`=unstable) | Stable/composed | disables |
| `extraversion` | Assertive/dominant | Passive/reserved | disables |
| `openness` | Creative/integrative | Rigid/conventional | disables |

Guides in `_GUIDANCE:76`; names in `_DIM_NAMES`. `none`/`null`/`nil`/`~`/omitting = not injected.

### 3.3 `tactics` — `PRESENT`/`ABSENT`/`NOT_APPLICABLE` (`persona/tactics_builder.py:14`, `scoring/negotiation_metrics.py:1`)

| Metric | `ABSENT` / `disabled` | `PRESENT` / `enabled` | `NOT_APPLICABLE` |
|---|---|---|---|
| `anchoring` | does not inject | Strong anchor | does not inject (no opportunity) |
| `conditional_concession` | does not inject | Strict exchanges | — |
| `value_creation` | does not inject | Integrative/creative | — |
| `clarity` | does not inject | Structured/mathematical | — |
| `loss_aversion` | does not inject (immune) | Loss-reactive | — |
| `fact_justification` | does not inject | Highly grounded | — |

`PRESENT` = behavior observed in the turn; `ABSENT` = opportunity existed but absent; `NOT_APPLICABLE` = turn without enough opportunity (excluded from the denominator) and must carry short `evidence`. Alias `enabled→PRESENT`, `disabled/absent→ABSENT`, `none/not_applicable→ignored`. Legacy `1-5` still works (`1-2→ABSENT`, `4-5→PRESENT`) for compatibility. All 10 `configs/*.yaml` ship the 6 metrics `present`/`enabled` on both agents and `judge.metrics` with the 11 dimensions, and `experimento.py:272` evaluates **utility** (continuous 0-1 `utility`) and **satisfaction** (1-7 PSI) separately.

**Batched categorical observation per round (same base):** the same `NEGOTIATION_META` used to induce is used to judge. The judge (`scoring/evaluator.py` `_JUDGE_SYSTEM_ROUND`) receives **per round (2 consecutive turns, one per agent), in 1 call**, only the two complete responses + rubrics for **all** metrics listed in `judge.metrics` — no resent history, no summary. Returns `{"turn_evaluations": [{"turn_index": i, "evaluations": {"anchoring": {"result": "PRESENT", "evidence": "..."}, ...}}, ...]}` (`N` turns = `N/2` calls). Later aggregation: `occurrence_rate = PRESENT / (PRESENT + ABSENT)` over applicable turns (`NOT_APPLICABLE` ignored) → `65% (13/20; 5 NA)`, enabling `persistence_rate/first/last_occurrence` over the `Turn 1:0, Turn 2:N/A...` sequence. Big Five follows Goldberg (`positive→PRESENT`, `negative→ABSENT`); tactics `enabled→PRESENT`. E.g., `Induced PRESENT` vs `Observed 65%` → `✅ Compatible` (`≥50%`), `20%` → `❌` (`report/generator.py:250`).

### 3.4 `judge.metrics` (batched behavioral per round, no history)

Free list of Big Five + `NegotiationMetric` (`scoring/negotiation_metrics.py:42`). When omitted, evaluates Big Five only. Per round, **a single call** evaluates all metrics of both turns: `Turn 2k` + `Turn 2k+1` (complete responses, each is the other's context) + `PRESENT/ABSENT` rubrics (Goldberg for Big Five). Returns `{"turn_evaluations": [{"turn_index": i, "evaluations": {metric: {result: PRESENT|ABSENT|NOT_APPLICABLE, evidence}}}]}`. Behavioral metrics do **not** use immediate averaging: later `occurrence_rate` (`NOT_APPLICABLE` ignored) → `65%`. `utility` (continuous 0-1, `utility.py:147`) and `satisfaction` (ordinal 1-7, `satisfaction.py:48`) are separate; `agreement` is `AGREEMENT|NO_AGREEMENT`. The report splits `3 Behavioral Metrics | 5 Utility | 6 Subjective`; per-`turn×metric` `evidence` persists in `scores.jsonl` (`storage/jsonl_store.py:89`).

---

## 4. `utility` block — inside the agent, after `tactics` (no `currency`/`unit`/`role_type`)

```yaml
models:
  agent_1:
    tactics:
      anchoring: present
    utility:
      p_target: 3600   # target (USD)
      p_floor: 3100    # limit (BATNA)
  agent_2:
    tactics:
      anchoring: absent   # no anchor → negotiates freely
    utility:
      p_target: 2800
      p_floor: 3300
```

Formulas `scoring/utility.py:8`: `u_s(p)=(p-p_s)/(p̄_s-p_s)`, `u_b(p)=(p̄_b-p)/(p̄_b-p_b)`. When absent, the `5. Utility` section is omitted.

> Conditional anchor (`experimento.py`, `simulation/engine.py`): the `salary_negotiation`, `company_acquisition` and `vga_purchase` scenarios carry **no values** in prompts. Values (`p_target`/`p_floor`) are only injected into the `system` of the agent whose `tactics.anchoring` is active (`present`/`enabled`, legacy `>=4`) — the `[YOUR PRIVATE VALUE REFERENCES]` block. With `anchoring: absent/disabled`, the agent negotiates freely, with no numbers.

---

## 5. Available scenarios (`scenarios/__init__.py:56`)

| `scenario` | Description | `roles` (order) | `settlement_keywords` | `max_turns` |
|---|---|---|---|---|
| `salary_negotiation` | Salary negotiation: engineer vs company (values via `utility.p_target/p_floor`, bonus/vacation) | `candidate` → `recruiter` | `SIMULACAO_CONCLUIDA`, `ACORDO_FECHADO` | 8 |
| `company_acquisition` | Company acquisition: founder vs buyer (values via `utility`, earn-out, IP) | `seller` → `buyer` | `SIMULACAO_CONCLUIDA`, `ACORDO_FECHADO` | 10 |
| `strategic_supplier_contract` | Supplier contract: buyer vs supplier ($240 vs $190, volume/terms) | `buyer` → `supplier` | `SIMULACAO_CONCLUIDA`, `ACORDO_FECHADO` | 10 |
| `property_boundary_dispute` | Property dispute: owner_a vs owner_b (12 sqm, $16k, wall) | `owner_a` → `owner_b` | `SIMULACAO_CONCLUIDA`, `ACORDO_FECHADO` | 10 |
| `vga_purchase` | GPU purchase: seller vs buyer (values via `utility`, bundles) | `seller` → `buyer` | `SIMULACAO_CONCLUIDA`, `ACORDO_FECHADO` | 8 |
| *custom* | Create in `scenarios/__init__.py:228` `NegotiationScenario(name=..., roles={...}, opening_role=..., max_turns=...)` and register in `SCENARIO_REGISTRY:175` | — | — | — |

Settlement only when **both** confirm agreement (`simulation/engine.py:271`).

> No fixed opening prompt: scenarios have no `opening_prompt`. The `opening_role` freely generates `Turn 0` from `roles[role]` + persona + context.

> `max_turns` = **rounds** (1 round = 1 utterance per agent). E.g., `max_turns: 4` with 2 agents = 8 utterances (`Turn 0..7`). When exhausted without both-side agreement, the simulation ends as `NO_AGREEMENT` (`metadata.ended_by = "turn_limit"`, `[Limit]` message in the terminal and `Ending: turn limit` in the report).

---

## 6. Minimal example per context

```yaml
# configs/estagflacao.yaml
experiment: {name: estagflacao, scenario: salary_negotiation, max_turns: 8}
context: {preset: estagflacao}
models:
  agent_1: {provider: ollama, name: "gemma4:31b-cloud", persona: {agreeableness: positive}}
  agent_2: {provider: gemini, name: "gemma-4-26b-a4b-it", persona: {agreeableness: negative}}
  judge: {provider: ollama, name: "gpt-oss:120b-cloud", metrics: [agreeableness, anchoring]}
```

All 10 presets ship ready-made in `configs/*.yaml`.
