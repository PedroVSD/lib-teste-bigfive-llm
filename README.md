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
├── documentacao.md         # Documentação estendida do projeto
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

---

## Adaptadores
A pasta de adaptadores contém todos os adaptadores os agentes. É nessa pasta que é possível adicionar novos modelos às simulações, sejam eles locais ou de APIs.

#### IPC:
Um ponto importante é que para o caso do ollama, como o mesmo possui suporte para rodar localmente e via API. Optei por ter adaptadores individuais. Por padrão o ollama se utiliza de URLs bases diferentes para o uso local ou API. Abaixo estão ambas.

* Ollama_cloud: base_url: "https://ollama.com/api/generate"
* Ollama_local: base_url:  http://localhost:11434/api/generate -d

---

## context/situational.py

Define o contexto e situação da simulação. Pode ser ativado e desativado no config.yaml.
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
  year: "2023"
  country: "Argentina"
  custom_conditions:
    - "O fornecedor principal ameaçou cancelar contratos anteriores."
    - "Só há orçamento aprovado para pagamentos parcelados."
```
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

Os valores de inflação e interesse, são definidos logo abaixo da declaração do enum, no dict correspondente a cada um.

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
            country="Brasil",
            custom_conditions=[
                "O mercado de venture capital está em baixa.",
                "A parte vendedora precisa demonstrar tração imediata aos investidores."
            ]
        )
```
---

## scenarios/\_\_init\_\_.py

É possível criar ou modificar os cenários da simulação.

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
        PROCUREMENT_NEGOTIATION, 
        HOSTAGE_CRISIS_DEBRIEF,
        CENARIO_PERSONALIZADO  # <- Seu cenário personalizado
    ]
}
```
---

## persona/big5_persona.py

Onde ocorre toda a configuração de injeção de traços de personalidade das LLMs.

O modelo big5 se baseia em cindo traços dde personalidade, sendo eles:

| Dimensão | Polo alto | Polo baixo | Descrição |
|----------|-----------|------------|-----------------|
| **Agreeableness** | Cooperativo / Pró-social | Competitivo / Adversarial | Cooperação e compaixão |
| **Conscientiousness** | Organizado / Preciso | Impulsivo / Vago | É o nível organizacional, disciplina e orientação a objetivos. |
| **Neuroticism** | Instável / Reativo | Estável / Composto | Como é a reação a emoções negativas |
| **Extraversion** | Assertivo / Dominante | Passivo / Reservado | A busca por estímulos sociais e assertividade(habilidade social de se expressar de forma clara, direta e honesta). |
| **Openness** | Criativo / Integrativo | Rígido / Convencional | É a abertura a novas experiências |

Os níveis vão de 1 até 5, sendo:

* 1: "a very low level of"
* 2: "a low level of"
* 3: "a moderate level of"
* 4: "a high level of"
* 5: "a very high level of"

##### IPC: As instruções do que representa cada valor, podem ser alteradas no dicionário _GUIDANCE

**Toda e qualquer alteração para personalidades pode ser feita no arquivo big5_persona.py**

Os traços de personalidades são enviados ao modelo posteriormente a negociação se iniciar.

---

## scoring/evaluator

Como mencionado antes, a avaliação da negociação é feita por uma ou duas LLMs, que ao final da negociação verifica como foi o andamento e dá as "notas" para algumas classes das avaliações. Um ponto importante é que, as métricas de utilidade e satisfação **NÃO** são de responsabilidade do juiz, e sim de quem utiliza a biblioteca para o experimento.

Abaixo segue como é feita a avaliação realizada pelo juiz.

1. Avaliação Granular (Frase por Frase)
O motor não envia a transcrição inteira para o juiz. A função evaluate_turn isola uma única fala (utterance) de um agente de cada vez. Se um agente falou 5 vezes durante a simulação, o juiz avaliará esse agente 5 vezes separadas. Além disso, a avaliação é isolada por métrica: para uma mesma frase, o juiz é consultado individualmente para cada dimensão exigida (ex: uma consulta para Ancoragem, outra para Amabilidade, etc).

2. O "Gabarito" de Correção (Behavioral Anchors)
Para que o juiz (que é um LLM) não use seus próprios critérios subjetivos, o sistema injeta um "gabarito" estrito no prompt. Esse gabarito vem dos dicionários BIG5_META e NEGOTIATION_META.
Quando o juiz vai avaliar a "Firmeza na Oferta Inicial" (Anchoring), por exemplo, o código extrai as Âncoras Comportamentais (Behavioral Anchors) específicas daquela métrica e envia para o modelo, explicando exatamente o que significa tirar nota 1, nota 3 e nota 5.

3. A Construção do Prompt (_JUDGE_USER)
Para cada frase avaliada, a função _score_one monta um prompt contextualizado. O juiz recebe:

* O contexto do cenário (para entender o que está sendo negociado).
* O papel de quem está falando (ex: cliente ou vendedor).
* O nome da métrica e seus extremos (ex: "Firmeza na Oferta Inicial: Âncora Forte ↔ Cede Rapidamente").
* O texto exato das réguas de nota 1, 3 e 5.
* A frase exata dita pelo agente naquele turno.

4. Resposta em JSON
* O prompt de sistema do juiz (_JUDGE_SYSTEM) é desenhado para proibir tagarelice. Ele obriga o LLM a responder exclusivamente com um objeto JSON contendo três chaves:
* score: Um número inteiro de 1 a 5.
* justification: Uma explicação curta (1 a 3 frases) referenciando palavras específicas usadas pelo agente para provar o porquê da nota.
* confidence: O nível de confiança da IA naquela avaliação (de 0.0 a 1.0).

5. O Boletim Final (Média Matemática)
Após avaliar todos os turnos da conversa, o método evaluate_transcript é executado para a nota. Ele pega todas as notas que um agente tirou ao longo do tempo para uma métrica específica e calcula a média aritmética (sum(dim_scores) / len(dim_scores)).
Exemplo: Se o candidato tirou as notas 1, 3 e 5 em "Criação de Valor" durante os três turnos que falou, a nota final dele no relatório (o Big5Profile.scores) será 3.00.

6. Opcional: Duplo Juiz
É possível instanciar a classe Evaluator passando um second_judge (um segundo modelo LLM), o sistema fará com que os dois juízes avaliem a mesma frase independentemente. O código então calcula o IRR (Confiabilidade Interavaliadores), substituindo a métrica de "confiança" padrão por um cálculo matemático que reflete o quanto os dois modelos concordaram entre si (1.0 - abs(score1 - score2) / 4.0).

#### Confiabilidade inter-avaliadores (IRR)

Quando `second_judge` é fornecido, o campo `confidence` de cada `DimensionScore` contém o IRR normalizado entre os dois juízes:

``` python
IRR = 1 - |score_juiz1 - score_juiz2| / 4
```

- `1.0` = concordância total
- `0.75` = diferença de 1 ponto (aceitável)
- `< 0.5` = desacordo significativo (verificar rubricas)

**Importante:** use o mesmo modelo-juiz em todos os runs de uma comparação. Trocar o juiz entre runs invalida a comparabilidade dos scores.

---
##  Métricas

Há várias métricas que são avaliadas com as simulações.

Sendo que elas foram separadas por categorias, sendo:

* Categoria 1: Táticas e Comportamento de Negociação
  * Essas métricas avaliam como o agente está negociando, independentemente da sua personalidade.
* Categoria 2: Inteligência Emocional e Relacionamento
  * Avaliando a forma como o agente constrói a relação e lida com a situação.
* Categoria 3: Argumentação Lógica e Uso de Dados
  * Avaliando a racionalidade por trás das propostas.
* Categoria 4: Vieses e Erros Cognitivos
  * Economia comportamental.
* Utilidade(métrica objetiva)
  * Avalia o quão efetivo foi o efeito âncora.
* Satisfação(métrica subjetiva)
  * Avalia como foi a satisfação final do agente ao final da negociação.

#### Utilidade

Utilidade do vendedor: $$u_s(p) = \frac{p - \underline{p_s}}{\overline{p_s} - \underline{p_s}}$$
Utilidade do comprador: $$u_b(p) = \frac{\overline{p_b} - p}{\overline{p_b} - \underline{p_b}}$$

* $p$ é o valor dado ao produto
* $\overline{p_s}$ representa o valor alvo do vendedor
* $\underline{p_s}$ representa o mínimo valor aceitável
* $\overline{p_b}$ representa o máximo valor aceitável
* $\underline{p_s}$ representa o valor alvo

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
    extra={"seed": 42}   # suportado pelo Ollama; OpenAI suporta via "seed"
)
```

---

## Referências

- Yin Jou Huang and Rafk Hadf (2024). *How Personality Traits Infuence Negotiation Outcomes? A Simulation based on Large Language Models.*
- Yoshiki Takenami ,Yin Jou Huang, Yugo Murawaki, Chenhui Chu (2025). *How Does Cognitive Bias Affect Large Language Models? A Case Study on the Anchoring Effect in Price Negotiation Simulations.*
- Minjun Ren, Wentao Xu (2025). *The Impact of Big Five Personality Traits on AI Agent Decision-Making in Public Spaces: A Social Simulation Study.*
- Aleksandra Sorokovikova, Natalia Fedorova, Sharwin Rezagholi, Ivan P. Yamshchikov (2024). *LLMs Simulate Big Five Personality Traits: Further Evidence.*
- Junhyuk Choi, Hyeonchu Park , Haemin Lee, Hyebeen Shin, Hyun Joung Jin, Bugeun Kim (2025).*Pay What LLM Wants: Can LLM Simulate Economics Experiment with 522 Real-human Persona?.*
- TVERSKY, A.; KAHNEMAN, D. *Judgment under Uncertainty: Heuristics and Biases. Science, v. 185, n. 4157, p. 1124-1131, 1974.*
- KAHNEMAN, D.; TVERSKY, A. *Prospect Theory: An Analysis of Decision under Risk. Econometrica, v. 47, n. 2, p. 263-291, 1979.*
- FISHER, R.; URY, W.; *PATTON, B. Getting to Yes: Negotiating Agreement Without Giving In. 2. ed*
- WALTON, R. E.; MCKERSIE, R. B. *A Behavioral Theory of Labor Negotiations: An Analysis of a Social Interaction System. Nova York: McGraw-Hill, 1965.*
- CIALDINI, R. B. *Influence: The Psychology of Persuasion. Nova York: Harper Business, 1984.*
- GOULDNER, A. W. *The Norm of Reciprocity: A Preliminary Statement. American Sociological Review, v. 25, n. 2, p. 161-178, 1960.*
- BARRY, B.; OLIVER, R. L. *Affect in Negotiation: A Model and Propositions. Organizational Behavior and Human Decision Processes, v. 67, n. 2, p. 127-143, 1996.*
- ALTER, A. L.; OPPENHEIMER, D. M. *Uniting the Tribes of Fluency to Form a Metacognitive Nation. Personality and Social Psychology Review, v. 13, n. 3, p. 219-235, 2009.*
- GRICE, H. P. Logic and conversation. In: COLE, P.; MORGAN, J. L. (Eds.). *Syntax and semantics: Vol. 3. Speech acts. Nova York: Academic Press, 1975. p. 41-58.*
