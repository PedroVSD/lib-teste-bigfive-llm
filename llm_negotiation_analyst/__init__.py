"""
llm_negotiation_analyst
========================
A library for studying Big Five personality expression in LLM negotiations.

Quick start — agent-vs-agent:

    from llm_negotiation_analyst import run_negotiation
    from llm_negotiation_analyst.adapters import OpenAIAdapter, OllamaAdapter
    from llm_negotiation_analyst.scenarios import SALARY_NEGOTIATION

    result, profiles, report = run_negotiation(
        scenario=SALARY_NEGOTIATION,
        agents={
            "candidate": OllamaAdapter("llama3.1:8b"),
            "recruiter": OpenAIAdapter("gpt-4o"),
        },
        judge=OpenAIAdapter("gpt-4o"),
        output_dir="results/",
    )
    print(report)
"""

from .adapters import LLMAdapter, OpenAIAdapter, OllamaAdapter, GeminiAdapter
from .scenarios import NegotiationScenario, SCENARIO_REGISTRY
from .simulation import SimulationEngine, NegotiationResult
from .scoring import Evaluator, EvaluatorConfig, Dimension, Big5Profile
from .storage import StorageManager
from .report import generate_report


def run_negotiation(
    scenario: NegotiationScenario,
    agents: dict[str, LLMAdapter],
    judge: LLMAdapter,
    output_dir: str = "results",
    evaluator_config=None,
    second_judge=None,
    personas=None,#parte referente à customização da personalidade
    context=None,#parte referente à customização do contexto
    turn_delay_seconds = 0.0,
    verbose: bool = True,
    use_system_reminder: bool = True,
    tactics=None,
    experiment_name=None,
):
    import logging
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Usa experiment_name do YAML (file stem) se não fornecido
    if not experiment_name:
        # tenta inferir de result depois, mas passa para engine para metadata
        experiment_name = None

    engine = SimulationEngine(
        scenario=scenario,
        agents=agents,
        personas=personas,
        context=context,
        turn_delay_seconds=turn_delay_seconds,
        use_system_reminder=use_system_reminder,
        tactics=tactics,
        experiment_name=experiment_name,
    )
    result = engine.run()
    # Se engine não tinha nome mas result tem, propaga
    if not experiment_name:
        experiment_name = result.metadata.get("experiment_name")

    evaluator = Evaluator(judge=judge, config=evaluator_config, second_judge=second_judge)
    profiles = evaluator.evaluate_transcript(
        transcript=result.to_messages(),
        agent_roles=result.agent_roles,
        scenario_context=result.scenario_context,
    )
    for agent_id, profile in profiles.items():
        profile.model_identifier = result.agents.get(agent_id, agent_id)

    storage = StorageManager(base_dir=output_dir)
    storage.save_result(result)
    storage.save_scores(result, profiles)

    # Nome do arquivo inclui experiment_name se disponível
    prefix = f"{experiment_name}_{result.scenario_name}" if experiment_name else result.scenario_name
    report_path = f"{output_dir}/{prefix}_{result.run_id}_report.md"
    report_md = generate_report(result, profiles, output_path=report_path)
    return result, profiles, report_md


def run_benchmark(
    scenario: NegotiationScenario,
    agent_role: str,
    agent_adapter: LLMAdapter,
    benchmark_turns: list,
    judge: LLMAdapter,
    output_dir: str = "results",
    evaluator_config=None,
    verbose: bool = True,
):
    import logging
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    engine = SimulationEngine(
        scenario=scenario,
        agents={agent_role: agent_adapter},
        benchmark_turns=benchmark_turns,
    )
    result = engine.run()

    evaluator = Evaluator(judge=judge, config=evaluator_config)
    profiles = evaluator.evaluate_transcript(
        transcript=result.to_messages(),
        agent_roles=result.agent_roles,
        scenario_context=result.scenario_context,
    )
    for agent_id, profile in profiles.items():
        profile.model_identifier = result.agents.get(agent_id, agent_id)

    storage = StorageManager(base_dir=output_dir)
    storage.save_result(result)
    storage.save_scores(result, profiles)

    report_path = f"{output_dir}/{result.scenario_name}_{result.run_id}_report.md"
    report_md = generate_report(result, profiles, output_path=report_path)
    return result, profiles, report_md
