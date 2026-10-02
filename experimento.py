import os
import pathlib
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
from llm_negotiation_analyst.context import (
    SituationalContext, InflationLevel, InterestRateLevel,
    GovernmentOrientation, CrisisType,
)
from llm_negotiation_analyst.scoring.evaluator import EvaluatorConfig
from llm_negotiation_analyst.scoring.utility import UtilityCalculator, RoleUtilityParams
from llm_negotiation_analyst.scoring.satisfaction import SatisfactionEvaluator
from llm_negotiation_analyst.report.generator import generate_report

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def load_config(filepath: str):
    with open(filepath, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    # Experiment name = YAML filename (without extension)
    # E.g., `estagflacao.yaml` → experiment.name = "estagflacao" (always overrides the YAML)
    # display_name/title with spaces (e.g., "test with X variable") is preserved in display_name/yaml_name
    try:
        from pathlib import Path
        file_stem = Path(filepath).stem
        exp = data.get("experiment") or {}
        yaml_name = exp.get("name")
        # explicit title/display_name/label takes priority as the human name
        display_raw = exp.get("title") or exp.get("display_name") or exp.get("label") or yaml_name
        exp["name"] = file_stem
        if display_raw and str(display_raw).strip() and str(display_raw).strip() != file_stem:
            exp["display_name"] = str(display_raw).strip()
            exp["yaml_name"] = str(display_raw).strip()  # compat
        elif yaml_name and yaml_name != file_stem:
            exp["yaml_name"] = yaml_name
            exp["display_name"] = yaml_name
        exp["config_file"] = filepath
        data["experiment"] = exp
    except Exception:
        pass
    return data


def create_adapter(config_dict: dict):
    provider   = config_dict["provider"].lower()
    model_name = config_dict["name"]
    temp       = config_dict.get("temperature", 0.5)
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
        return OllamaAdapter(model=model_name, base_url=config_dict.get("base_url"), config=config_obj)
    elif provider == "deepseek":
        return DeepSeekAdapter(model=model_name, base_url=config_dict.get("base_url"), config=config_obj)
    elif provider == "openrouter":
        from llm_negotiation_analyst.adapters.openrouter_adapter import OpenRouterAdapter
        return OpenRouterAdapter(
            model=model_name,
            api_key=config_dict.get("api_key"),
            base_url=config_dict.get("base_url"),
            referer=config_dict.get("referer"),
            title=config_dict.get("title"),
            config=config_obj,
        )
    raise ValueError(f"Unknown provider: {provider}. Options: gemini, openai, lmstudio, ollama, ollama_local, deepseek, openrouter")


def parse_persona(agent_config: dict) -> Big5Persona | None:
    persona_dict = agent_config.get("persona") or {}

    if not persona_dict:
        return None

    chaves_big5 = {"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"}
    # 'none'/'null'/'nil' (string) also disables; None already filters but we still
    # pass it through so Big5Persona normalizes it to None (omitted trait)
    filtered_persona = {}
    for k, v in persona_dict.items():
        if k not in chaves_big5:
            continue
        # keep 'none' as a string for normalization -> None; real None is already disabled
        filtered_persona[k] = v

    instrucoes_originais = persona_dict.get("extra_instructions", "")
    # Tactics are NOT injected into the Agent Prompt — they are observational only (Judge).
    # Do not merge tactics into extra_instructions.
    if instrucoes_originais:
        filtered_persona["extra_instructions"] = instrucoes_originais

    # When there were only empty extra_instructions and no Big Five, return None
    if not filtered_persona:
        return None

    return Big5Persona(**filtered_persona)


def parse_context(context_dict: dict) -> SituationalContext:
    if not context_dict or not context_dict.get("enabled", True):
        return SituationalContext.disabled()

    # `preset:` support — loads one of the 10 ready contexts and allows field overrides
    from llm_negotiation_analyst.context import ContextPresets
    import unicodedata
    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode()
        return s.lower().replace("-", "_").replace(" ", "_")
    _PRESET_MAP = {
        "crescimento_forte": ContextPresets.crescimento_forte,
        "crescimento_economico_forte": ContextPresets.crescimento_forte,
        "expansao": ContextPresets.crescimento_forte,
        "recessao": ContextPresets.recessao,
        "recessao_economica": ContextPresets.recessao,
        "estagflacao": ContextPresets.estagflacao,
        "boom_inflacionario": ContextPresets.boom_inflacionario,
        "crise_financeira": ContextPresets.crise_financeira,
        "crise_politica": ContextPresets.crise_politica,
        "governo_intervencionista": ContextPresets.governo_intervencionista,
        "intervencionista": ContextPresets.governo_intervencionista,
        "governo_liberal": ContextPresets.governo_liberal,
        "liberal": ContextPresets.governo_liberal,
        "governo_liberal_pro_mercado": ContextPresets.governo_liberal,
        "crise_desemprego": ContextPresets.crise_desemprego,
        "crise_emprego": ContextPresets.crise_desemprego,
        "anarcho_capitalist": ContextPresets.anarcho_capitalist,
        "anarco_capitalista": ContextPresets.anarcho_capitalist,
        "anarchocapitalist": ContextPresets.anarcho_capitalist,
    }
    preset_name = context_dict.get("preset")
    if preset_name:
        key = _norm(preset_name)
        factory = _PRESET_MAP.get(key)
        if not factory:
            raise ValueError(f"Unknown context preset: '{preset_name}'. Options: {list(_PRESET_MAP.keys())}")
        base = factory()
        # Override with manual fields when provided
        if context_dict.get("inflation"):
            base.inflation = getattr(InflationLevel, context_dict["inflation"])
        if context_dict.get("interest_rates"):
            base.interest_rates = getattr(InterestRateLevel, context_dict["interest_rates"])
        if context_dict.get("government"):
            base.government = getattr(GovernmentOrientation, context_dict["government"])
        if context_dict.get("crises") is not None:
            base.crises = [getattr(CrisisType, c) for c in context_dict.get("crises", []) if c]
        if context_dict.get("gdp_growth") is not None:
            base.gdp_growth = context_dict.get("gdp_growth")
        if context_dict.get("unemployment") is not None:
            base.unemployment = context_dict.get("unemployment")
        if context_dict.get("custom_conditions") is not None:
            extra = [c for c in context_dict.get("custom_conditions") or [] if c]
            # When the preset already has custom_conditions, append
            base.custom_conditions = list(base.custom_conditions) + extra
        return base

    return SituationalContext(
        enabled=True,
        inflation=getattr(InflationLevel, context_dict["inflation"]) if context_dict.get("inflation") else None,
        interest_rates=getattr(InterestRateLevel, context_dict["interest_rates"]) if context_dict.get("interest_rates") else None,
        government=getattr(GovernmentOrientation, context_dict["government"]) if context_dict.get("government") else None,
        crises=[getattr(CrisisType, c) for c in context_dict.get("crises", []) if c],
        gdp_growth=context_dict.get("gdp_growth"),
        unemployment=context_dict.get("unemployment"),
        custom_conditions=[c for c in context_dict.get("custom_conditions") or [] if c],
    )


def parse_utility_params(utility_cfg: dict, chaves_agentes: list[str], papeis: list[str],) -> dict[str, RoleUtilityParams]:
    """
    Read the 'utility' block (legacy top-level or new inside the agent).

    New format (preferred, inside the agent, after tactics):
        models:
          agent_1:
            utility:
              p_target: 18000
              p_floor: 15500
          agent_2:
            utility:
              p_target: 14000
              p_floor: 16500

    Legacy still supported:
        utility:
          agent_1: {p_target, p_floor, role_type?, currency?, unit?}
    """
    if not utility_cfg:
        return {}

    agent_to_role = {chave: role for chave, role in zip(chaves_agentes, papeis)}

    params = {}
    for key, cfg in utility_cfg.items():
        role = agent_to_role.get(key, key)
        # role_type optional, inferred when missing; currency/unit removed from yaml (legacy)
        rt = cfg.get("role_type")
        if not rt:
            # simple inference: both calculations are symmetric, default seller
            try:
                rt = "seller" if float(cfg["p_target"]) > float(cfg["p_floor"]) else "buyer"
            except Exception:
                rt = "seller"
        params[role] = RoleUtilityParams(
            role=role,
            role_type=rt,
            p_target=float(cfg["p_target"]),
            p_floor=float(cfg["p_floor"]),
            currency=cfg.get("currency", ""),
            unit=cfg.get("unit", ""),
        )
    return params


def parse_utility_from_agents(models_cfg: dict, chaves_agentes: list[str], papeis: list[str]) -> dict[str, RoleUtilityParams]:
    """New: read utility inside models.agent_x.utility (after tactics)."""
    agent_to_role = {chave: role for chave, role in zip(chaves_agentes, papeis)}
    params = {}
    for chave in chaves_agentes:
        role = agent_to_role.get(chave, chave)
        agent_cfg = models_cfg.get(chave, {}) if isinstance(models_cfg, dict) else {}
        util_cfg = agent_cfg.get("utility") if isinstance(agent_cfg, dict) else None
        if isinstance(util_cfg, dict) and "p_target" in util_cfg and "p_floor" in util_cfg:
            rt = util_cfg.get("role_type")
            if not rt:
                try:
                    rt = "seller" if float(util_cfg["p_target"]) > float(util_cfg["p_floor"]) else "buyer"
                except Exception:
                    rt = "seller"
            params[role] = RoleUtilityParams(
                role=role,
                role_type=rt,
                p_target=float(util_cfg["p_target"]),
                p_floor=float(util_cfg["p_floor"]),
                currency=util_cfg.get("currency", ""),
                unit=util_cfg.get("unit", ""),
            )
    return params


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)
    exp    = config["experiment"]
    print(f"Starting experiment: {exp['name']}...")

    # Scenario
    scenario_name = exp["scenario"]
    if scenario_name not in SCENARIO_REGISTRY:
        raise ValueError(f"Scenario '{scenario_name}' not found!")
    scenario = SCENARIO_REGISTRY[scenario_name]

    if "max_turns" in exp:
        scenario.max_turns = exp["max_turns"]
        print(f"Turn limit set to: {scenario.max_turns} rounds ({scenario.max_turns * len(scenario.roles)} utterances)")
    else:
        print(f"Turn limit: {scenario.max_turns} rounds ({scenario.max_turns * len(scenario.roles)} utterances) — max_turns = rounds (1 utterance per agent)")

    # 1. Judge(s)
    ag_judge = create_adapter(config["models"]["judge"])
    second_judge = None
    if "second_judge" in config["models"]:
        second_judge = create_adapter(config["models"]["second_judge"])
        print(f"Second judge enabled: {config['models']['second_judge'].get('name')} ({config['models']['second_judge'].get('provider')}) — IRR will be computed")
    elif "judge2" in config["models"]:
        second_judge = create_adapter(config["models"]["judge2"])
        print(f"Second judge enabled: {config['models']['judge2'].get('name')}")

    # 2. Agentes e personas
    papeis_do_cenario    = list(scenario.roles.keys())
    chaves_agentes_yaml  = [k for k in config["models"].keys() if k not in ("judge", "second_judge", "judge2")]

    if len(chaves_agentes_yaml) < len(papeis_do_cenario):
        raise ValueError("Not enough agents in the YAML for this scenario!")

    agents_dict  = {}
    personas_dict = {}
    tactics_dict = {}

    print("-" * 40)
    for i, role_name in enumerate(papeis_do_cenario):
        chave = chaves_agentes_yaml[i]
        agents_dict[role_name]  = create_adapter(config["models"][chave])
        personas_dict[role_name] = parse_persona(config["models"][chave])
        # Store tactics separately for the report (even when persona is None)
        raw_tactics = config["models"][chave].get("tactics") or {}
        # Filter none/null/nil and normalize
        tactics_dict[role_name] = {k: v for k, v in raw_tactics.items() if v is not None and str(v).lower() not in ("none","null","nil")}
        print(f"Role '{role_name.upper()}' <- '{chave}'")
    print("-" * 40)

    # 3. Macroeconomic context (+ experimental on/off switch with minimal context)
    context_cfg = config.get("context", {}) or {}
    macro_context = parse_context(context_cfg)
    macro_context_enabled = context_cfg.get("macro_context_enabled", True)
    if isinstance(macro_context_enabled, str):
        macro_context_enabled = macro_context_enabled.strip().lower() in ("true", "1", "yes", "on")
    if macro_context_enabled:
        print("Macroeconomic context: ENABLED")
    else:
        print("Macroeconomic context: DISABLED — agents negotiate on minimal context only")

    # 3.1 Utility (new: inside the agent after tactics) — parse BEFORE the simulation
    # to inject anchor values into the prompt when anchoring is active.
    utility_params = parse_utility_from_agents(config.get("models", {}), chaves_agentes_yaml, papeis_do_cenario)
    if not utility_params:
        legacy_cfg = config.get("utility", {})
        if legacy_cfg:
            utility_params = parse_utility_params(legacy_cfg, chaves_agentes_yaml, papeis_do_cenario)

    # 3.2 Anchor hints: only roles with active tactics.anchoring receive p_target/p_floor.
    def _is_anchor_active(v) -> bool:
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return float(v) >= 4
        s = str(v).strip().lower()
        if s in ("present", "enabled", "true", "on", "1", "yes", "active"):
            return True
        if s in ("absent", "disabled", "false", "off", "0", "no", "none", "inactive", "not_applicable", "null", "nil"):
            return False
        try:
            return float(s) >= 4
        except Exception:
            return False

    anchor_hints: dict[str, dict] = {}
    for role_name in papeis_do_cenario:
        tact = (tactics_dict.get(role_name) or {})
        if _is_anchor_active(tact.get("anchoring")) and role_name in utility_params:
            up = utility_params[role_name]
            anchor_hints[role_name] = {"p_target": float(up.p_target), "p_floor": float(up.p_floor)}
            print(f"Anchor ACTIVE for '{role_name.upper()}': target={up.p_target:g}, limit={up.p_floor:g}")
        else:
            print(f"Anchor inactive for '{role_name.upper()}': negotiating freely (no values in prompt)")

    # 3.3 Specific context: every role receives its own [Specific context] block
    # (reference values only when anchoring is active for that role).
    # Optional per-agent free text overrides the default role line:
    #   models:
    #     agent_1:
    #       specific_context: "You are the hiring manager. Defend the budget."
    specific_context_texts: dict[str, str] = {}
    agent_to_role_sc = {chave: role for chave, role in zip(chaves_agentes_yaml, papeis_do_cenario)}
    for chave in chaves_agentes_yaml:
        custom = (config["models"].get(chave) or {}).get("specific_context")
        if isinstance(custom, str) and custom.strip():
            role_sc = agent_to_role_sc.get(chave, chave)
            specific_context_texts[role_sc] = custom.strip()
            print(f"Specific context override for '{role_sc.upper()}' from YAML")

    # 4. Evaluator configuration (Big Five + negotiation metrics)
    metricas_textos = config["models"]["judge"].get("metrics", [])
    config_juiz = EvaluatorConfig.from_strings(metricas_textos) if metricas_textos else None

    # 4.1 BFI — questionnaire right after conditioning (before the negotiation)
    # Per-agent details are saved to *_bfi.json and the report; terminal shows only applying/done.
    # With macro disabled, BFI runs without macroeconomic context as well.
    bfi_results = None
    try:
        from llm_negotiation_analyst.scoring.bfi import run_bfi_all
        from llm_negotiation_analyst.context import SituationalContext as _SC
        print("Applying BFI-44 to agents...")
        bfi_results = run_bfi_all(agents_dict, personas_dict, macro_context if macro_context_enabled else _SC.disabled())
        print("BFI done.")
    except Exception as e:
        print(f"BFI failed (continuing without BFI): {e}")
        import traceback; traceback.print_exc()
        bfi_results = None

    # 5. Main simulation
    experiment_name = exp.get("name")
    display_name = exp.get("display_name") or exp.get("title") or exp.get("yaml_name")
    result, profiles, _ = run_negotiation(
        scenario=scenario,
        agents=agents_dict,
        judge=ag_judge,
        second_judge=second_judge,
        evaluator_config=config_juiz,
        turn_delay_seconds=exp.get("turn_delay_seconds", 0.0),
        personas=personas_dict,
        tactics=tactics_dict,
        context=macro_context,
        output_dir="results/",
        use_system_reminder=exp.get("use_system_reminder", False),
        experiment_name=experiment_name,
        experiment_display_name=display_name,
        anchor_hints=anchor_hints,
        macro_context_enabled=macro_context_enabled,
        specific_context_texts=specific_context_texts or None,
    )
    print("Simulation done.")

    # 6. Economic utility (uses utility_params already parsed in 3.1)
    utility_results = None
    if utility_params:
        print("Computing economic utility...")
        utility_calc    = UtilityCalculator(judge=ag_judge, role_params=utility_params)
        utility_results = utility_calc.evaluate(result)
        print("Utility computed.")

    # 7. Post-negotiation satisfaction — PSI (always runs)
    print("Evaluating satisfaction (PSI — 16 questions per agent)...")
    sat_evaluator        = SatisfactionEvaluator(judge=ag_judge)
    satisfaction_results = sat_evaluator.evaluate_all(result)
    print("Satisfaction evaluated.")

    # 8. Save BFI when present (for report and persistence)
    if 'bfi_results' in locals() and bfi_results:
        try:
            import json
            # Keep only agent_id entries (with '_') and dedupe
            bfi_to_save = {}
            for k, v in bfi_results.items():
                if "_" in k and k not in bfi_to_save:
                    try:
                        bfi_to_save[k] = v.to_dict()
                    except Exception:
                        bfi_to_save[k] = {"scores": getattr(v, "scores", {}), "raw_answers": getattr(v, "raw_answers", {})}
            result.metadata["bfi"] = bfi_to_save
            # Prefix for saving BFI (same prefix as the report)
            exp_name_tmp = result.metadata.get("experiment_name") or experiment_name or result.scenario_name
            display_tmp = result.metadata.get("experiment_display_name") or result.metadata.get("experiment_title") or result.metadata.get("yaml_name")
            if display_tmp:
                safe_tmp = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in str(display_tmp)).strip()
                prefix_tmp = f"{safe_tmp}_{exp_name_tmp}_{result.scenario_name}" if exp_name_tmp else f"{safe_tmp}_{result.scenario_name}"
            else:
                prefix_tmp = f"{exp_name_tmp}_{result.scenario_name}" if exp_name_tmp else result.scenario_name
            bfi_path = f"results/{prefix_tmp}_{result.run_id}_bfi.json"
            pathlib.Path("results").mkdir(parents=True, exist_ok=True)
            with open(bfi_path, "w", encoding="utf-8") as f:
                json.dump(bfi_to_save, f, ensure_ascii=False, indent=2)
            print(f"BFI saved: {bfi_path}")
        except Exception as e:
            print(f"BFI save failed: {e}")
            import traceback; traceback.print_exc()

    # 9. Regenerate the report with the new sections (uses display_name + experiment_name in the filename)
    exp_name = result.metadata.get("experiment_name") or experiment_name or result.scenario_name
    display = result.metadata.get("experiment_display_name") or result.metadata.get("experiment_title") or result.metadata.get("yaml_name")
    if display:
        safe_display = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in str(display)).strip()
        prefix = f"{safe_display}_{exp_name}_{result.scenario_name}" if exp_name else f"{safe_display}_{result.scenario_name}"
    else:
        prefix = f"{exp_name}_{result.scenario_name}" if exp_name else result.scenario_name
    report_path = f"results/{prefix}_{result.run_id}_report.md"
    # bfi_results may be dict agent_id -> BFIResult
    bfi_for_report = locals().get("bfi_results")
    generate_report(
        result=result,
        profiles=profiles,
        output_path=report_path,
        utility_results=utility_results,
        satisfaction_results=satisfaction_results,
        bfi_results=bfi_for_report,
    )

    ended_by = result.metadata.get("ended_by", "agreement" if result.settled else "turn_limit")
    if ended_by == "turn_limit" and not result.settled:
        print(f"\nExperiment done with no agreement (turn limit)! Report: {report_path}")
    else:
        print(f"\nExperiment done! Report: {report_path}")
