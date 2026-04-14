"""
Simulation engine for multi-turn LLM negotiations.

Supports two modes:
  1. Agent-vs-Agent: two LLM adapters take turns (fully automated)
  2. Benchmark: one LLM agent responds to a fixed sequence of prompts

Personas
--------
Each agent can optionally receive a Big5Persona that injects behavioral
instructions into its system prompt before the negotiation starts.
Pass personas via the `personas` dict in SimulationEngine:

    engine = SimulationEngine(
        scenario=SALARY_NEGOTIATION,
        agents={
            "candidate": OllamaAdapter("llama3.1:8b"),
            "recruiter": OpenAIAdapter("gpt-4o"),
        },
        personas={
            "candidate": Big5Persona(agreeableness=5, neuroticism=1),
            "recruiter": Big5Persona(agreeableness=1, extraversion=5),
        },
    )

Personas are stored in NegotiationResult.metadata["personas"] for
full traceability in the dataset.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from ..adapters.base import LLMAdapter
from ..scenarios import NegotiationScenario
from ..persona import Big5Persona, PersonaPromptBuilder
from ..context import SituationalContext, ContextPromptBuilder

logger = logging.getLogger(__name__)

_persona_builder = PersonaPromptBuilder()
_context_builder = ContextPromptBuilder()


@dataclass
class Turn:
    """A single utterance in the negotiation."""
    turn_index: int
    agent_id: str
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    latency_ms: Optional[float] = None


@dataclass
class NegotiationResult:
    """
    Complete output of a simulated negotiation.

    This is the central data object passed to Evaluator and StorageManager.
    """
    run_id: str
    scenario_name: str
    scenario_description: str
    scenario_context: str
    agents: dict[str, str]           # {agent_id: model_identifier}
    agent_roles: dict[str, str]      # {agent_id: role_name}
    transcript: list[Turn]
    settled: bool                    # True if settlement detected
    total_turns: int
    started_at: float
    ended_at: float
    metadata: dict = field(default_factory=dict)
    # metadata["personas"] = {role: {dim: score}} when personas are used

    @property
    def duration_seconds(self) -> float:
        return self.ended_at - self.started_at

    def to_messages(self) -> list[dict]:
        """Convert transcript to OpenAI-style message list for downstream use."""
        return [
            {"role": t.role, "agent_id": t.agent_id, "content": t.content}
            for t in self.transcript
        ]


class NegotiationAgent:
    """
    Wraps an LLMAdapter with a negotiation role, conversation history,
    and an optional Big5Persona injected into the system prompt.
    """

    def __init__(
        self,
        agent_id: str,
        role: str,
        system_prompt: str,
        adapter: LLMAdapter,
        persona: Optional[Big5Persona] = None,
        context: Optional[SituationalContext] = None,
    ):
        self.agent_id = agent_id
        self.role = role
        self.adapter = adapter
        self.persona = persona
        self.context = context

        # Inject persona block
        prompt = system_prompt
        if persona is not None:
            prompt = _persona_builder.inject(prompt, persona)
            logger.debug("Persona injected for agent '%s': %s", agent_id, persona.to_dict())

        # Inject situational context block
        if context is not None:
            prompt = _context_builder.inject(prompt, context)
            if context.is_active():
                logger.debug("Context injected for agent '%s'", agent_id)

        self._system = prompt
        self._history: list[dict] = []

    def receive(self, speaker_role: str, content: str):
        """Record a message from the other party."""
        self._history.append({"role": "user", "content": f"[{speaker_role}]: {content}"})

    def speak(self, context_hint: str = "") -> tuple[str, float]:
        """Generate next utterance. Returns (content, latency_ms)."""
        messages = [{"role": "system", "content": self._system}]
        if context_hint:
            messages[0]["content"] += f"\n\nContext: {context_hint}"
        messages.extend(self._history)

        start = time.time()
        content = self.adapter.complete(messages)
        latency_ms = (time.time() - start) * 1000

        self._history.append({"role": "assistant", "content": content})
        return content, latency_ms


class SimulationEngine:
    """
    Orchestrates a negotiation between two agents or in benchmark mode.

    Args:
        scenario:        The NegotiationScenario to simulate.
        agents:          Dict mapping role_name → LLMAdapter.
        personas:        Optional dict mapping role_name → Big5Persona.
                         Roles not listed receive no persona (model default).
        context:         Optional SituationalContext injected into ALL agents.
                         Pass SituationalContext.disabled() or simply omit to
                         disable context injection. The same context is shared
                         by all agents (both parties are aware of the conditions).
        benchmark_turns: If provided, these are fixed prompts sent to the
                         single agent (benchmark mode).
    """

    def __init__(
        self,
        scenario: NegotiationScenario,
        agents: dict[str, LLMAdapter],
        personas: Optional[dict[str, Big5Persona]] = None,
        context: Optional[SituationalContext] = None,
        benchmark_turns: Optional[list[str]] = None,
        turn_delay_seconds: float = 0.0,
    ):
        self.scenario = scenario
        self.raw_agents = agents
        self.personas = personas or {}
        self.context = context
        self.benchmark_turns = benchmark_turns
        self.turn_delay_seconds = turn_delay_seconds

    def run(self) -> NegotiationResult:
        if self.benchmark_turns is not None:
            return self._run_benchmark()
        return self._run_agent_vs_agent()

    # ------------------------------------------------------------------
    # Agent-vs-Agent mode
    # ------------------------------------------------------------------

    def _run_agent_vs_agent(self) -> NegotiationResult:
        scenario = self.scenario
        run_id = str(uuid.uuid4())[:8]
        started_at = time.time()
        transcript: list[Turn] = []
        settled = False

        agents: dict[str, NegotiationAgent] = {}
        agent_roles: dict[str, str] = {}

        for role, adapter in self.raw_agents.items():
            agent_id = f"{role}_{adapter.model.replace(':', '-')}"
            agents[role] = NegotiationAgent(
                agent_id=agent_id,
                role=role,
                system_prompt=scenario.roles[role],
                adapter=adapter,
                persona=self.personas.get(role),
                context=self.context,
            )
            agent_roles[agent_id] = role

        role_order = list(scenario.roles.keys())
        if role_order[0] != scenario.opening_role:
            role_order = [scenario.opening_role] + [r for r in role_order if r != scenario.opening_role]

        turn_index = 0

        if scenario.opening_prompt:
            opening_role = scenario.opening_role
            opening_agent = agents[opening_role]
            transcript.append(Turn(
                turn_index=turn_index,
                agent_id=opening_agent.agent_id,
                role=opening_role,
                content=scenario.opening_prompt,
            ))
            for role, agent in agents.items():
                if role != opening_role:
                    agent.receive(opening_role, scenario.opening_prompt)
            opening_agent._history.append({"role": "assistant", "content": scenario.opening_prompt})
            turn_index += 1

        for _ in range(scenario.max_turns):
            for role in role_order:
                if turn_index == 0 and scenario.opening_prompt:
                    continue

                agent = agents[role]
                content, latency = agent.speak(
                    context_hint=scenario.shared_context if turn_index <= 1 else ""
                )

                turn = Turn(
                    turn_index=turn_index,
                    agent_id=agent.agent_id,
                    role=role,
                    content=content,
                    latency_ms=latency,
                )
                transcript.append(turn)
                logger.info("[Turn %d | %s] %s", turn_index, role, content[:120])

                if self.turn_delay_seconds > 0:
                    logger.info("Aguardando %.1fs antes do próximo turno...", self.turn_delay_seconds)
                    time.sleep(self.turn_delay_seconds)

                for other_role, other_agent in agents.items():
                    if other_role != role:
                        other_agent.receive(role, content)

                if any(kw.lower() in content.lower() for kw in scenario.settlement_keywords):
                    settled = True
                    logger.info("Settlement detected at turn %d.", turn_index)
                    break

                turn_index += 1

            if settled:
                break

        # Serialize personas for metadata
        personas_meta = {
            role: persona.to_dict()
            for role, persona in self.personas.items()
        }
        context_meta = self.context.to_dict() if self.context else None

        return NegotiationResult(
            run_id=run_id,
            scenario_name=scenario.name,
            scenario_description=scenario.description,
            scenario_context=scenario.shared_context,
            agents={a.agent_id: a.adapter.identifier for a in agents.values()},
            agent_roles=agent_roles,
            transcript=transcript,
            settled=settled,
            total_turns=len(transcript),
            started_at=started_at,
            ended_at=time.time(),
            metadata={**scenario.metadata, "personas": personas_meta, "context": context_meta},
        )

    # ------------------------------------------------------------------
    # Benchmark mode
    # ------------------------------------------------------------------

    def _run_benchmark(self) -> NegotiationResult:
        scenario = self.scenario
        run_id = str(uuid.uuid4())[:8]
        started_at = time.time()
        transcript: list[Turn] = []

        assert len(self.raw_agents) == 1, "Benchmark mode requires exactly one agent."
        role, adapter = next(iter(self.raw_agents.items()))

        agent = NegotiationAgent(
            agent_id=f"{role}_{adapter.model.replace(':', '-')}",
            role=role,
            system_prompt=scenario.roles[role],
            adapter=adapter,
            persona=self.personas.get(role),
            context=self.context,
        )
        opponent_role = [r for r in scenario.roles if r != role][0]

        for i, prompt in enumerate(self.benchmark_turns):
            transcript.append(Turn(
                turn_index=i * 2,
                agent_id=f"benchmark_{opponent_role}",
                role=opponent_role,
                content=prompt,
            ))
            agent.receive(opponent_role, prompt)

            content, latency = agent.speak()
            transcript.append(Turn(
                turn_index=i * 2 + 1,
                agent_id=agent.agent_id,
                role=role,
                content=content,
                latency_ms=latency,
            ))
            logger.info("[Benchmark turn %d] %s", i, content[:120])

        personas_meta = {
            r: p.to_dict() for r, p in self.personas.items()
        }
        context_meta = self.context.to_dict() if self.context else None

        return NegotiationResult(
            run_id=run_id,
            scenario_name=scenario.name,
            scenario_description=scenario.description,
            scenario_context=scenario.shared_context,
            agents={agent.agent_id: adapter.identifier},
            agent_roles={agent.agent_id: role},
            transcript=transcript,
            settled=False,
            total_turns=len(transcript),
            started_at=started_at,
            ended_at=time.time(),
            metadata={**scenario.metadata, "mode": "benchmark", "personas": personas_meta, "context": context_meta},
        )
