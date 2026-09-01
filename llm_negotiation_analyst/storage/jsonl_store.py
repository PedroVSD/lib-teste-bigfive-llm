"""
Storage layer for negotiation results and scores — categorical.

Behavioral: summaries {present,absent,not_applicable,occurrence_rate}
Outcomes: utility continuous, agreement categorical.
Subjective: satisfaction 1-7 separate.
"""

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ..simulation.engine import NegotiationResult
from ..scoring.big5 import Big5Profile, Dimension


class StorageManager:
    def __init__(self, base_dir: str = "results"):
        self.base = Path(base_dir)
        self._init_dirs()

    def _init_dirs(self):
        for sub in ["transcripts", "scores"]:
            (self.base / sub).mkdir(parents=True, exist_ok=True)

    def save_result(self, result: NegotiationResult) -> dict[str, Path]:
        exp_name = result.metadata.get("experiment_name") if result.metadata else None
        slug = f"{exp_name}_{result.scenario_name}_{result.run_id}" if exp_name else f"{result.scenario_name}_{result.run_id}"
        transcript_path = self.base / "transcripts" / f"{slug}.jsonl"
        with open(transcript_path, "w", encoding="utf-8") as f:
            for turn in result.transcript:
                line = {
                    "run_id": result.run_id,
                    "scenario": result.scenario_name,
                    "turn_index": turn.turn_index,
                    "agent_id": turn.agent_id,
                    "role": turn.role,
                    "content": turn.content,
                    "timestamp": turn.timestamp,
                    "latency_ms": turn.latency_ms,
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        index_path = self.base / "runs_index.jsonl"
        summary = {
            "run_id": result.run_id,
            "scenario": result.scenario_name,
            "agents": result.agents,
            "settled": result.settled,
            "total_turns": result.total_turns,
            "duration_seconds": round(result.duration_seconds, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "transcript_file": str(transcript_path.relative_to(self.base)),
            "metadata": result.metadata,
        }
        with open(index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

        return {"transcript": transcript_path, "index": index_path}

    def save_scores(
        self,
        result: NegotiationResult,
        profiles: dict[str, Big5Profile],
    ) -> Path:
        """Save categorical profiles (summaries + observations)."""
        exp_name = result.metadata.get("experiment_name") if result.metadata else None
        slug = f"{exp_name}_{result.scenario_name}_{result.run_id}" if exp_name else f"{result.scenario_name}_{result.run_id}"
        scores_path = self.base / "scores" / f"{slug}_scores.jsonl"

        with open(scores_path, "w", encoding="utf-8") as f:
            for agent_id, profile in profiles.items():
                # summaries categorical
                summaries_out = {}
                for metric, summ in profile.summaries.items():
                    key = metric.value if hasattr(metric, "value") else str(metric)
                    if hasattr(summ, "to_dict"):
                        summaries_out[key] = summ.to_dict()
                    else:
                        summaries_out[key] = summ
                # also include legacy scores (occurrence_rate) for compat
                scores_out = {}
                for k, v in profile.scores.items():
                    key = k.value if hasattr(k, "value") else str(k)
                    scores_out[key] = v

                observations_out = []
                source = profile.observations  # canonical; per_turn_scores is property alias
                # fallback for legacy profiles that might have only per_turn_scores
                if not source and hasattr(profile, "per_turn_scores"):
                    try:
                        source = profile.per_turn_scores  # type: ignore
                    except Exception:
                        source = []
                for o in source:
                    # handle both new BehaviorObservation and legacy DimensionScore
                    dim_val = o.dimension.value if hasattr(o.dimension, "value") else str(o.dimension)
                    result_val = o.result.value if hasattr(o.result, "value") else getattr(o, "score", None)
                    # legacy score -> map to PRESENT/ABSENT approx
                    if not hasattr(o, "result"):
                        result_val = "PRESENT" if o.score >= 3 else "ABSENT"
                    observations_out.append({
                        "dimension": dim_val,
                        "result": result_val if isinstance(result_val, str) else result_val.value,
                        "evidence": getattr(o, "evidence", getattr(o, "justification", "")),
                        "turn_index": o.turn_index,
                        "confidence": o.confidence,
                    })

                line = {
                    "run_id": result.run_id,
                    "scenario": result.scenario_name,
                    "agent_id": agent_id,
                    "model_identifier": profile.model_identifier,
                    "role": result.agent_roles.get(agent_id, "unknown"),
                    "summaries": summaries_out,
                    "scores": scores_out,
                    "observations": observations_out,
                    "notes": profile.notes,
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        return scores_path

    def load_transcript(self, run_id: str, scenario: str) -> list[dict]:
        path = self.base / "transcripts" / f"{scenario}_{run_id}.jsonl"
        return self._read_jsonl(path)

    def load_scores(self, run_id: str, scenario: str) -> list[dict]:
        path = self.base / "scores" / f"{scenario}_{run_id}_scores.jsonl"
        return self._read_jsonl(path)

    def list_runs(self) -> list[dict]:
        path = self.base / "runs_index.jsonl"
        if not path.exists():
            return []
        return self._read_jsonl(path)

    def to_dataframe(self, data_type: str = "scores"):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Install pandas: pip install pandas")
        folder = self.base / data_type
        records = []
        for path in sorted(folder.glob("*.jsonl")):
            records.extend(self._read_jsonl(path))
        return pd.DataFrame(records)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
