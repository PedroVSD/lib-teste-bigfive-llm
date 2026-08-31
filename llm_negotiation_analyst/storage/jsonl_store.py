"""
Storage layer for negotiation results and scores.

Design goals:
  - Zero external dependencies (only stdlib)
  - Append-only JSONL for raw transcripts (easy to grep, stream, process with pandas)
  - Separate JSONL for scored profiles (can be loaded independently)
  - Each run identified by run_id; all files use run_id in filenames

Output structure:
    results/
    ├── transcripts/
    │   └── {scenario}_{run_id}.jsonl     # one line per turn
    ├── scores/
    │   └── {scenario}_{run_id}_scores.jsonl  # one line per agent Big5Profile
    └── runs_index.jsonl                   # one line per run (summary)
"""

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ..simulation.engine import NegotiationResult
from ..scoring.big5 import Big5Profile, Dimension


class StorageManager:
    """
    Persists negotiation results to a directory tree of JSONL files.

    Args:
        base_dir: Root directory for all outputs. Created if it doesn't exist.
    """

    def __init__(self, base_dir: str = "results"):
        self.base = Path(base_dir)
        self._init_dirs()

    def _init_dirs(self):
        for sub in ["transcripts", "scores"]:
            (self.base / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_result(self, result: NegotiationResult) -> dict[str, Path]:
        """
        Save transcript and return file paths.
        Also appends a summary line to runs_index.jsonl.
        """
        exp_name = result.metadata.get("experiment_name") if result.metadata else None
        slug = f"{exp_name}_{result.scenario_name}_{result.run_id}" if exp_name else f"{result.scenario_name}_{result.run_id}"

        # --- Transcript JSONL (one line per turn) ---
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

        # --- Runs index ---
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
        """Save Big5 profiles to JSONL."""
        exp_name = result.metadata.get("experiment_name") if result.metadata else None
        slug = f"{exp_name}_{result.scenario_name}_{result.run_id}" if exp_name else f"{result.scenario_name}_{result.run_id}"
        scores_path = self.base / "scores" / f"{slug}_scores.jsonl"

        with open(scores_path, "w", encoding="utf-8") as f:
            for agent_id, profile in profiles.items():
                line = {
                    "run_id": result.run_id,
                    "scenario": result.scenario_name,
                    "agent_id": agent_id,
                    "model_identifier": profile.model_identifier,
                    "role": result.agent_roles.get(agent_id, "unknown"),
                    "scores": {d.value: v for d, v in profile.scores.items()},
                    "per_turn_scores": [
                        {
                            "dimension": s.dimension.value,
                            "score": s.score,
                            "justification": s.justification,
                            "turn_index": s.turn_index,
                            "confidence": s.confidence,
                        }
                        for s in profile.per_turn_scores
                    ],
                    "notes": profile.notes,
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        return scores_path

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------

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
        """
        Load all records of a given type into a pandas DataFrame.

        Args:
            data_type: "scores" or "transcripts"
        """
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
