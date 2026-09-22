"""Policy loading: YAML files -> laya typed-question schemas."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Policy:
    name: str
    type: str
    instructions: str
    criteria: dict = field(default_factory=dict)

    def question(self) -> dict:
        return {
            self.name: {
                "type": self.type,
                "instructions": self.instructions,
                "criteria": self.criteria,
            }
        }


def load_policies(path: str | Path) -> dict[str, Policy]:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    policies = {}
    for name, spec in raw.items():
        policies[name] = Policy(
            name=name,
            type=spec.get("type", "choice"),
            instructions=spec.get("instructions", ""),
            criteria=spec.get("criteria", {}),
        )
    return policies
