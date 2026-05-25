from dataclasses import dataclass, field
from typing import List


@dataclass
class RouteGateOutput:
    route: str = "complex"
    reason: str = ""
    confidence: float = 0.0
    key_signals: List[str] = field(default_factory=list)
    initial_answer: str = ""
