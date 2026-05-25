from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Agent4Output:
    final_answer: str
    answer_status: str
    used_chunks: List[str] = field(default_factory=list)
    filled_slots: List[str] = field(default_factory=list)
    unresolved_slots: List[str] = field(default_factory=list)
    confidence: float = 0.0
    token_usage: Optional[Dict[str, int]] = None
