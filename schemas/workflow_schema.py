from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Agent1Output:
    task_type: str
    question_type: str
    task_operation: str
    answer_target: str
    hard_constraints: List[str] = field(default_factory=list)
    workflow_guidance: str = ""
    target_variable: str = ""
    answer_type: str = ""
    key_constraints: List[str] = field(default_factory=list)
    time_constraint: str = ""
    scope_constraint: str = ""
    comparison_axes: List[str] = field(default_factory=list)
    calc_requirements: List[str] = field(default_factory=list)
    modality_hint: str = ""
    risk_flags: List[str] = field(default_factory=list)
    must_check_slots: List[Dict[str, str]] = field(default_factory=list)
    reasoning_focus: List[str] = field(default_factory=list)
    forbidden_shortcuts: List[str] = field(default_factory=list)
    answer_style: str = "concise"
