from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RouteEvidenceItem:
    text: str = ""
    type: str = "background"


@dataclass
class RouteAgent2Output:
    question_type: str = "simple"
    entities: List[str] = field(default_factory=list)
    target: str = ""
    constraints: List[str] = field(default_factory=list)
    requires_multi_hop: bool = False
    requires_table: bool = False
    requires_image: bool = False
    candidate_evidence: List[RouteEvidenceItem] = field(default_factory=list)
    reason: str = ""


@dataclass
class RouteAgent3Output:
    aligned_evidence: List[Dict] = field(default_factory=list)
    evidence_groups: List[Dict] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    missing_info: List[str] = field(default_factory=list)
    ready_for_answer: bool = False
    final_text_chunks: List[str] = field(default_factory=list)
    final_image_chunks: List[str] = field(default_factory=list)
    cross_modal_relations: List[Dict] = field(default_factory=list)
    alignment_points: List[str] = field(default_factory=list)
    alignment_instructions: List[str] = field(default_factory=list)
    response_mode: str = "text-first"


@dataclass
class RouteAgent4Output:
    verdict: str = "pass"
    support_span: str = ""
    missing_constraints: List[str] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    conflict_points: List[str] = field(default_factory=list)
    repair_instruction: str = ""
    downgrade_instruction: str = ""
