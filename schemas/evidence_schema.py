from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ChunkDecision:
    chunk_id: str
    modality: str
    decision: str
    answer_role: str
    support_type: str
    constraint_match: str
    information_gain: str
    supported_slots: List[str] = field(default_factory=list)
    missing_slots: List[str] = field(default_factory=list)


@dataclass
class IntraModalRelation:
    source_chunk_id: str
    target_chunk_id: str
    relation_type: str
    note: str = ""


@dataclass
class Agent2Output:
    chunk_decisions: List[ChunkDecision] = field(default_factory=list)
    intra_modal_relations: List[IntraModalRelation] = field(default_factory=list)
    selection_summary: Dict[str, str] = field(default_factory=dict)
