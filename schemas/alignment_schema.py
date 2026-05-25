from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CrossModalRelation:
    text_chunk_id: str
    image_chunk_id: str
    relation_type: str
    note: str = ""


@dataclass
class RewrittenEvidence:
    evidence_id: str
    source_chunk_id: str
    modality: str
    semantic_preserved_summary: str
    query_relevance_analysis: str
    answer_support: str = ""
    uncertainty_or_gap: str = ""
    verbatim_anchor: str = ""
    source_excerpt: str = ""


@dataclass
class Agent3Output:
    final_text_chunks: List[str] = field(default_factory=list)
    final_image_chunks: List[str] = field(default_factory=list)
    cross_modal_relations: List[CrossModalRelation] = field(default_factory=list)
    rewritten_evidence: List[RewrittenEvidence] = field(default_factory=list)
    alignment_points: List[str] = field(default_factory=list)
    alignment_instructions: List[str] = field(default_factory=list)
    response_mode: str = "multimodal-joint"
