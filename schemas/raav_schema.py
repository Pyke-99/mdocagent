"""
Schema definitions for Route-Analysis-Answer-Verify (RAAV) pipeline.

This module defines the dataclass schemas for each agent in the RAAV architecture.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass
class RouteOutput:
    """Output from Route Agent."""
    route: str = "multi"  # single | multi
    question_type: str = ""
    route_reason: str = ""
    token_usage: Dict = field(default_factory=dict)


@dataclass
class AnalysisOutput:
    """Output from Analysis Agent."""
    question_requirements: List[str] = field(default_factory=list)
    key_evidence: List[str] = field(default_factory=list)
    answer_plan: str = ""
    token_usage: Dict = field(default_factory=dict)


@dataclass
class AnswerOutput:
    """Output from Answer Agent."""
    answer: str = ""
    used_evidence: List[str] = field(default_factory=list)
    token_usage: Dict = field(default_factory=dict)


@dataclass
class VerifyOutput:
    """Output from Verify Agent."""
    verdict: str = "abstain"  # pass | minor_revise | major_revise | abstain
    issues: List[str] = field(default_factory=list)
    revision_instruction: str = ""
    token_usage: Dict = field(default_factory=dict)


@dataclass
class RAAvTraceEntry:
    """Trace entry for the RAAV pipeline."""
    mode: str = "raav"
    route: str = ""  # single | multi
    question_type: str = ""
    initial_answer: str = ""
    verify_verdict: str = ""
    revised_answer: str = ""
    final_answer: str = ""
    final_action: str = ""  # single_return | multi_pass | minor_revise | major_revise_success | major_revise_failed | fallback_conservative_answer | abstain
    analysis_requirements: List[str] = field(default_factory=list)
    analysis_key_evidence_count: int = 0
    verify_issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert trace entry to dict."""
        return asdict(self)
