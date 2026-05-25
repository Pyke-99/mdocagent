from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class VerifierOutput:
    verdict: str = "abstain"
    pass_reason: str = ""
    issues: List[str] = field(default_factory=list)
    claim_evidence_map: List[Dict[str, str]] = field(default_factory=list)
    missing_requirements: List[str] = field(default_factory=list)
    format_issues: List[str] = field(default_factory=list)
    revision_instruction: str = ""
