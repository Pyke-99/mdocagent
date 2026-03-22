from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from models.base_model import BaseModel
from models.qwen3vl import MyQwen3VL


logger = logging.getLogger(__name__)


@dataclass
class EvidencePack:
    """
    Evidence container used by ReflectionAgent.

    Attributes:
        question: Original user question (q).
        text_segments: Top-k retrieved text segments (Tq).
        image_paths: Top-k retrieved image paths (Iq).
        general_answer: Answer from the General agent (aG).
        text_critical: Critical textual info from Critical agent (Tc).
        image_critical: Critical visual info from Critical agent (Ic).
    """

    question: str
    text_segments: List[str]
    image_paths: List[str]
    general_answer: str
    text_critical: str = ""
    image_critical: str = ""


class ReflectionAgent:
    """
    Internal reflection agent that checks factual consistency between
    a candidate answer and the EvidencePack.

    It uses the local Qwen3-VL-8B-Instruct model via the existing BaseModel
    / MyQwen3VL inference pipeline.
    """

    def __init__(self, model: BaseModel, max_iterations: int = 3):
        self.model = model
        self.max_iterations = max_iterations

    @classmethod
    def from_local_qwen3vl(cls, max_iterations: int = 3) -> "ReflectionAgent":
        """
        Build a ReflectionAgent backed by the locally installed
        Qwen3-VL-8B-Instruct model.
        """
        # Minimal config object compatible with MyQwen3VL
        class _Cfg:
            # Both fields are used in MyQwen3VL.__init__
            model_id = "/root/MDocAgent/models/Qwen3-VL-8B-Instruct"
            model = "/root/MDocAgent/models/Qwen3-VL-8B-Instruct"
            # A slightly larger generation length for structured JSON
            max_new_tokens = 512

        qwen3_model = MyQwen3VL(_Cfg())
        return cls(qwen3_model, max_iterations=max_iterations)

    def _build_reflection_prompt(self, answer: str, evidence: EvidencePack) -> str:
        """
        Construct the reflection prompt that:
        - Extracts atomic factual claims from `answer`
        - Checks them against text and image evidence
        - Returns the required JSON structure
        """
        lines: List[str] = []
        lines.append(
            "You are a meticulous factual consistency checker. "
            "Your job is to verify a candidate answer against the provided evidence."
        )
        lines.append(
            "You MUST output a single JSON object with the exact structure:\n"
            '{\n'
            '  "claims": [\n'
            '    {"text": "...", "status": "SUPPORTED|INSUFFICIENT|CONFLICT", "evidence_refs": ["T0", "I1", ...]}\n'
            "  ],\n"
            '  "overall_status": "PASS" or "FAIL",\n'
            '  "suggested_focus": {\n'
            '    "text_ids": ["T0", "T1", ...],\n'
            '    "image_ids": ["I0", "I1", ...]\n'
            "  }\n"
            "}\n"
            "Return ONLY this JSON, with no extra explanation or formatting."
        )

        lines.append(f"\nQuestion (q): {evidence.question}")
        lines.append(f"\nGeneral agent answer (aG): {evidence.general_answer}")
        if evidence.text_critical:
            lines.append(f"\nCritical text info (Tc): {evidence.text_critical}")
        if evidence.image_critical:
            lines.append(f"\nCritical image info (Ic): {evidence.image_critical}")

        # Enumerate text evidence segments
        lines.append("\nText evidence segments (Tq):")
        if evidence.text_segments:
            for idx, seg in enumerate(evidence.text_segments):
                lines.append(f"[T{idx}] {seg}")
        else:
            lines.append("No text evidence available.")

        # Enumerate image evidence references
        lines.append("\nImage evidence segments (Iq):")
        if evidence.image_paths:
            for idx, _ in enumerate(evidence.image_paths):
                lines.append(
                    f"[I{idx}] Image at index {idx}. The actual image content is provided separately."
                )
        else:
            lines.append("No image evidence available.")

        lines.append("\nCandidate final answer (aS) to be checked:")
        lines.append(answer)

        lines.append(
            "\nYour tasks:\n"
            "1) Extract atomic factual claims from aS. An atomic claim is a minimal verifiable factual statement.\n"
            "2) For each claim, check whether it is SUPPORTED by the evidence, has INSUFFICIENT evidence, "
            "or is in direct CONFLICT with the evidence.\n"
            "3) For each claim, provide evidence_refs as a list of IDs like \"T0\" or \"I1\" that are most relevant "
            "to your judgment (empty list if no evidence).\n"
            "4) overall_status must be:\n"
            '   - \"PASS\" if and only if ALL claims are SUPPORTED;\n'
            '   - \"FAIL\" otherwise (any INSUFFICIENT or CONFLICT).\n'
            "5) suggested_focus.text_ids and suggested_focus.image_ids should contain the IDs of evidence "
            "segments that would be most helpful to focus on when rewriting the answer to fix factual issues.\n"
            "Respond with STRICTLY valid JSON as defined above."
        )

        return "\n".join(lines)

    def reflect(self, answer: str, evidence: EvidencePack) -> Dict[str, Any]:
        """
        Run a single reflection pass on the given answer.

        Returns a dict with keys:
            - claims: list of {text, status, evidence_refs}
            - overall_status: PASS | FAIL
            - suggested_focus: {text_ids, image_ids}
        If the model output cannot be parsed, a conservative PASS with empty claims is returned.
        """
        prompt = self._build_reflection_prompt(answer, evidence)

        # Use both text and image evidence for multimodal alignment
        raw_output, _ = self.model.predict(
            question=prompt,
            texts=evidence.text_segments,
            images=evidence.image_paths,
            history=None,
        )

        json_str = raw_output
        try:
            # Try to robustly locate JSON in the output
            start = raw_output.find("{")
            end = raw_output.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = raw_output[start : end + 1]

            result: Dict[str, Any] = json.loads(json_str)
        except Exception as e:
            logger.warning(
                "ReflectionAgent: failed to parse JSON from model output: %s; raw output: %r",
                e,
                raw_output[:500],
            )
            # Fallback to a safe PASS if parsing fails
            return {
                "claims": [],
                "overall_status": "PASS",
                "suggested_focus": {"text_ids": [], "image_ids": []},
            }

        # Normalize structure and enforce required fields
        claims = result.get("claims", [])
        if not isinstance(claims, list):
            claims = []

        normalized_claims: List[Dict[str, Any]] = []
        for c in claims:
            if not isinstance(c, dict):
                continue
            text = c.get("text", "")
            status = str(c.get("status", "")).upper()
            if status not in {"SUPPORTED", "INSUFFICIENT", "CONFLICT"}:
                status = "INSUFFICIENT"
            evidence_refs = c.get("evidence_refs", [])
            if not isinstance(evidence_refs, list):
                evidence_refs = []
            normalized_claims.append(
                {"text": text, "status": status, "evidence_refs": evidence_refs}
            )

        # Derive overall_status if missing or invalid
        overall_status = str(result.get("overall_status", "")).upper()
        if overall_status not in {"PASS", "FAIL"}:
            has_issue = any(
                c.get("status") in {"INSUFFICIENT", "CONFLICT"}
                for c in normalized_claims
            )
            overall_status = "FAIL" if has_issue else "PASS"

        suggested_focus = result.get("suggested_focus") or {}
        text_ids = suggested_focus.get("text_ids", [])
        image_ids = suggested_focus.get("image_ids", [])
        if not isinstance(text_ids, list):
            text_ids = []
        if not isinstance(image_ids, list):
            image_ids = []

        return {
            "claims": normalized_claims,
            "overall_status": overall_status,
            "suggested_focus": {"text_ids": text_ids, "image_ids": image_ids},
        }

