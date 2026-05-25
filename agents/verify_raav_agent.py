"""
Verify Agent for RAAV pipeline.

Only verifies, does not re-answer or directly rewrite facts.
"""

import json
from typing import Dict, List, Optional

from prompts.raav_prompts import VERIFY_RAAV_PROMPT
from schemas.raav_schema import VerifyOutput


class VerifyRaavAgent:
    """Verify Agent: only verifies answers."""

    def __init__(self, model):
        self.model = model

    def run(
        self,
        question: str,
        chunks: List[str],
        candidate_answer: str,
        used_evidence: List[str],
        analysis_result: Optional[Dict] = None,
    ) -> VerifyOutput:
        """
        Verify candidate answer against question and chunks.

        Args:
            question: The question string
            chunks: List of text chunks
            candidate_answer: Answer to verify
            used_evidence: Evidence used in the answer
            analysis_result: Optional analysis result

        Returns:
            VerifyOutput with verdict, issues, revision_instruction
        """
        analysis_info = ""
        if analysis_result:
            analysis_info = (
                "\n\nAnalysis Result:\n"
                + json.dumps(analysis_result, ensure_ascii=False)
            )

        verification_payload = {
            "question": question,
            "chunks": chunks,
            "candidate_answer": candidate_answer,
            "used_evidence": used_evidence,
            "analysis_result": analysis_result or {},
        }

        prompt = (
            VERIFY_RAAV_PROMPT
            + "\n\nVerification Payload:\n"
            + json.dumps(verification_payload, ensure_ascii=False)
        )

        raw = ""
        token_usage = None
        try:
            raw, _, token_usage = self.model.predict(prompt, texts=chunks or None, images=None)
        except Exception as e:
            print(f"VerifyRaavAgent error: {e}")
            raw = ""

        parsed = self._safe_json_parse(raw)
        if parsed:
            output = self._normalize(parsed)
            output.token_usage = token_usage or {}
            return output

        output = self._fallback()
        output.token_usage = token_usage or {}
        return output

    def to_dict(self, output: VerifyOutput) -> Dict:
        """Convert output to dict."""
        return {
            "verdict": output.verdict,
            "issues": output.issues,
            "revision_instruction": output.revision_instruction,
        }

    def _normalize(self, parsed: Dict) -> VerifyOutput:
        """Normalize parsed JSON to VerifyOutput."""
        verdict = str(parsed.get("verdict", "abstain")).strip().lower()
        if verdict not in {"pass", "minor_revise", "major_revise", "abstain"}:
            verdict = "abstain"

        issues = parsed.get("issues", [])
        if not isinstance(issues, list):
            issues = []
        issues = [str(x).strip() for x in issues if str(x).strip()]

        revision_instruction = str(parsed.get("revision_instruction", "")).strip()

        return VerifyOutput(
            verdict=verdict,
            issues=issues,
            revision_instruction=revision_instruction,
        )

    def _fallback(self) -> VerifyOutput:
        """Fallback when parsing fails."""
        return VerifyOutput(
            verdict="major_revise",
            issues=["Unable to parse verifier output"],
            revision_instruction="Review the candidate answer against the chunks and provide a corrected answer if the chunks contain relevant evidence.",
        )

    def _safe_json_parse(self, text: str):
        """Safely parse JSON from text."""
        if not text:
            return None

        try:
            return json.loads(text)
        except Exception:
            pass

        # Try to extract JSON from text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None

        return None
