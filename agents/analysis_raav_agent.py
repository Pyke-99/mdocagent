"""
Analysis Agent for RAAV pipeline.

Only enabled when route=multi. Analyzes question and organizes evidence.
Does not generate final answer.
"""

import json
from typing import Dict, List

from prompts.raav_prompts import ANALYSIS_RAAV_PROMPT
from schemas.raav_schema import AnalysisOutput


class AnalysisRaavAgent:
    """Analysis Agent: analyzes complex questions and organizes evidence."""

    def __init__(self, model):
        self.model = model

    def run(self, question: str, chunks: List[str], route_result: Dict = None) -> AnalysisOutput:
        """
        Analyze question and chunks, organize structured information.

        Args:
            question: The question string
            chunks: List of text chunks
            route_result: Optional route agent output

        Returns:
            AnalysisOutput with question_requirements, key_evidence, answer_plan
        """
        route_info = ""
        if route_result:
            route_info = (
                f"\nRoute Result:\n"
                f"- Route: {route_result.get('route', 'multi')}\n"
                f"- Question Type: {route_result.get('question_type', '')}\n"
                f"- Route Reason: {route_result.get('route_reason', '')}\n"
            )

        prompt = ANALYSIS_RAAV_PROMPT + route_info + "\n\nQuestion:\n" + question

        raw = ""
        token_usage = None
        try:
            raw, _, token_usage = self.model.predict(prompt, texts=chunks or None, images=None)
        except Exception as e:
            print(f"AnalysisRaavAgent error: {e}")
            raw = ""

        parsed = self._safe_json_parse(raw)
        if parsed:
            output = self._normalize(parsed)
            output.token_usage = token_usage or {}
            return output

        output = self._fallback()
        output.token_usage = token_usage or {}
        return output

    def to_dict(self, output: AnalysisOutput) -> Dict:
        """Convert output to dict."""
        return {
            "question_requirements": output.question_requirements,
            "key_evidence": output.key_evidence,
            "answer_plan": output.answer_plan,
        }

    def _normalize(self, parsed: Dict) -> AnalysisOutput:
        """Normalize parsed JSON to AnalysisOutput."""
        question_requirements = parsed.get("question_requirements", [])
        if not isinstance(question_requirements, list):
            question_requirements = []
        question_requirements = [str(x).strip() for x in question_requirements if str(x).strip()]

        key_evidence = parsed.get("key_evidence", [])
        if not isinstance(key_evidence, list):
            key_evidence = []
        key_evidence = [str(x).strip() for x in key_evidence if str(x).strip()]

        answer_plan = str(parsed.get("answer_plan", "")).strip()

        return AnalysisOutput(
            question_requirements=question_requirements,
            key_evidence=key_evidence,
            answer_plan=answer_plan,
        )

    def _fallback(self) -> AnalysisOutput:
        """Fallback when parsing fails."""
        return AnalysisOutput(
            question_requirements=[],
            key_evidence=[],
            answer_plan="Unable to analyze. Please answer based on available chunks.",
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
