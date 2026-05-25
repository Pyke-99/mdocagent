"""
Answer Agent for RAAV pipeline.

Unified answer generator. Supports two modes: single and multi.
"""

import json
import re
from typing import Dict, List, Optional

from prompts.raav_prompts import ANSWER_RAAV_PROMPT, ANSWER_RAAV_REVISION_PROMPT
from schemas.raav_schema import AnswerOutput


class AnswerRaavAgent:
    """Answer Agent: unified answer generator."""

    def __init__(self, model):
        self.model = model

    def run(
        self,
        question: str,
        chunks: List[str],
        analysis_result: Optional[Dict] = None,
    ) -> AnswerOutput:
        """
        Generate answer based on question and chunks.

        Args:
            question: The question string
            chunks: List of text chunks
            analysis_result: Optional analysis result from Analysis Agent

        Returns:
            AnswerOutput with answer and used_evidence
        """
        analysis_info = ""
        if analysis_result:
            analysis_info = (
                "\n\nAnalysis Result:\n"
                + json.dumps(analysis_result, ensure_ascii=False)
            )

        prompt = ANSWER_RAAV_PROMPT + "\n\nQuestion:\n" + question + analysis_info

        raw = ""
        token_usage = None
        try:
            raw, _, token_usage = self.model.predict(prompt, texts=chunks or None, images=None)
        except Exception as e:
            print(f"AnswerRaavAgent error: {e}")
            raw = ""

        parsed = self._safe_json_parse(raw)
        if parsed:
            output = self._normalize(parsed)
            output.token_usage = token_usage
            return output

        output = self._fallback(raw)
        output.token_usage = token_usage
        return output

    def run_revision(
        self,
        question: str,
        chunks: List[str],
        analysis_result: Optional[Dict],
        previous_answer: str,
        issues: List[str],
        revision_instruction: str,
    ) -> AnswerOutput:
        """
        Revise answer based on Verify Agent feedback.

        Args:
            question: The question string
            chunks: List of text chunks
            analysis_result: Optional analysis result
            previous_answer: Previously generated answer
            issues: Issues identified by Verify Agent
            revision_instruction: Revision instructions

        Returns:
            AnswerOutput with revised answer and used_evidence
        """
        revision_payload = {
            "question": question,
            "chunks": chunks,
            "analysis_result": analysis_result or {},
            "previous_answer": previous_answer,
            "issues": issues,
            "revision_instruction": revision_instruction,
        }

        prompt = ANSWER_RAAV_REVISION_PROMPT + "\n\nRevision Payload:\n" + json.dumps(
            revision_payload, ensure_ascii=False
        )

        raw = ""
        token_usage = None
        try:
            raw, _, token_usage = self.model.predict(prompt, texts=chunks or None, images=None)
        except Exception as e:
            print(f"AnswerRaavAgent revision error: {e}")
            raw = ""

        parsed = self._safe_json_parse(raw)
        if parsed:
            output = self._normalize(parsed)
            output.token_usage = token_usage or {}
            return output

        output = self._fallback(raw)
        output.token_usage = token_usage or {}
        return output

    def to_dict(self, output: AnswerOutput) -> Dict:
        """Convert output to dict."""
        return {
            "answer": output.answer,
            "used_evidence": output.used_evidence,
        }

    def _normalize(self, parsed: Dict) -> AnswerOutput:
        """Normalize parsed JSON to AnswerOutput."""
        answer = str(parsed.get("answer", "")).strip()

        used_evidence = parsed.get("used_evidence", [])
        if not isinstance(used_evidence, list):
            used_evidence = []
        used_evidence = [str(x).strip() for x in used_evidence if str(x).strip()]

        return AnswerOutput(
            answer=answer,
            used_evidence=used_evidence,
        )

    def _fallback(self, raw: str) -> AnswerOutput:
        """Fallback when parsing fails."""
        fallback_answer = str(raw).strip() if raw else ""

        if fallback_answer:
            match = re.search(r'"answer"\s*:\s*"((?:\\.|[^"\\])*)"', fallback_answer, re.S)
            if match:
                fallback_answer = match.group(1).strip()
                fallback_answer = fallback_answer.replace('\\"', '"')
                fallback_answer = fallback_answer.replace("\\n", " ")

        if fallback_answer.startswith("{") and "\"answer\"" in fallback_answer:
            # Avoid returning a raw JSON blob as the final answer.
            fallback_answer = ""

        if not fallback_answer:
            fallback_answer = "Cannot be determined from the provided chunks."

        return AnswerOutput(
            answer=fallback_answer,
            used_evidence=[],
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
