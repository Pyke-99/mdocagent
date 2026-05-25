import json
from dataclasses import asdict
from typing import Dict, List

from prompts.route_agent_prompts import ROUTE_AGENT3_PROMPT
from schemas.route_agent_schema import RouteAgent3Output


class RouteAgent3:
    def __init__(self, model):
        self.model = model

    def run(self, question: str, agent2_output: Dict, texts: List[str], images: List[str]) -> tuple[RouteAgent3Output, dict | None]:
        prompt = ROUTE_AGENT3_PROMPT + "\n\nQuestion:\n" + question + "\n\nAgent2 output:\n" + json.dumps(agent2_output, ensure_ascii=True)

        raw = ""
        token_usage = None
        try:
            raw, _, token_usage = self.model.predict(prompt, texts=texts or None, images=images or None)
        except Exception:
            raw = ""

        parsed = self._safe_json_parse(raw)
        if parsed:
            return self._normalize(parsed, agent2_output, raw_answer=raw), token_usage

        return self._fallback(agent2_output), token_usage

    def to_dict(self, output: RouteAgent3Output) -> Dict:
        return asdict(output)

    def _normalize(self, parsed: Dict, agent2_output: Dict, raw_answer: str = "") -> RouteAgent3Output:
        aligned_evidence = parsed.get("aligned_evidence", [])
        if not isinstance(aligned_evidence, list):
            aligned_evidence = []

        evidence_groups = parsed.get("evidence_groups", [])
        if not isinstance(evidence_groups, list):
            evidence_groups = []

        conflicts = parsed.get("conflicts", [])
        if not isinstance(conflicts, list):
            conflicts = []
        conflicts = [str(x).strip() for x in conflicts if str(x).strip()]

        missing_info = parsed.get("missing_info", [])
        if not isinstance(missing_info, list):
            missing_info = []
        missing_info = [str(x).strip() for x in missing_info if str(x).strip()]

        ready_for_answer = bool(parsed.get("ready_for_answer", False))

        return RouteAgent3Output(
            aligned_evidence=aligned_evidence,
            evidence_groups=evidence_groups,
            conflicts=conflicts,
            missing_info=missing_info,
            ready_for_answer=ready_for_answer,
            final_text_chunks=parsed.get("final_text_chunks", []),
            final_image_chunks=parsed.get("final_image_chunks", []),
            cross_modal_relations=parsed.get("cross_modal_relations", []),
            alignment_points=parsed.get("alignment_points", []),
            alignment_instructions=parsed.get("alignment_instructions", []),
            response_mode=str(parsed.get("response_mode", "text-first")).strip().lower(),
        )

    def _fallback(self, agent2_output: Dict) -> RouteAgent3Output:
        return RouteAgent3Output(
            aligned_evidence=[],
            evidence_groups=[],
            conflicts=[],
            missing_info=[],
            ready_for_answer=False,
            final_text_chunks=[],
            final_image_chunks=[],
            cross_modal_relations=[],
            alignment_points=["fallback_alignment_empty"],
            alignment_instructions=["use available selected chunks conservatively"],
            response_mode="text-first",
        )

    def _safe_json_parse(self, text: str):
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
