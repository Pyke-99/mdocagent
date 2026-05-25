import json
from dataclasses import asdict
from typing import Dict, List

from prompts.route_agent_prompts import ROUTE_AGENT2_PROMPT
from schemas.route_agent_schema import RouteAgent2Output, RouteEvidenceItem


class RouteAgent2:
    def __init__(self, model):
        self.model = model

    def run(self, question: str, texts: List[str], images: List[str]) -> tuple[RouteAgent2Output, dict | None]:
        prompt = ROUTE_AGENT2_PROMPT + "\n\nQuestion:\n" + question

        raw = ""
        token_usage = None
        try:
            raw, _, token_usage = self.model.predict(prompt, texts=texts or None, images=images or None)
        except Exception:
            raw = ""

        parsed = self._safe_json_parse(raw)
        if parsed:
            return self._normalize(parsed, texts, images, question, raw_answer=raw), token_usage

        return self._fallback(question, texts, images), token_usage

    def to_dict(self, output: RouteAgent2Output) -> Dict:
        result = asdict(output)
        # Convert nested dataclass objects to dicts
        if "candidate_evidence" in result:
            result["candidate_evidence"] = [asdict(ev) if hasattr(ev, '__dataclass_fields__') else ev 
                                           for ev in result["candidate_evidence"]]
        return result

    def _normalize(self, parsed: Dict, texts: List[str], images: List[str], question: str, raw_answer: str = "") -> RouteAgent2Output:
        question_type = str(parsed.get("question_type", "simple")).strip().lower()
        if question_type not in {"comparison", "counting", "temporal", "table", "multi-hop", "constrained", "simple"}:
            question_type = "simple"

        entities = parsed.get("entities", [])
        if not isinstance(entities, list):
            entities = []
        entities = [str(x).strip() for x in entities if str(x).strip()]

        target = str(parsed.get("target", "")).strip()

        constraints = parsed.get("constraints", [])
        if not isinstance(constraints, list):
            constraints = []
        constraints = [str(x).strip() for x in constraints if str(x).strip()]

        requires_multi_hop = bool(parsed.get("requires_multi_hop", False))
        requires_table = bool(parsed.get("requires_table", False))
        requires_image = bool(parsed.get("requires_image", False))

        candidate_evidence = self._parse_evidence(parsed.get("candidate_evidence", []))
        route_to_multi = bool(parsed.get("route_to_multi", False))
        reason = str(parsed.get("reason", parsed.get("route_reason", ""))).strip()
        
        return RouteAgent2Output(
            question_type=question_type,
            entities=entities,
            target=target,
            constraints=constraints,
            requires_multi_hop=requires_multi_hop,
            requires_table=requires_table,
            requires_image=requires_image,
            candidate_evidence=candidate_evidence,
            route_to_multi=route_to_multi,
            reason=reason,
        )

    def _parse_evidence(self, evidence_list) -> List[RouteEvidenceItem]:
        if not isinstance(evidence_list, list):
            return []
        
        items = []
        for ev in evidence_list:
            if isinstance(ev, dict):
                items.append(RouteEvidenceItem(
                    text=str(ev.get("text", ev.get("snippet", ""))).strip()[:300],
                    type=str(ev.get("type", ev.get("evidence_role", "background"))).strip().lower(),
                ))
        return items

    def _fallback(self, question: str, texts: List[str], images: List[str]) -> RouteAgent2Output:
        return RouteAgent2Output(
            question_type="simple",
            entities=[],
            target="",
            constraints=[],
            requires_multi_hop=False,
            requires_table=False,
            requires_image=bool(images),
            candidate_evidence=[],
            route_to_multi=False,
            reason="fallback",
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
