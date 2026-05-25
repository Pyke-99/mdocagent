import json
import re
from dataclasses import asdict
from typing import Dict, List

from prompts.route_gate_prompts import ROUTE_GATE_PROMPT
from schemas.route_gate_schema import RouteGateOutput


class RouteGateAgent:
    def __init__(self, model):
        self.model = model

    def run(self, question: str, texts: List[str], images: List[str]) -> tuple[RouteGateOutput, dict | None]:
        prompt = ROUTE_GATE_PROMPT + "\n\nQuestion:\n" + question

        raw = ""
        token_usage = None
        try:
            raw, _, token_usage = self.model.predict(prompt, texts=texts or None, images=images or None)
        except Exception:
            raw = ""

        parsed = self._safe_json_parse(raw)
        if parsed:
            return self._normalize(parsed, raw_answer=raw, question=question), token_usage

        return self._fallback(question, texts, images, raw), token_usage

    def to_dict(self, output: RouteGateOutput) -> Dict:
        return asdict(output)

    def _normalize(self, parsed: Dict, raw_answer: str, question: str) -> RouteGateOutput:
        route = str(parsed.get("route", "complex")).strip().lower()
        if route not in {"simple", "complex"}:
            route = self._infer_route(question)

        reason = str(parsed.get("reason", "")).strip()
        if not reason:
            reason = "model_decision"

        confidence = parsed.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        key_signals = parsed.get("key_signals", [])
        if not isinstance(key_signals, list):
            key_signals = []
        key_signals = [str(x).strip() for x in key_signals if str(x).strip()]

        return RouteGateOutput(
            route=route,
            reason=reason,
            confidence=confidence,
            key_signals=key_signals,
            initial_answer=str(parsed.get("initial_answer", raw_answer or "")).strip(),
        )

    def _fallback(self, question: str, texts: List[str], images: List[str], raw_answer: str) -> RouteGateOutput:
        route = self._infer_route(question)
        key_signals = self._extract_key_signals(question)
        reason = "lightweight_rule_based_gate" if route == "simple" else "complexity_signals_detected"
        return RouteGateOutput(
            route=route,
            reason=reason,
            confidence=0.55 if route == "simple" else 0.7,
            key_signals=key_signals,
            initial_answer=str(raw_answer or "").strip(),
        )

    def _infer_route(self, question: str) -> str:
        q = str(question or "").lower()
        complex_cues = [
            "compare",
            "difference",
            "more than",
            "less than",
            "how many",
            "count",
            "table",
            "chart",
            "figure",
            "timeline",
            "before",
            "after",
            "across",
            "between",
            "which one",
            "multiple",
            "both",
        ]
        return "complex" if any(c in q for c in complex_cues) else "simple"

    def _extract_key_signals(self, question: str) -> List[str]:
        q = str(question or "").lower()
        signals = []
        for cue in ["compare", "count", "table", "chart", "timeline", "before", "after", "across", "between"]:
            if cue in q:
                signals.append(cue)
        return signals[:5]

    def _safe_json_parse(self, text: str):
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
        return None
