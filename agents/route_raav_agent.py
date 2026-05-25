"""
Route Agent for RAAV pipeline.

Only responsible for lightweight routing. Does not answer the question.
"""

import json
from typing import Dict, List, Optional, Tuple

from prompts.raav_prompts import ROUTE_RAAV_PROMPT
from schemas.raav_schema import RouteOutput


class RouteRaavAgent:
    """Route Agent: lightweight routing only."""

    def __init__(self, model):
        self.model = model

    def run(self, question: str, chunks: List[str]) -> Tuple[RouteOutput, Dict]:
        """
        Route the question to single or multi path.

        Args:
            question: The question string
            chunks: List of text chunks

        Returns:
            (RouteOutput with route, question_type, route_reason, token_usage)
        """
        heuristic = self._heuristic_route(question)
        if heuristic is not None:
            heuristic.token_usage = {}
            return heuristic, {}

        prompt = ROUTE_RAAV_PROMPT + "\n\nQuestion:\n" + question

        raw = ""
        token_usage = None
        try:
            raw, _, token_usage = self.model.predict(prompt, texts=chunks or None, images=None)
        except Exception as e:
            print(f"RouteRaavAgent error: {e}")
            raw = ""

        parsed = self._safe_json_parse(raw)
        if parsed:
            output = self._normalize(parsed)
            output.token_usage = token_usage or {}
            return output, token_usage

        output = self._fallback(question)
        output.token_usage = token_usage or {}
        return output, token_usage

    def to_dict(self, output: RouteOutput) -> Dict:
        """Convert output to dict."""
        return {
            "route": output.route,
            "question_type": output.question_type,
            "route_reason": output.route_reason,
        }

    def _normalize(self, parsed: Dict) -> RouteOutput:
        """Normalize parsed JSON to RouteOutput."""
        route = str(parsed.get("route", "multi")).strip().lower()
        if route not in {"single", "multi"}:
            route = "multi"

        question_type = str(parsed.get("question_type", "")).strip()
        route_reason = str(parsed.get("route_reason", "")).strip()

        return RouteOutput(
            route=route,
            question_type=question_type,
            route_reason=route_reason,
        )

    def _is_yes_no_question(self, q: str) -> bool:
        return q.strip().startswith(("do ", "does ", "did ", "is ", "are ", "was ", "were ", "can ", "could "))

    def _heuristic_route(self, question: str) -> Optional[RouteOutput]:
        """Conservative deterministic override for clearly complex questions."""
        q = str(question or "").strip().lower()
        if not q:
            return None

        is_yes_no = self._is_yes_no_question(q)

        enumeration_signals = [
            "which datasets", "which tasks", "which languages", "which baselines", "which models",
            "which methods", "which metrics", "which approaches", "which techniques", "which hyperparameters",
            "what datasets", "what tasks", "what languages", "what baselines", "what models",
            "what methods", "what metrics", "what approaches", "what techniques", "what hyperparameters",
            "list", "enumerate",
        ]

        quantity_signals = [
            "how many", "how much", "by how much", "how big", "number of", "size of", "amount of",
            "improvement", "improved", "outperform", "outperforms", "better than", "compared to", "compared with",
        ]

        comparison_signals = [
            "compare", "comparison", "difference", "differences", "versus", " vs ",
            "best", "worst", "better", "higher", "lower", "more than", "less than",
            "previous state-of-the-art", "state-of-the-art results", "sota results",
        ]

        table_result_signals = [
            "table", "figure", "chart", "row", "column",
            "results on both", "both datasets", "all datasets", "across datasets", "across tasks",
            "performance on", "results on", "scores on",
        ]

        multi_field_signals = [
            "both", "respectively", "each", "all", "for each", "across",
            "source and target", "source-target", "before and after",
        ]

        has_enum = any(sig in q for sig in enumeration_signals)
        has_quantity = any(sig in q for sig in quantity_signals)
        has_comparison = any(sig in q for sig in comparison_signals)
        has_table_results = any(sig in q for sig in table_result_signals)

        if has_enum or has_quantity or has_comparison or has_table_results:
            return RouteOutput(
                route="multi",
                question_type="heuristic_complex",
                route_reason="heuristic override: explicit complexity signal",
            )

        has_multi_field = any(sig in q for sig in multi_field_signals)
        if has_multi_field and not is_yes_no:
            return RouteOutput(
                route="multi",
                question_type="heuristic_multi_field",
                route_reason="heuristic override: multi-field or multi-object signal",
            )

        return None

    def _fallback(self, question: str) -> RouteOutput:
        """Fallback routing based on simple heuristics."""
        q = str(question or "").lower()

        multi_cues = [
            "compare", "difference", "versus", "vs", "between",
            "more than", "less than", "how many", "count",
            "table", "chart", "figure", "timeline",
            "before", "after", "across", "which one",
            "multiple", "both", "either", "neither",
            "only", "except", "not", "at least", "at most",
        ]

        route = "multi" if any(cue in q for cue in multi_cues) else "single"

        return RouteOutput(
            route=route,
            question_type="fallback",
            route_reason="fallback routing based on heuristics",
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
