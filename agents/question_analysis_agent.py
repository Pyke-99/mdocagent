import json
import re
from dataclasses import asdict
from typing import Dict, List

from prompts.question_analysis_prompts import QUESTION_ANALYSIS_PROMPT
from schemas.workflow_schema import Agent1Output


class QuestionAnalysisAgent:
    def __init__(self, model):
        self.model = model

    def run(self, question: str) -> Agent1Output:
        prompt = (
            QUESTION_ANALYSIS_PROMPT
            + "\nQuestion:\n"
            + question
        )

        raw = ""
        try:
            raw, _, _ = self.model.predict(prompt)
        except Exception:
            raw = ""

        parsed = self._safe_json_parse(raw)
        task_operation = self._infer_task_operation(question)
        answer_target = self._infer_answer_target(question)
        answer_type = self._infer_answer_type(question)
        question_type = self._infer_question_type(question)
        modality_hint = self._infer_modality_hint(question)
        risk_flags = self._infer_risk_flags(question)
        constraints_from_model = []
        slots_from_model = []

        if parsed:
            task_operation = str(parsed.get("task_operation", task_operation)).lower().strip() or task_operation
            answer_target = str(parsed.get("answer_target", answer_target)).strip() or answer_target
            answer_type = str(parsed.get("answer_type", answer_type)).strip().lower() or answer_type
            question_type = str(parsed.get("question_type", question_type)).strip().lower() or question_type
            modality_hint = self._normalize_modality_hint(parsed.get("modality_hint", modality_hint))
            risk_flags = self._normalize_text_list(parsed.get("risk_flags", risk_flags))
            constraints_from_model = self._normalize_text_list(
                parsed.get("hard_constraints", parsed.get("key_constraints", []))
            )
            slots_from_model = parsed.get("must_check_slots", [])

        hard_constraints = self._build_hard_constraints(question, constraints_from_model)
        must_check_slots = self._normalize_slot_objects(slots_from_model, question)
        must_check_slots = self._compact_slots(must_check_slots, hard_constraints)
        if not must_check_slots:
            must_check_slots = self._slots_from_constraints(hard_constraints)

        task_type = self._infer_task_type(question, task_operation, answer_type)
        return Agent1Output(
            task_type=task_type,
            question_type=question_type,
            task_operation=task_operation,
            answer_target=answer_target,
            hard_constraints=hard_constraints,
            workflow_guidance=f"{task_operation}: {answer_target}",
            target_variable=self._infer_target_variable(question, task_operation),
            answer_type=answer_type,
            # Keep legacy compatibility by mirroring hard constraints.
            key_constraints=hard_constraints,
            time_constraint=self._extract_time_constraint(question),
            scope_constraint=self._extract_scope_constraint(question),
            comparison_axes=[],
            calc_requirements=[],
            modality_hint=modality_hint,
            risk_flags=risk_flags[:2],
            must_check_slots=must_check_slots,
            reasoning_focus=[],
            forbidden_shortcuts=[],
            answer_style=self._infer_answer_style(question),
        )

    def to_dict(self, output: Agent1Output):
        return asdict(output)

    def _safe_json_parse(self, text: str):
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        fenced = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        for candidate in fenced:
            try:
                return json.loads(candidate)
            except Exception:
                continue
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
        return None

    def _ensure_list(self, value):
        if isinstance(value, list):
            return [str(v) for v in value]
        if value is None:
            return []
        return [str(value)]

    def _extract_slot_terms(self, question: str):
        phrases = self._extract_candidate_phrases(question, max_items=5)
        if phrases:
            return phrases

        # Fallback to single-term extraction when no valid phrase can be found.
        tokens = re.findall(r"[A-Za-z0-9%]+", question.lower())
        stop = {
            "the", "is", "are", "what", "which", "who", "whom", "when", "where", "how", "why", "a", "an", "of", "to", "in", "for", "on", "and", "or",
            "this", "that", "from", "among", "report", "according", "one", "all"
        }
        slots = []
        for t in tokens:
            if len(t) < 3 or t in stop:
                continue
            if t not in slots:
                slots.append(t)
            if len(slots) >= 5:
                break
        return slots

    def _extract_candidate_phrases(
        self,
        question: str,
        min_words: int = None,
        max_words: int = None,
        max_items: int = 6,
    ) -> List[str]:
        q = " ".join(str(question).split()).lower()
        tokens = re.findall(r"[a-z0-9%]+", q)
        if not tokens:
            return []

        dyn_min = 1 if min_words is None else max(1, int(min_words))
        dyn_max = min(6, max(1, len(tokens) // 2)) if max_words is None else max(1, int(max_words))
        if dyn_max < dyn_min:
            dyn_max = dyn_min

        stop = {
            "the", "is", "are", "was", "were", "what", "which", "who", "whom", "when", "where", "how", "why",
            "a", "an", "to", "in", "for", "on", "and", "or", "this", "that", "from", "among", "report",
            "according", "one", "all", "it", "its", "their", "with", "by", "as", "at", "be", "been", "being", "many",
        }
        connector_words = {"of", "the", "and", "or", "to", "in", "for", "with", "by", "from", "among", "between"}
        banned_inside = {
            "which", "what", "who", "whom", "when", "where", "how", "why", "one", "all",
            "according", "report", "this", "that", "their", "them", "they", "these", "those",
        }
        broad_singletons = {"group", "people", "public", "report", "question", "survey"}

        candidates = []
        for n in range(dyn_max, dyn_min - 1, -1):
            for i in range(0, len(tokens) - n + 1):
                span = tokens[i : i + n]
                if n > 1 and any(w in banned_inside for w in span):
                    continue
                informative = [w for w in span if w not in stop and len(w) >= 3]
                if n == 1 and len(informative) < 1:
                    continue
                if n >= 2 and len(informative) < 1:
                    continue
                if span[0] in connector_words or span[-1] in connector_words:
                    continue
                if re.fullmatch(r"(?:19|20)\d{2}", span[0]) and len([w for w in informative if not re.fullmatch(r"(?:19|20)\d{2}", w)]) < 1:
                    # Avoid awkward spans like "2015 one group" while keeping true year-range phrases.
                    if not (n >= 3 and span[1] == "to" and re.fullmatch(r"(?:19|20)\d{2}", span[2])):
                        continue
                if n >= 4 and (len(informative) / max(1, n)) < 0.55:
                    continue
                phrase = " ".join(span).strip()
                if len(phrase) > 52 and not ("from" in span and "to" in span):
                    continue
                if not self._is_informative_term(phrase):
                    continue

                score = 1.2 * len(informative)
                score += min(1.0, 0.22 * n)
                if n > 4:
                    score -= 0.35 * (n - 4)
                if any(re.fullmatch(r"(?:19|20)\d{2}", w) for w in span):
                    score += 1
                if "%" in phrase:
                    score += 1
                if n == 1 and span[0] in broad_singletons:
                    score -= 1.2
                if "from" in span and "to" in span:
                    score += 0.6
                candidates.append((score, phrase))

        # Keep stable order by score first, then first appearance in the question.
        candidates.sort(key=lambda x: (-x[0], q.find(x[1])))
        out = []
        out_scores = []
        for score, phrase in candidates:
            if phrase in out:
                continue
            too_overlapping = False
            phrase_tokens = set(re.findall(r"[a-z0-9%]+", phrase))
            for idx, chosen in enumerate(out):
                chosen_tokens = set(re.findall(r"[a-z0-9%]+", chosen))
                overlap = len(phrase_tokens & chosen_tokens) / max(1, len(phrase_tokens | chosen_tokens))
                if overlap >= 0.8 and score <= out_scores[idx]:
                    too_overlapping = True
                    break
            if too_overlapping:
                continue
            out.append(phrase)
            out_scores.append(score)
            if len(out) >= max_items:
                break
        return out

    def _is_informative_term(self, text: str) -> bool:
        t = str(text).strip().lower()
        if len(t) < 3:
            return False
        if re.fullmatch(r"\d+", t):
            # Keep only meaningful year-like numeric constraints.
            return len(t) == 4 and (t.startswith("19") or t.startswith("20"))
        if " " in t:
            words = re.findall(r"[a-z0-9%]+", t)
            if len(words) < 2:
                return False
            weak_words = {
                "the", "is", "are", "was", "were", "be", "being", "been", "do", "does", "did",
                "has", "have", "had", "their", "them", "they", "those", "these", "this", "that",
                "with", "from", "into", "among", "between", "about", "which", "what", "who", "when",
                "where", "why", "how", "many", "more", "most", "less", "than", "then", "report",
                "according", "question", "there", "here", "over", "under", "across", "some", "any",
                "its", "own", "see", "whose", "current", "former", "latter", "such",
                "one", "two", "three", "four", "five", "first", "second", "third",
            }
            informative_count = sum(1 for w in words if w not in weak_words and (len(w) >= 3 or re.fullmatch(r"(?:19|20)\d{2}", w)))
            return informative_count >= 2
        weak = {
            "the", "is", "are", "was", "were", "be", "being", "been", "do", "does", "did",
            "has", "have", "had", "their", "them", "they", "those", "these", "this", "that",
            "with", "from", "into", "among", "between", "about", "which", "what", "who", "when",
            "where", "why", "how", "many", "more", "most", "less", "than", "then", "report",
            "according", "question", "there", "here", "over", "under", "across", "some", "any",
            "its", "own", "see", "whose", "current", "former", "latter", "former", "such",
            "one", "two", "three", "four", "five", "first", "second", "third",
        }
        return t not in weak

    def _compact_slots(self, slots: List[Dict], constraints: List[str]) -> List[Dict]:
        compact = []
        for item in slots or []:
            slot_name = str(item.get("slot_name", "")).strip().lower()
            if not self._is_informative_term(slot_name):
                continue
            if len(slot_name.split()) == 1 and slot_name in {"group", "people", "public"}:
                # Too broad single-word slots are weak as verification targets.
                continue
            compact.append(item)

        if not compact and constraints:
            compact = self._slots_from_constraints(constraints)

        dedup = []
        seen = set()
        for item in compact:
            name = str(item.get("slot_name", "")).strip().lower()
            if not name or name in seen:
                continue
            seen.add(name)
            dedup.append(item)
        return dedup[:4]

    def _build_hard_constraints(self, question: str, model_constraints: List[str]) -> List[str]:
        base = []
        for c in model_constraints or []:
            text = str(c).strip()
            if text and self._is_informative_term(text):
                base.append(text)

        for c in self._extract_key_constraints(question):
            if self._is_informative_term(c):
                base.append(c)

        time_c = self._extract_time_constraint(question)
        if time_c:
            base.append(time_c)
        scope_c = self._extract_scope_constraint(question)
        if scope_c:
            base.append(scope_c)

        dedup = []
        seen = set()
        for item in base:
            k = item.lower().strip()
            if not k or k in seen:
                continue
            seen.add(k)
            dedup.append(item)
        return dedup[:5]

    def _infer_question_type(self, question: str):
        q = question.lower()
        if any(k in q for k in ["why", "reason", "cause"]):
            return "causal"
        if any(k in q for k in ["compare", "difference", "versus", "vs", "greater", "higher", "lower"]):
            return "comparison"
        if any(k in q for k in ["trend", "increase", "decrease", "over time", "from", "to", "between"]):
            return "temporal"
        if any(k in q for k in ["how many", "number of", "count"]):
            return "counting"
        return "factoid"

    def _infer_task_type(self, question: str, task_operation: str, answer_type: str):
        q = question.lower()
        op = str(task_operation).lower()
        ans_t = str(answer_type).lower()
        if ans_t in {"integer", "int"} or op == "count" or any(k in q for k in ["how many", "number of", "count"]):
            return "count"
        if ans_t in {"percentage", "float", "ratio"} or "%" in q or "percentage" in q:
            return "ratio"
        if op == "compare" or any(k in q for k in ["compare", "greater", "higher", "lower", "versus", "vs"]):
            return "comparison"
        return "extract"

    def _build_default_slots(self, question: str):
        terms = self._extract_slot_terms(question)
        slots = []
        for idx, term in enumerate(terms):
            slots.append(
                {
                    "slot_id": f"slot_{idx+1}",
                    "slot_name": term,
                    "slot_description": f"evidence dimension about {term}",
                    "requiredness": "required" if idx < 3 else "optional",
                    "expected_evidence": "either",
                }
            )
        return slots

    def _normalize_slot_objects(self, raw_slots, question: str):
        if not isinstance(raw_slots, list):
            raw_slots = []
        normalized = []
        for idx, item in enumerate(raw_slots[:5]):
            if isinstance(item, dict):
                slot_name = str(item.get("slot_name", item.get("name", ""))).strip().lower()
                if not slot_name:
                    continue
                if not self._is_informative_term(slot_name):
                    continue
                normalized.append(
                    {
                        "slot_id": str(item.get("slot_id", f"slot_{idx+1}")).strip() or f"slot_{idx+1}",
                        "slot_name": slot_name,
                        "slot_description": str(item.get("slot_description", f"evidence dimension about {slot_name}")),
                        "requiredness": self._enum_text(item.get("requiredness"), {"required", "optional"}, "required"),
                        "expected_evidence": self._enum_text(item.get("expected_evidence"), {"text", "image", "either", "joint"}, "either"),
                    }
                )
                continue
            text = str(item).strip()
            if not text:
                continue
            if not self._is_informative_term(text):
                continue
            normalized.append(
                {
                    "slot_id": f"slot_{idx+1}",
                    "slot_name": text.lower(),
                    "slot_description": f"evidence dimension about {text.lower()}",
                    "requiredness": "required" if idx < 3 else "optional",
                    "expected_evidence": "either",
                }
            )

        if not normalized:
            normalized = self._build_default_slots(question)
        return normalized[:5]

    def _normalize_text_list(self, value):
        out = []
        for v in self._ensure_list(value):
            text = str(v).strip()
            if text and text.lower() not in [x.lower() for x in out]:
                out.append(text)
        return out

    def _slots_from_constraints(self, constraints: List[str]):
        slots = []
        for idx, c in enumerate(constraints[:4]):
            text = str(c).strip().lower()
            if not text or not self._is_informative_term(text):
                continue
            slots.append(
                {
                    "slot_id": f"slot_{idx+1}",
                    "slot_name": text,
                    "slot_description": f"core constraint: {text}",
                    "requiredness": "required" if idx < 3 else "optional",
                    "expected_evidence": "either",
                }
            )
        return slots

    def _extract_time_constraint(self, question: str):
        q = " ".join(question.split())
        m = re.search(r"(from\s+\d{4}\s+to\s+\d{4})", q, flags=re.IGNORECASE)
        if m:
            return m.group(1)
        m2 = re.findall(r"\b(19|20)\d{2}\b", q)
        years = re.findall(r"\b(?:19|20)\d{2}\b", q)
        if len(years) >= 2:
            return f"{years[0]} to {years[1]}"
        if len(years) == 1:
            return years[0]
        return ""

    def _extract_scope_constraint(self, question: str):
        q = " ".join(question.split())
        patterns = [
            r"among\s+all\s+\d+\s+[^,?.]+",
            r"compared\s+to\s+the\s+entire\s+population",
            r"among\s+[^,?.]+",
        ]
        for p in patterns:
            m = re.search(p, q, flags=re.IGNORECASE)
            if m:
                return m.group(0).strip()
        return ""

    def _extract_comparison_axes(self, question: str):
        q = question.lower()
        axes = []
        if any(k in q for k in ["greater", "higher", "lower", "compare", "compared"]):
            axes.append("comparative relation")
        if "percentage" in q or "%" in q:
            axes.append("percentage metric")
        if any(k in q for k in ["most", "highest", "lowest"]):
            axes.append("extremum selection")
        return axes

    def _extract_calc_requirements(self, question: str):
        q = question.lower()
        req = []
        if "percentage" in q or "%" in q:
            req.append("requires percentage-compatible evidence")
        if any(k in q for k in ["compared to", "ratio", "out of", "entire population"]):
            req.append("requires numerator and denominator alignment")
        if any(k in q for k in ["how many", "number of", "count"]):
            req.append("requires countable set and filtering condition")
        return req

    def _extract_key_constraints(self, question: str):
        q = " ".join(question.split())
        constraints = []

        time_c = self._extract_time_constraint(q)
        if time_c:
            constraints.append(time_c)

        scope_c = self._extract_scope_constraint(q)
        if scope_c:
            constraints.append(scope_c)

        own_center = re.search(r"its\s+own\s+research\s+center", q, flags=re.IGNORECASE)
        if own_center:
            constraints.append("from its own research center")

        phrase_constraints = self._extract_candidate_phrases(q, max_items=10)
        for p in phrase_constraints:
            lp = p.lower()
            exists = False
            for c in constraints:
                lc = str(c).lower()
                if lp == lc or lp in lc or lc in lp:
                    exists = True
                    break
            if not exists:
                constraints.append(p)
            if len(constraints) >= 6:
                break

        if len(constraints) < 2:
            tokens = re.findall(r"[A-Za-z0-9%]+", q.lower())
            stop = {
                "the", "is", "are", "what", "which", "who", "whom", "when", "where", "how", "why", "a", "an", "of", "to", "in", "for", "on", "and", "or",
                "this", "that", "from", "among", "report", "according", "one", "all", "its", "own", "many"
            }
            existing_tokens = set(re.findall(r"[a-z0-9%]+", " ".join(str(c).lower() for c in constraints)))
            for t in tokens:
                if len(t) < 4 or t in stop:
                    continue
                if t in existing_tokens:
                    continue
                if t not in [c.lower() for c in constraints]:
                    constraints.append(t)
                if len(constraints) >= 6:
                    break
        return constraints[:6]

    def _infer_target_variable(self, question: str, task_operation: str):
        q = question.lower()
        if task_operation == "count":
            return "count that satisfies question constraints"
        if "percentage" in q or "%" in q:
            return "percentage under specified scope and condition"
        if task_operation == "compare":
            return "comparative outcome between target entities"
        return "fact requested by question"

    def _infer_answer_type(self, question: str):
        q = question.lower()
        if any(k in q for k in ["how many", "number of"]):
            return "integer"
        if "percentage" in q or "%" in q:
            return "percentage"
        if any(k in q for k in ["which", "who", "whose"]):
            return "entity_or_label"
        return "short_text"

    def _infer_modality_hint(self, question: str):
        q = question.lower()
        if any(k in q for k in ["chart", "figure", "table", "image"]):
            return "image-first"
        if any(k in q for k in ["reference", "paragraph", "text"]):
            return "text-first"
        return "multimodal-joint"

    def _normalize_modality_hint(self, value):
        v = str(value).strip().lower()
        if v in {"text-first", "image-first", "multimodal-joint"}:
            return v
        return "multimodal-joint"

    def _infer_risk_flags(self, question: str):
        q = question.lower()
        flags = []
        if "percentage" in q and "entire population" in q:
            flags.append("denominator may be missing")
        years = re.findall(r"\b(?:19|20)\d{2}\b", q)
        if years and len(set(years)) == 1 and "from" in q and "to" in q:
            flags.append("time range may be underspecified")
        return flags

    def _infer_task_operation(self, question: str):
        q = question.lower()
        if any(k in q for k in ["how many", "number of", "count"]):
            return "count"
        if any(k in q for k in ["compare", "greater", "smaller", "higher", "lower", "which one"]):
            return "compare"
        if any(k in q for k in ["from", "between", "to", "year", "trend"]):
            return "infer"
        if any(k in q for k in ["where", "which page", "locate"]):
            return "locate"
        return "extract"

    def _infer_answer_target(self, question: str):
        return f"resolve the requested outcome in: {question.strip()}"

    def _infer_answer_style(self, question: str):
        q = question.lower()
        if any(k in q for k in ["how many", "number", "%", "percentage"]):
            return "count_with_unit"
        if any(k in q for k in ["which one", "greater", "compare"]):
            return "comparison_statement"
        return "short_fact"

    def _enum_text(self, value, valid_set, default):
        v = str(value).strip().lower()
        return v if v in valid_set else default
