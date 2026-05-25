import json
import re
from difflib import SequenceMatcher
from dataclasses import asdict
from typing import Dict, List

from prompts.evidence_selection_prompts import EVIDENCE_SELECTION_PROMPT
from schemas.evidence_schema import Agent2Output, ChunkDecision, IntraModalRelation


class EvidenceSelectionAgent:
    def __init__(self, model, max_use_text: int = 4, max_use_image: int = 3):
        self.model = model
        self.max_use_text = max_use_text
        self.max_use_image = max_use_image

    def run(self, question: str, agent1_output: Dict, text_chunks: List[Dict], image_chunks: List[Dict]) -> Agent2Output:
        q_terms = self._extract_terms(question)
        must_slots = self._normalize_slots(agent1_output.get("must_check_slots", []))
        modality_plan = self._decide_modality_plan(question, agent1_output, text_chunks, image_chunks)

        prompt = (
            EVIDENCE_SELECTION_PROMPT
            + "\nQuestion:\n"
            + question
            + "\n\nAgent1 output:\n"
            + json.dumps(agent1_output, ensure_ascii=True)
        )

        raw = ""
        try:
            raw, _, _ = self.model.predict(prompt, texts=[c["content"] for c in text_chunks], images=[c["content"] for c in image_chunks])
        except Exception:
            raw = ""

        parsed = self._safe_json_parse(raw)
        if parsed and parsed.get("chunk_decisions"):
            normalized = self._normalize_from_model(parsed, text_chunks, image_chunks)
            calibrated = self._post_calibrate(normalized, question, must_slots, text_chunks, image_chunks, modality_plan)
            return calibrated

        decisions: List[ChunkDecision] = []
        for chunk in text_chunks:
            decision, role, match, gain, supported_slots, missing_slots = self._score_text_chunk(
                question,
                chunk["content"],
                q_terms,
                must_slots,
            )
            decisions.append(
                ChunkDecision(
                    chunk_id=chunk["id"],
                    modality="text",
                    decision=decision,
                    answer_role=role,
                    support_type=self._derive_support_type(role, match),
                    constraint_match=match,
                    information_gain=gain,
                    supported_slots=supported_slots,
                    missing_slots=missing_slots,
                )
            )

        for idx, chunk in enumerate(image_chunks):
            decision = "use" if idx < 3 else "reserve"
            role = "supporting" if idx < 2 else "background"
            gain = "high" if idx == 0 else ("medium" if idx == 1 else "low")
            supported_slots, missing_slots = self._infer_slot_coverage_from_text(" ".join(q_terms), must_slots)
            decisions.append(
                ChunkDecision(
                    chunk_id=chunk["id"],
                    modality="image",
                    decision=decision,
                    answer_role=role,
                    support_type="complementary" if role == "supporting" else "background",
                    constraint_match="unknown",
                    information_gain=gain,
                    supported_slots=supported_slots,
                    missing_slots=missing_slots,
                )
            )

        decisions = self._enforce_budget_and_dedup(
            decisions,
            question,
            must_slots,
            text_chunks,
            image_chunks,
            modality_plan,
        )
        relations = self._build_intra_modal_relations(decisions)
        summary = {
            "text_use": str(len([d for d in decisions if d.modality == "text" and d.decision == "use"])),
            "image_use": str(len([d for d in decisions if d.modality == "image" and d.decision == "use"])),
            "mode": modality_plan.get("mode", "joint"),
            "budget": f"text[{modality_plan.get('min_text', 0)},{modality_plan.get('max_text', 0)}],image[{modality_plan.get('min_image', 0)},{modality_plan.get('max_image', 0)}]",
            "strategy": "dynamic modality budget + soft-slot coverage + complementary evidence retention",
        }
        return Agent2Output(chunk_decisions=decisions, intra_modal_relations=relations, selection_summary=summary)

    def to_dict(self, output: Agent2Output):
        return asdict(output)

    def _normalize_from_model(self, parsed: Dict, text_chunks: List[Dict], image_chunks: List[Dict]) -> Agent2Output:
        all_ids = {c["id"]: "text" for c in text_chunks}
        all_ids.update({c["id"]: "image" for c in image_chunks})

        decisions = []
        for item in parsed.get("chunk_decisions", []):
            if not isinstance(item, dict):
                continue
            cid = str(item.get("chunk_id", ""))
            if cid not in all_ids:
                continue
            answer_role = self._enum(
                item.get("answer_role"),
                {"direct", "partial", "supporting", "background", "distractor"},
                "supporting",
            )
            constraint_match = self._enum(
                item.get("constraint_match"),
                {"exact", "partial", "mismatch", "unknown"},
                "unknown",
            )
            decisions.append(
                ChunkDecision(
                    chunk_id=cid,
                    modality=all_ids[cid],
                    decision=self._enum(item.get("decision"), {"use", "reserve", "drop"}, "reserve"),
                    answer_role=answer_role,
                    support_type=self._enum(
                        item.get("support_type"),
                        {"direct", "partial", "complementary", "background", "mismatch"},
                        self._derive_support_type(answer_role, constraint_match),
                    ),
                    constraint_match=constraint_match,
                    information_gain=self._enum(
                        item.get("information_gain"),
                        {"high", "medium", "low", "redundant"},
                        "medium",
                    ),
                    supported_slots=self._normalize_slots(item.get("supported_slots", [])),
                    missing_slots=self._normalize_slots(item.get("missing_slots", [])),
                )
            )

        for cid, modality in all_ids.items():
            if not any(d.chunk_id == cid for d in decisions):
                decisions.append(
                    ChunkDecision(
                        chunk_id=cid,
                        modality=modality,
                        decision="drop",
                        answer_role="distractor",
                        support_type="mismatch",
                        constraint_match="unknown",
                        information_gain="redundant",
                        supported_slots=[],
                        missing_slots=[],
                    )
                )

        relations = []
        for rel in parsed.get("intra_modal_relations", []):
            if not isinstance(rel, dict):
                continue
            relations.append(
                IntraModalRelation(
                    source_chunk_id=str(rel.get("source_chunk_id", "")),
                    target_chunk_id=str(rel.get("target_chunk_id", "")),
                    relation_type=self._enum(
                        rel.get("relation_type"),
                        {"support", "complement", "duplicate", "conflict", "topical_related"},
                        "topical_related",
                    ),
                    note=str(rel.get("note", "")),
                )
            )

        summary = parsed.get("selection_summary", {}) if isinstance(parsed.get("selection_summary", {}), dict) else {}
        summary = {str(k): str(v) for k, v in summary.items()}
        return Agent2Output(chunk_decisions=decisions, intra_modal_relations=relations, selection_summary=summary)

    def _post_calibrate(
        self,
        output: Agent2Output,
        question: str,
        must_slots: List[str],
        text_chunks: List[Dict],
        image_chunks: List[Dict],
        modality_plan: Dict,
    ) -> Agent2Output:
        decisions = self._enforce_budget_and_dedup(
            output.chunk_decisions,
            question,
            must_slots,
            text_chunks,
            image_chunks,
            modality_plan,
        )
        relations = self._build_intra_modal_relations(decisions)
        summary = {
            "text_use": str(len([d for d in decisions if d.modality == "text" and d.decision == "use"])),
            "image_use": str(len([d for d in decisions if d.modality == "image" and d.decision == "use"])),
            "mode": modality_plan.get("mode", "joint"),
            "budget": f"text[{modality_plan.get('min_text', 0)},{modality_plan.get('max_text', 0)}],image[{modality_plan.get('min_image', 0)},{modality_plan.get('max_image', 0)}]",
            "strategy": "model-then-dynamic-modality-calibration with dedup and slot-gain early stop",
        }
        return Agent2Output(chunk_decisions=decisions, intra_modal_relations=relations, selection_summary=summary)

    def _enforce_budget_and_dedup(
        self,
        decisions: List[ChunkDecision],
        question: str,
        must_slots: List[str],
        text_chunks: List[Dict],
        image_chunks: List[Dict],
        modality_plan: Dict,
    ) -> List[ChunkDecision]:
        text_map = {c["id"]: c["content"] for c in text_chunks}
        image_map = {c["id"]: c["content"] for c in image_chunks}

        for d in decisions:
            if d.modality == "text":
                content = text_map.get(d.chunk_id, "")
                d._agent2_score = self._score_for_budget(content, question, must_slots, d)  # type: ignore[attr-defined]
            else:
                content_hint = image_map.get(d.chunk_id, "")
                d._agent2_score = self._score_for_budget(content_hint, question, must_slots, d)  # type: ignore[attr-defined]

        self._mark_text_duplicates(decisions, text_map)
        self._apply_budget(
            decisions,
            modality="text",
            max_use=int(modality_plan.get("max_text", self.max_use_text)),
            min_use=int(modality_plan.get("min_text", 0)),
            must_slots=must_slots,
        )
        self._apply_budget(
            decisions,
            modality="image",
            max_use=int(modality_plan.get("max_image", self.max_use_image)),
            min_use=int(modality_plan.get("min_image", 0)),
            must_slots=must_slots,
        )

        for d in decisions:
            if hasattr(d, "_agent2_score"):
                delattr(d, "_agent2_score")
        return decisions

    def _score_for_budget(self, content: str, question: str, must_slots: List[str], decision: ChunkDecision) -> float:
        text = str(content).lower()
        q_terms = self._extract_terms(question)
        hit_q = sum(1 for t in q_terms if t in text)
        semantic = self._semantic_relevance(question, text, must_slots, q_terms)
        hit_slot = len(getattr(decision, "supported_slots", []) or [])
        missing_slot_penalty = 0.1 * len(getattr(decision, "missing_slots", []) or [])
        noise_penalty = self._noise_penalty(text)

        role_weight = {
            "direct": 3.0,
            "partial": 2.0,
            "supporting": 1.6,
            "background": 0.5,
            "distractor": 0.0,
        }.get(decision.answer_role, 0.8)
        gain_weight = {
            "high": 2.0,
            "medium": 1.2,
            "low": 0.5,
            "redundant": 0.0,
        }.get(decision.information_gain, 0.6)
        match_weight = {
            "exact": 2.0,
            "partial": 1.2,
            "unknown": 0.4,
            "mismatch": -1.0,
        }.get(decision.constraint_match, 0.0)
        return role_weight + gain_weight + match_weight + 1.8 * semantic + 0.18 * hit_q + 0.65 * hit_slot - missing_slot_penalty - noise_penalty

    def _mark_text_duplicates(self, decisions: List[ChunkDecision], text_map: Dict[str, str]):
        seen = {}
        for d in [x for x in decisions if x.modality == "text"]:
            content = text_map.get(d.chunk_id, "")
            sig = " ".join(content.split())[:300]
            if not sig:
                continue
            if sig in seen:
                d.decision = "drop"
                d.answer_role = "distractor"
                d.support_type = "mismatch"
                d.constraint_match = "unknown"
                d.information_gain = "redundant"
                d.supported_slots = []
                d.missing_slots = []
            else:
                seen[sig] = d.chunk_id

    def _apply_budget(self, decisions: List[ChunkDecision], modality: str, max_use: int, min_use: int, must_slots: List[str]):
        items = [d for d in decisions if d.modality == modality]
        ranked = sorted(items, key=lambda x: getattr(x, "_agent2_score", 0.0), reverse=True)
        max_use = max(0, min(max_use, len(ranked)))
        min_use = max(0, min(min_use, max_use))

        keep_use_ids = set()
        covered_slots = set()

        for d in ranked:
            if len(keep_use_ids) >= max_use:
                break

            supported = set(d.supported_slots or [])
            slot_gain = len(supported - covered_slots)
            score = float(getattr(d, "_agent2_score", 0.0)) + 0.7 * slot_gain

            if len(keep_use_ids) < min_use:
                keep_use_ids.add(d.chunk_id)
                covered_slots.update(supported)
                continue

            threshold = 3.6 if modality == "text" else 3.2
            if score >= threshold or (slot_gain > 0 and d.constraint_match != "mismatch"):
                keep_use_ids.add(d.chunk_id)
                covered_slots.update(supported)

        for d in items:
            if d.chunk_id in keep_use_ids:
                if d.decision == "drop":
                    d.decision = "reserve"
                    if d.answer_role == "distractor":
                        d.answer_role = "supporting"
                    d.support_type = "complementary"
                else:
                    d.decision = "use"
            else:
                if d.decision == "use":
                    d.decision = "reserve"
                    if d.answer_role == "direct":
                        d.answer_role = "partial"
                    d.support_type = "partial"
                if d.decision == "drop" and d.constraint_match != "mismatch":
                    # Missing slot coverage alone should not force hard rejection.
                    d.decision = "reserve"
                    if d.answer_role == "distractor":
                        d.answer_role = "background"
                    if d.support_type == "mismatch":
                        d.support_type = "background"

    def _decide_modality_plan(self, question: str, agent1_output: Dict, text_chunks: List[Dict], image_chunks: List[Dict]) -> Dict:
        q = str(question).lower()
        task_type = str(agent1_output.get("task_type", "")).lower()
        answer_type = str(agent1_output.get("answer_type", "")).lower()
        hint = str(agent1_output.get("modality_hint", "")).lower()
        hard_constraints = agent1_output.get("hard_constraints", []) or []
        if not hard_constraints:
            hard_constraints = agent1_output.get("key_constraints", []) or []
        constraints = " ".join(str(x).lower() for x in hard_constraints)
        risk_flags = " ".join(str(x).lower() for x in agent1_output.get("risk_flags", []) or [])

        has_text = len(text_chunks) > 0
        has_image = len(image_chunks) > 0

        image_cues = ["figure", "chart", "table", "image", "diagram", "photo", "map", "visual"]
        text_cues = ["according to", "report", "survey", "statement", "paragraph", "described"]

        score_image = 0
        score_text = 0

        if any(c in q for c in image_cues):
            score_image += 2
        if any(c in constraints for c in image_cues):
            score_image += 1
        if "visual" in risk_flags or "chart" in risk_flags:
            score_image += 1

        if any(c in q for c in text_cues):
            score_text += 2
        if "time" in constraints or "year" in constraints:
            score_text += 1

        # Counting and ratio tasks usually require broader textual context and explicit numeric traces.
        if task_type in {"count", "ratio"} or answer_type in {"integer", "int", "percentage", "float", "ratio"}:
            score_text += 2

        if "image-first" in hint or "vision" in hint:
            score_image += 3
        if "text-first" in hint or "text-only" in hint:
            score_text += 3
        if "joint" in hint or "multimodal" in hint:
            score_text += 1
            score_image += 1

        if has_image and not has_text:
            mode = "image-first"
        elif has_text and not has_image:
            mode = "text-first"
        elif score_image >= score_text + 2:
            mode = "image-first"
        elif score_text >= score_image + 2:
            mode = "text-first"
        else:
            mode = "joint"

        if mode == "text-first":
            max_text = min(len(text_chunks), max(2, self.max_use_text + 1))
            max_image = min(len(image_chunks), 1 if score_image > 0 else 0)
            min_text = 1 if has_text else 0
            min_image = 0
        elif mode == "image-first":
            max_text = min(len(text_chunks), 2)
            max_image = min(len(image_chunks), max(2, self.max_use_image + 1))
            min_text = 0
            min_image = 1 if has_image else 0
        else:
            max_text = min(len(text_chunks), self.max_use_text)
            max_image = min(len(image_chunks), self.max_use_image)
            min_text = 1 if has_text else 0
            min_image = 1 if has_image and score_image > 0 else 0

        return {
            "mode": mode,
            "max_text": max_text,
            "max_image": max_image,
            "min_text": min_text,
            "min_image": min_image,
        }

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

    def _score_text_chunk(self, question: str, content: str, q_terms: List[str], must_slots: List[str]):
        text = content.lower()
        semantic_score = self._semantic_relevance(question, text, must_slots, q_terms)
        hits = sum(1 for t in q_terms if t in text)
        supported_slots, missing_slots = self._infer_slot_coverage_from_text(text, must_slots)
        mismatch = self._is_scope_mismatch(question, text)
        noisy = self._is_noisy_text_chunk(text)

        if noisy and semantic_score < 0.15 and len(supported_slots) == 0:
            return "drop", "distractor", "unknown", "redundant", supported_slots, missing_slots

        if mismatch and semantic_score < 0.18 and len(supported_slots) == 0:
            return "drop", "distractor", "mismatch", "redundant", supported_slots, missing_slots

        if semantic_score >= 0.46 or len(supported_slots) >= 2:
            return "use", "direct", "exact", "high", supported_slots, missing_slots
        if semantic_score >= 0.28 or len(supported_slots) == 1:
            return "reserve", "partial", "partial", "medium", supported_slots, missing_slots
        if semantic_score >= 0.16 or hits >= 1:
            return "reserve", "supporting", "unknown", "low", supported_slots, missing_slots

        # Keep potentially complementary evidence as reserve instead of hard drop.
        return "reserve", "background", "unknown", "low", supported_slots, missing_slots

    def _build_intra_modal_relations(self, decisions: List[ChunkDecision]) -> List[IntraModalRelation]:
        relations: List[IntraModalRelation] = []
        by_modality = {"text": [], "image": []}
        for d in decisions:
            by_modality[d.modality].append(d)

        for modality in ["text", "image"]:
            use_items = [d for d in by_modality[modality] if d.decision == "use"]
            for i in range(len(use_items) - 1):
                relations.append(
                    IntraModalRelation(
                        source_chunk_id=use_items[i].chunk_id,
                        target_chunk_id=use_items[i + 1].chunk_id,
                        relation_type="complement",
                        note="selected together by evidence selection",
                    )
                )
        return relations

    def _extract_terms(self, question: str):
        raw = re.findall(r"[A-Za-z0-9%]+", question.lower())
        stop = {
            "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "what", "which", "who", "whose",
            "how", "many", "according", "report", "from", "to", "in", "on", "of", "and", "or", "this", "that",
            "among", "with", "for", "by", "at", "as", "it", "its", "their", "than", "then",
        }
        out = []
        for t in raw:
            if len(t) < 3:
                continue
            if t in stop:
                continue
            if t not in out:
                out.append(t)
            if len(out) >= 30:
                break
        return out

    def _is_noisy_text_chunk(self, text: str) -> bool:
        cues = [
            "methodology",
            "table of contents",
            "about pew research center",
            "www.pewresearch.org",
            "copyright",
            "for more information",
            "survey conducted",
            "weighting procedure",
            "telephone interviews",
        ]
        return sum(1 for c in cues if c in text) >= 2

    def _noise_penalty(self, text: str) -> float:
        cues = [
            "methodology",
            "table of contents",
            "about pew research center",
            "www.pewresearch.org",
            "copyright",
            "for more information",
            "survey conducted",
            "weighting procedure",
            "telephone interviews",
        ]
        hits = sum(1 for c in cues if c in text)
        if hits >= 3:
            return 2.2
        if hits == 2:
            return 1.4
        if hits == 1:
            return 0.6
        return 0.0

    def _extract_years(self, text: str):
        return set(re.findall(r"\b(?:19|20)\d{2}\b", text))

    def _is_scope_mismatch(self, question: str, content: str):
        q_years = self._extract_years(question)
        c_years = self._extract_years(content)
        if q_years and c_years and q_years.isdisjoint(c_years):
            return True
        return False

    def _ensure_list(self, value):
        if isinstance(value, list):
            return value
        if value is None:
            return []
        return [value]

    def _normalize_slots(self, raw_slots):
        slots = []
        for raw in self._ensure_list(raw_slots):
            if isinstance(raw, dict):
                name = str(raw.get("slot_name", raw.get("name", ""))).strip().lower()
                if name:
                    slots.append(name)
                continue
            text = str(raw).strip()
            if not text:
                continue
            try:
                parsed = json.loads(text.replace("'", '"'))
                if isinstance(parsed, dict):
                    name = str(parsed.get("slot_name", parsed.get("name", ""))).strip().lower()
                    if name:
                        slots.append(name)
                        continue
            except Exception:
                pass
            slots.append(text.lower())
        dedup = []
        for s in slots:
            if s and s not in dedup:
                dedup.append(s)
        return dedup

    def _infer_slot_coverage_from_text(self, text: str, must_slots: List[str]):
        normalized_text = str(text).lower()
        supported = []
        for slot in must_slots:
            if slot in normalized_text:
                supported.append(slot)
                continue
            terms = [t for t in re.findall(r"[a-z0-9%]+", slot) if len(t) >= 3]
            if not terms:
                continue
            hits = sum(1 for t in terms if t in normalized_text)
            token_overlap = hits / max(1, len(terms))
            fuzzy = self._fuzzy_partial_match(slot, normalized_text)
            if token_overlap >= 0.5 or fuzzy >= 0.72:
                supported.append(slot)
        missing = [s for s in must_slots if s not in supported]
        return supported, missing

    def _semantic_relevance(self, question: str, content: str, must_slots: List[str], q_terms: List[str]) -> float:
        text = str(content).lower()
        if not text.strip():
            return 0.0

        queries = self._build_semantic_queries(question, must_slots, q_terms)
        if not queries:
            queries = q_terms[:8]

        best = 0.0
        weighted_sum = 0.0
        weights = 0.0
        for q in queries:
            ql = str(q).lower().strip()
            if not ql:
                continue
            overlap = self._token_jaccard(ql, text)
            fuzzy = self._fuzzy_partial_match(ql, text)
            local = 0.65 * overlap + 0.35 * fuzzy
            w = 1.2 if len(ql.split()) >= 2 else 0.8
            weighted_sum += w * local
            weights += w
            if local > best:
                best = local

        avg = (weighted_sum / weights) if weights > 0 else 0.0
        return 0.55 * best + 0.45 * avg

    def _build_semantic_queries(self, question: str, must_slots: List[str], q_terms: List[str], max_items: int = 12) -> List[str]:
        queries = []

        # 1) Prefer informative phrases with adaptive granularity (can be 1..N words).
        for p in self._extract_phrases(question, max_items=max_items):
            pp = str(p).strip().lower()
            if pp and pp not in queries:
                queries.append(pp)
            if len(queries) >= max_items:
                return queries

        # 2) Add normalized slot constraints from Agent1.
        for s in must_slots or []:
            ss = str(s).strip().lower()
            if ss and ss not in queries:
                queries.append(ss)
            if len(queries) >= max_items:
                return queries

        # 3) Backfill with informative single terms if needed.
        for t in q_terms or []:
            tt = str(t).strip().lower()
            if len(tt) < 3:
                continue
            if tt in queries:
                continue
            queries.append(tt)
            if len(queries) >= max_items:
                return queries

        return queries

    def _extract_phrases(self, text: str, min_words: int = None, max_words: int = None, max_items: int = 8) -> List[str]:
        q = " ".join(str(text).split()).lower()
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
            "according", "one", "all", "it", "its", "their", "with", "by", "as", "at", "be", "been", "being",
        }
        connectors = {"of", "the", "and", "or", "to", "in", "for", "with", "by", "from", "among", "between"}
        weak_singletons = {"group", "people", "public", "report", "question", "survey"}

        phrases = []
        for n in range(dyn_max, dyn_min - 1, -1):
            for i in range(0, len(tokens) - n + 1):
                span = tokens[i : i + n]
                informative = [w for w in span if w not in stop and len(w) >= 3]
                if n == 1 and len(informative) < 1:
                    continue
                if n >= 2 and len(informative) < 2:
                    continue
                if span[0] in connectors or span[-1] in connectors:
                    continue
                if n == 1 and span[0] in weak_singletons:
                    continue
                phrase = " ".join(span)
                if phrase not in phrases:
                    phrases.append(phrase)
                if len(phrases) >= max_items:
                    return phrases
        return phrases

    def _token_jaccard(self, query: str, text: str) -> float:
        q_tokens = {t for t in re.findall(r"[a-z0-9%]+", query.lower()) if len(t) >= 3}
        t_tokens = {t for t in re.findall(r"[a-z0-9%]+", text.lower()) if len(t) >= 3}
        if not q_tokens or not t_tokens:
            return 0.0
        return len(q_tokens & t_tokens) / max(1, len(q_tokens | t_tokens))

    def _fuzzy_partial_match(self, query: str, text: str) -> float:
        q = str(query).lower().strip()
        t = str(text).lower().strip()
        if not q or not t:
            return 0.0

        if q in t:
            return 1.0

        q_len = len(q)
        if q_len < 6:
            return SequenceMatcher(None, q, t[: max(32, q_len * 2)]).ratio()

        max_ratio = 0.0
        window = min(max(40, int(q_len * 1.8)), len(t))
        step = max(8, q_len // 3)
        for i in range(0, max(1, len(t) - window + 1), step):
            piece = t[i : i + window]
            ratio = SequenceMatcher(None, q, piece).ratio()
            if ratio > max_ratio:
                max_ratio = ratio
            if max_ratio >= 0.9:
                break
        return max_ratio

    def _derive_support_type(self, answer_role: str, constraint_match: str):
        role = str(answer_role).lower()
        if constraint_match == "mismatch":
            return "mismatch"
        if role == "direct":
            return "direct"
        if role == "partial":
            return "partial"
        if role == "supporting":
            return "complementary"
        if role == "background":
            return "background"
        return "complementary"

    def _enum(self, value, valid_set, default):
        v = str(value).lower()
        return v if v in valid_set else default
