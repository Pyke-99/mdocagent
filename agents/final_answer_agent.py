import json
import re
from dataclasses import asdict
from typing import Dict, List, Tuple

from prompts.final_answer_prompts import FINAL_ANSWER_PROMPT
from schemas.answer_schema import Agent4Output


class FinalAnswerAgent:
    def __init__(self, model):
        self.model = model

    def _normalize_chunk_ids(self, values) -> List[str]:
        if not isinstance(values, list):
            return []
        return [str(cid).strip() for cid in values if str(cid).strip()]

    def _split_route_evidence_ids(self, values: List[str]) -> Tuple[List[str], List[str]]:
        text_ids = []
        image_ids = []
        for cid in values:
            if cid.startswith("text_"):
                text_ids.append(cid)
            elif cid.startswith("image_"):
                image_ids.append(cid)
        return text_ids, image_ids

    def run(
        self,
        question: str,
        agent1_output: Dict,
        agent3_output: Dict,
        agent2_output: Dict = None,
        text_chunks_by_id: Dict = None,
        image_chunks_by_id: Dict = None,
        input_mode: str = "rewritten_only",
    ) -> Agent4Output:
        agent2_output = agent2_output or {}
        text_chunks_by_id = text_chunks_by_id or {}
        image_chunks_by_id = image_chunks_by_id or {}
        mode = str(input_mode or "rewritten_only").strip().lower()

        final_text_chunks = self._normalize_chunk_ids(agent3_output.get("final_text_chunks", []))
        final_image_chunks = self._normalize_chunk_ids(agent3_output.get("final_image_chunks", []))

        route_candidate_answer = str(agent3_output.get("candidate_answer", "")).strip()
        route_used_evidence = self._normalize_chunk_ids(agent3_output.get("used_evidence", agent3_output.get("used_chunks", [])))
        route_text_chunks, route_image_chunks = self._split_route_evidence_ids(route_used_evidence)

        if not final_text_chunks and route_text_chunks:
            final_text_chunks = route_text_chunks
        if not final_image_chunks and route_image_chunks:
            final_image_chunks = route_image_chunks

        if not final_text_chunks and not final_image_chunks:
            final_text_chunks = list(text_chunks_by_id.keys())
            final_image_chunks = list(image_chunks_by_id.keys())

        used_chunks = list(dict.fromkeys(final_text_chunks + final_image_chunks + route_used_evidence))

        selected_text_inputs = [text_chunks_by_id[cid] for cid in final_text_chunks if cid in text_chunks_by_id]
        selected_image_inputs = [image_chunks_by_id[cid] for cid in final_image_chunks if cid in image_chunks_by_id]

        evidence_texts = []
        if route_candidate_answer:
            evidence_texts.append("Route candidate answer: " + route_candidate_answer)
        evidence_texts.extend(selected_text_inputs)
        if not evidence_texts and route_candidate_answer:
            evidence_texts.append(route_candidate_answer)

        alignment_payload = {
            "cross_modal_relations": agent3_output.get("cross_modal_relations", []),
            "alignment_points": agent3_output.get("alignment_points", []),
            "alignment_instructions": agent3_output.get("alignment_instructions", []),
            "response_mode": agent3_output.get("response_mode", ""),
        }
        evidence_texts.append("Alignment package: " + json.dumps(alignment_payload, ensure_ascii=True))

        must_check_slots = self._normalize_slots(agent1_output.get("must_check_slots"))
        if not must_check_slots:
            must_check_slots = self._normalize_slots(agent1_output.get("hard_constraints"))
        coverage_inputs = selected_text_inputs[:]
        if not coverage_inputs and route_candidate_answer:
            coverage_inputs = [route_candidate_answer]
        filled_slots, slot_coverage = self._compute_slot_coverage(must_check_slots, coverage_inputs)
        unresolved_slots = [s for s in must_check_slots if s not in filled_slots]
        expected_status = self._expected_status(slot_coverage, coverage_inputs, selected_image_inputs, agent3_output)
        task_type = str(agent1_output.get("task_type", "extract")).lower()

        prompt = (
            FINAL_ANSWER_PROMPT
            + "\nQuestion:\n"
            + question
            + "\n\nAgent1 output:\n"
            + json.dumps(agent1_output, ensure_ascii=True)
            + "\n\nAgent2 output:\n"
            + json.dumps(agent2_output, ensure_ascii=True)
            + "\n\nAgent3 output:\n"
            + json.dumps(agent3_output, ensure_ascii=True)
            + "\n\nOriginal chunks of selected evidence:\n"
            + json.dumps(self._collect_original_chunks(used_chunks, text_chunks_by_id, image_chunks_by_id), ensure_ascii=True)
            + "\n\nAgent4 input mode:\n"
            + mode
            + "\n\nAnswer policy:\n"
            + f"- Slot coverage estimate: {slot_coverage:.2f}\n"
            + f"- Expected status target: {expected_status}\n"
            + "- If evidence is sufficient, do not output generic refusal text.\n"
            + "- If truly insufficient, set answer_status to unanswerable or partially_answerable.\n"
            + "- Follow Agent3 alignment instructions and handle cross-modal conflict explicitly when present.\n"
        )

        raw = ""
        try:
            raw, _, token_usage = self.model.predict(prompt, texts=evidence_texts or None, images=selected_image_inputs or None)
        except Exception:
            raw = ""
            token_usage = None

        parsed = self._safe_json_parse(raw)
        if parsed:
            output = self._normalize(parsed)
            return self._post_validate_output(
                output,
                expected_status,
                slot_coverage,
                used_chunks,
                filled_slots,
                unresolved_slots,
                task_type,
            )

        if expected_status == "answerable" and self._is_refusal(raw):
            raw = self._force_answer_retry(
                question,
                agent1_output,
                agent2_output,
                agent3_output,
                evidence_texts,
                selected_image_inputs,
                used_chunks,
                text_chunks_by_id,
                image_chunks_by_id,
                mode,
            )
            parsed_retry = self._safe_json_parse(raw)
            if parsed_retry:
                output = self._normalize(parsed_retry)
                return self._post_validate_output(
                    output,
                    expected_status,
                    slot_coverage,
                    used_chunks,
                    filled_slots,
                    unresolved_slots,
                    task_type,
                )

        nl_answer, nl_status, nl_confidence = self._extract_nl_fields(raw)
        fallback_answer = nl_answer or (raw.strip() if isinstance(raw, str) else "")
        if not fallback_answer:
            fallback_answer = "Insufficient information to answer the question."

        status = nl_status or expected_status
        if self._is_refusal(fallback_answer):
            status = "partially_answerable" if slot_coverage > 0 else "unanswerable"
        if nl_confidence is None:
            confidence = self._estimate_confidence(slot_coverage, status, len(used_chunks), self._is_refusal(fallback_answer))
        else:
            confidence = max(0.0, min(1.0, nl_confidence))

        return Agent4Output(
            final_answer=fallback_answer,
            answer_status=status,
            used_chunks=used_chunks,
            filled_slots=filled_slots,
            unresolved_slots=unresolved_slots,
            confidence=confidence,
            token_usage=token_usage,
        )

    def to_dict(self, output: Agent4Output):
        return asdict(output)

    def _normalize(self, parsed: Dict) -> Agent4Output:
        raw_status = parsed.get("answer_status", parsed.get("status"))
        status = self._enum(
            raw_status,
            {"answerable", "partially_answerable", "unanswerable", "conflicted", "conflicting"},
            "partially_answerable",
        )
        if status == "conflicting":
            status = "conflicted"
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        return Agent4Output(
            final_answer=str(parsed.get("final_answer", "")),
            answer_status=status,
            used_chunks=self._ensure_list(parsed.get("used_chunks", parsed.get("used_evidence", []))),
            filled_slots=self._normalize_slots(parsed.get("filled_slots", [])),
            unresolved_slots=self._normalize_slots(parsed.get("unresolved_slots", [])),
            confidence=confidence,
            token_usage=None,
        )

    def _post_validate_output(
        self,
        output: Agent4Output,
        expected_status: str,
        slot_coverage: float,
        fallback_used_chunks: List[str],
        fallback_filled_slots: List[str],
        fallback_unresolved_slots: List[str],
        task_type: str,
    ) -> Agent4Output:
        if not output.used_chunks:
            output.used_chunks = fallback_used_chunks
        if not output.filled_slots:
            output.filled_slots = fallback_filled_slots
        if not output.unresolved_slots:
            output.unresolved_slots = fallback_unresolved_slots

        refusal = self._is_refusal(output.final_answer)
        if refusal and output.answer_status == "answerable":
            output.answer_status = "partially_answerable" if slot_coverage > 0 else "unanswerable"
        elif expected_status == "answerable" and output.answer_status == "unanswerable" and slot_coverage >= 0.5:
            output.answer_status = "partially_answerable"

        # For count/ratio/comparison tasks, unresolved core slots should not remain fully answerable.
        if task_type in {"count", "ratio", "comparison"} and output.unresolved_slots and output.answer_status == "answerable":
            output.answer_status = "partially_answerable"

        output.confidence = self._estimate_confidence(slot_coverage, output.answer_status, len(output.used_chunks), refusal)
        return output

    def _force_answer_retry(
        self,
        question: str,
        agent1_output: Dict,
        agent2_output: Dict,
        agent3_output: Dict,
        evidence_texts: List[str],
        selected_image_inputs: List[str],
        used_chunks: List[str],
        text_chunks_by_id: Dict,
        image_chunks_by_id: Dict,
        mode: str,
    ) -> str:
        retry_prompt = (
            FINAL_ANSWER_PROMPT
            + "\nQuestion:\n"
            + question
            + "\n\nAgent1 output:\n"
            + json.dumps(agent1_output, ensure_ascii=True)
            + "\n\nAgent2 output:\n"
            + json.dumps(agent2_output, ensure_ascii=True)
            + "\n\nAgent3 output:\n"
            + json.dumps(agent3_output, ensure_ascii=True)
            + "\n\nOriginal chunks of selected evidence:\n"
            + json.dumps(self._collect_original_chunks(used_chunks, text_chunks_by_id, image_chunks_by_id), ensure_ascii=True)
            + "\n\nAgent4 input mode:\n"
            + mode
            + "\n\nStrict instruction:\n"
            + "- Do not use generic refusal text.\n"
            + "- Give the best supported answer and set answer_status accordingly.\n"
            + "- If evidence is weak, choose partially_answerable instead of unanswerable.\n"
        )
        try:
            raw, _, _ = self.model.predict(retry_prompt, texts=evidence_texts or None, images=selected_image_inputs or None)
            return raw or ""
        except Exception:
            return ""

    def _compute_slot_coverage(self, slots: List[str], selected_texts: List[str]) -> Tuple[List[str], float]:
        if not slots:
            return [], 0.0
        corpus = "\n".join(selected_texts).lower()
        filled = []
        for slot in slots:
            slot_str = str(slot).strip().lower()
            if not slot_str:
                continue
            if slot_str in corpus:
                filled.append(slot)
                continue
            terms = [t for t in re.findall(r"[a-z0-9%]+", slot_str) if len(t) >= 3]
            if terms and sum(1 for t in terms if t in corpus) >= max(1, (len(terms) + 1) // 2):
                filled.append(slot)
        coverage = len(filled) / max(1, len([s for s in slots if str(s).strip()]))
        return filled, coverage

    def _expected_status(self, slot_coverage: float, selected_texts: List[str], selected_images: List[str], agent3_output: Dict) -> str:
        has_evidence = bool(selected_texts or selected_images)
        if not has_evidence:
            return "unanswerable"

        conflict_relations = [
            r for r in agent3_output.get("cross_modal_relations", []) if str(r.get("relation_type", "")).lower() == "conflict"
        ]
        explicit_conflicts = agent3_output.get("conflicts", [])
        if conflict_relations or explicit_conflicts:
            return "conflicted"

        if slot_coverage >= 0.6:
            return "answerable"
        # Soft-slot policy: low slot coverage with available evidence should remain partially answerable.
        return "partially_answerable"

    def _estimate_confidence(self, slot_coverage: float, status: str, used_count: int, refusal: bool) -> float:
        status_bias = {
            "answerable": 0.2,
            "partially_answerable": 0.08,
            "conflicted": 0.05,
            "unanswerable": 0.02,
        }.get(status, 0.05)
        evidence_bonus = min(0.2, 0.03 * used_count)
        conf = 0.15 + 0.55 * slot_coverage + status_bias + evidence_bonus
        if refusal:
            conf -= 0.25
        return max(0.0, min(1.0, conf))

    def _is_refusal(self, text: str) -> bool:
        s = str(text or "").strip().lower()
        refusal_markers = [
            "insufficient information",
            "cannot answer",
            "can't answer",
            "not enough information",
            "unable to determine",
            "无法回答",
            "信息不足",
        ]
        return any(m in s for m in refusal_markers)

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

    def _extract_nl_fields(self, raw: str):
        if not isinstance(raw, str) or not raw.strip():
            return "", None, None

        text = raw.strip()
        cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()

        answer = ""
        status = None
        confidence = None

        m_answer = re.search(r"(?:^|\n)\s*final\s*answer\s*:\s*(.+)", cleaned, flags=re.IGNORECASE)
        if m_answer:
            answer = m_answer.group(1).strip()

        m_status = re.search(r"(?:^|\n)\s*answer\s*status\s*:\s*([a-z_]+)", cleaned, flags=re.IGNORECASE)
        if m_status:
            candidate = m_status.group(1).strip().lower()
            if candidate in {"answerable", "partially_answerable", "unanswerable", "conflicted", "conflicting"}:
                if candidate == "conflicting":
                    candidate = "conflicted"
                status = candidate

        m_conf = re.search(r"(?:^|\n)\s*confidence\s*:\s*([0-9]*\.?[0-9]+)", cleaned, flags=re.IGNORECASE)
        if m_conf:
            try:
                confidence = float(m_conf.group(1))
            except Exception:
                confidence = None

        if not answer:
            for line in cleaned.splitlines():
                s = line.strip()
                if not s:
                    continue
                if s.startswith("{") or s.startswith("["):
                    continue
                answer = s
                break

        return answer, status, confidence

    def _enum(self, value, valid_set, default):
        v = str(value).lower()
        return v if v in valid_set else default

    def _short_text(self, value: str, max_len: int = 300) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def _collect_original_chunks(self, used_chunks: List[str], text_chunks_by_id: Dict, image_chunks_by_id: Dict):
        records = []
        for cid in used_chunks:
            if cid in text_chunks_by_id:
                records.append(
                    {
                        "chunk_id": cid,
                        "modality": "text",
                        "content": self._short_text(text_chunks_by_id.get(cid, ""), max_len=900),
                    }
                )
            elif cid in image_chunks_by_id:
                records.append(
                    {
                        "chunk_id": cid,
                        "modality": "image",
                        "content": self._short_text(image_chunks_by_id.get(cid, ""), max_len=300),
                    }
                )
        return records
