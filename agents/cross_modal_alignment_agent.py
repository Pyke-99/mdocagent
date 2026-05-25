import json
import re
from dataclasses import asdict
from typing import Dict, List, Tuple

from prompts.alignment_prompts import ALIGNMENT_PROMPT
from schemas.alignment_schema import Agent3Output, CrossModalRelation


class CrossModalAlignmentAgent:
    def __init__(self, model):
        self.model = model

    def run(
        self,
        question: str,
        agent1_output: Dict,
        agent2_output: Dict,
        text_chunks_by_id: Dict[str, str],
        image_chunks_by_id: Dict[str, str],
    ) -> Agent3Output:
        fallback_text, fallback_image = self._fallback_final_chunks(agent2_output)

        text_candidates = self._collect_candidates(agent2_output, modality="text", fallback_ids=fallback_text)
        image_candidates = self._collect_candidates(agent2_output, modality="image", fallback_ids=fallback_image)

        prompt = (
            ALIGNMENT_PROMPT
            + "\nQuestion:\n"
            + question
            + "\n\nAgent1 output:\n"
            + json.dumps(agent1_output, ensure_ascii=True)
            + "\n\nAgent2 output:\n"
            + json.dumps(agent2_output, ensure_ascii=True)
            + "\n\nText candidates (ordered):\n"
            + json.dumps(text_candidates, ensure_ascii=True)
            + "\n\nImage candidates (ordered):\n"
            + json.dumps(image_candidates, ensure_ascii=True)
        )

        model_texts = [text_chunks_by_id[cid] for cid in text_candidates if cid in text_chunks_by_id]
        model_images = [image_chunks_by_id[cid] for cid in image_candidates if cid in image_chunks_by_id]

        raw = ""
        try:
            raw, _, _ = self.model.predict(prompt, texts=model_texts or None, images=model_images or None)
        except Exception:
            raw = ""

        parsed = self._safe_json_parse(raw)
        if parsed:
            normalized = self._normalize(parsed)
            normalized.final_text_chunks = self._sanitize_chunk_ids(
                normalized.final_text_chunks,
                text_chunks_by_id,
                fallback_text,
            )
            normalized.final_image_chunks = self._sanitize_chunk_ids(
                normalized.final_image_chunks,
                image_chunks_by_id,
                fallback_image,
            )

            if not normalized.final_text_chunks and not normalized.final_image_chunks:
                normalized.final_text_chunks = fallback_text
                normalized.final_image_chunks = fallback_image

            normalized.cross_modal_relations = self._sanitize_relations(
                normalized.cross_modal_relations,
                normalized.final_text_chunks,
                normalized.final_image_chunks,
            )
            if not normalized.cross_modal_relations:
                normalized.cross_modal_relations = self._default_relations(
                    normalized.final_text_chunks,
                    normalized.final_image_chunks,
                )

            has_conflict = any(r.relation_type == "conflict" for r in normalized.cross_modal_relations)
            if not normalized.alignment_points:
                normalized.alignment_points = self._default_alignment_points(
                    normalized.final_text_chunks,
                    normalized.final_image_chunks,
                    has_conflict,
                )
            if not normalized.alignment_instructions:
                normalized.alignment_instructions = self._default_alignment_instructions(has_conflict)

            normalized.response_mode = self._pick_response_mode(
                normalized.response_mode,
                normalized.final_text_chunks,
                normalized.final_image_chunks,
            )
            return normalized

        return self._build_fallback_output(agent2_output, fallback_text, fallback_image)

    def to_dict(self, output: Agent3Output):
        return asdict(output)

    def _build_fallback_output(self, agent2_output: Dict, fallback_text: List[str], fallback_image: List[str]) -> Agent3Output:
        relations = self._default_relations(fallback_text, fallback_image)
        has_conflict = any(r.relation_type == "conflict" for r in relations)
        return Agent3Output(
            final_text_chunks=fallback_text,
            final_image_chunks=fallback_image,
            cross_modal_relations=relations,
            alignment_points=self._default_alignment_points(fallback_text, fallback_image, has_conflict),
            alignment_instructions=self._default_alignment_instructions(has_conflict),
            response_mode=self._pick_response_mode("", fallback_text, fallback_image),
        )

    def _normalize(self, parsed: Dict) -> Agent3Output:
        relations: List[CrossModalRelation] = []
        for item in parsed.get("cross_modal_relations", []):
            if not isinstance(item, dict):
                continue
            relations.append(
                CrossModalRelation(
                    text_chunk_id=str(item.get("text_chunk_id", "")).strip(),
                    image_chunk_id=str(item.get("image_chunk_id", "")).strip(),
                    relation_type=self._enum(
                        item.get("relation_type"),
                        {"align", "complement", "verify", "conflict", "weak_related"},
                        "weak_related",
                    ),
                    note=str(item.get("note", "")).strip(),
                )
            )

        return Agent3Output(
            final_text_chunks=self._ensure_list(parsed.get("final_text_chunks")),
            final_image_chunks=self._ensure_list(parsed.get("final_image_chunks")),
            cross_modal_relations=relations,
            rewritten_evidence=[],
            alignment_points=self._ensure_list(parsed.get("alignment_points")),
            alignment_instructions=self._ensure_list(parsed.get("alignment_instructions")),
            response_mode=self._enum(
                parsed.get("response_mode"),
                {"text-first", "image-first", "multimodal-joint"},
                "multimodal-joint",
            ),
        )

    def _collect_candidates(self, agent2_output: Dict, modality: str, fallback_ids: List[str]) -> List[str]:
        decisions = agent2_output.get("chunk_decisions", []) if isinstance(agent2_output.get("chunk_decisions", []), list) else []

        use_ids = [
            str(c.get("chunk_id", "")).strip()
            for c in decisions
            if isinstance(c, dict) and c.get("modality") == modality and c.get("decision") == "use"
        ]
        reserve_ids = [
            str(c.get("chunk_id", "")).strip()
            for c in decisions
            if isinstance(c, dict) and c.get("modality") == modality and c.get("decision") == "reserve"
        ]

        ordered = []
        for cid in use_ids + reserve_ids + list(fallback_ids):
            if cid and cid not in ordered:
                ordered.append(cid)

        limit = 6 if modality == "text" else 4
        return ordered[:limit]

    def _default_relations(self, text_ids: List[str], image_ids: List[str]) -> List[CrossModalRelation]:
        if not text_ids or not image_ids:
            return []

        relations: List[CrossModalRelation] = []
        pair_count = min(len(text_ids), len(image_ids))
        for idx in range(pair_count):
            rel_type = "verify" if idx == 0 else "complement"
            relations.append(
                CrossModalRelation(
                    text_chunk_id=text_ids[idx],
                    image_chunk_id=image_ids[idx],
                    relation_type=rel_type,
                    note="fallback pairing by candidate rank",
                )
            )
        return relations

    def _sanitize_relations(
        self,
        relations: List[CrossModalRelation],
        final_text_ids: List[str],
        final_image_ids: List[str],
    ) -> List[CrossModalRelation]:
        valid_text = set(final_text_ids)
        valid_image = set(final_image_ids)

        cleaned: List[CrossModalRelation] = []
        seen: set[Tuple[str, str, str]] = set()
        for rel in relations:
            t_id = str(rel.text_chunk_id).strip()
            i_id = str(rel.image_chunk_id).strip()
            if t_id and t_id not in valid_text:
                continue
            if i_id and i_id not in valid_image:
                continue
            if not t_id and not i_id:
                continue
            key = (t_id, i_id, rel.relation_type)
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(rel)
        return cleaned

    def _default_alignment_points(self, text_ids: List[str], image_ids: List[str], has_conflict: bool) -> List[str]:
        points = [
            f"final_text_chunks={text_ids}",
            f"final_image_chunks={image_ids}",
        ]
        if has_conflict:
            points.append("cross-modal conflict detected; keep contradictory evidence explicit for Agent4")
        else:
            points.append("no explicit cross-modal conflict detected from selected chunks")
        return points

    def _default_alignment_instructions(self, has_conflict: bool) -> List[str]:
        if has_conflict:
            return [
                "compare text and image claims side by side and preserve conflicts in final reasoning",
                "prefer claims supported by both modalities; downgrade unsupported conflicting claims",
                "if conflict cannot be resolved, answer conservatively and mark answer_status as conflicted or partially_answerable",
            ]
        return [
            "fuse mutually supportive text and image evidence before final answering",
            "treat single-modality claims as weaker unless verified by another selected chunk",
            "do not expand evidence set in Agent4; answer only with selected aligned chunks",
        ]

    def _fallback_final_chunks(self, agent2_output: Dict) -> Tuple[List[str], List[str]]:
        chunk_decisions = agent2_output.get("chunk_decisions", []) if isinstance(agent2_output.get("chunk_decisions", []), list) else []

        text_use = [
            str(c.get("chunk_id", "")).strip()
            for c in chunk_decisions
            if isinstance(c, dict) and c.get("modality") == "text" and c.get("decision") == "use"
        ]
        image_use = [
            str(c.get("chunk_id", "")).strip()
            for c in chunk_decisions
            if isinstance(c, dict) and c.get("modality") == "image" and c.get("decision") == "use"
        ]

        if text_use or image_use:
            return text_use, image_use

        text_reserve = [
            str(c.get("chunk_id", "")).strip()
            for c in chunk_decisions
            if isinstance(c, dict) and c.get("modality") == "text" and c.get("decision") == "reserve"
        ][:2]
        image_reserve = [
            str(c.get("chunk_id", "")).strip()
            for c in chunk_decisions
            if isinstance(c, dict) and c.get("modality") == "image" and c.get("decision") == "reserve"
        ][:2]

        return text_reserve, image_reserve

    def _sanitize_chunk_ids(self, ids: List[str], chunk_map: Dict[str, str], fallback_ids: List[str]) -> List[str]:
        cleaned = []
        for cid in ids or []:
            key = str(cid).strip()
            if key in chunk_map and key not in cleaned:
                cleaned.append(key)
        if cleaned:
            return cleaned
        return [cid for cid in fallback_ids if cid in chunk_map]

    def _pick_response_mode(self, raw_mode: str, text_ids: List[str], image_ids: List[str]) -> str:
        mode = str(raw_mode or "").strip().lower()
        if mode in {"text-first", "image-first", "multimodal-joint"}:
            return mode
        if text_ids and image_ids:
            return "multimodal-joint"
        if text_ids:
            return "text-first"
        return "image-first"

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
            return [str(v).strip() for v in value if str(v).strip()]
        if value is None:
            return []
        text = str(value).strip()
        return [text] if text else []

    def _enum(self, value, valid_set, default):
        v = str(value).strip().lower()
        return v if v in valid_set else default
