from typing import Dict, List, Tuple
import json
import re

from agents.base_agent import Agent


class ReorderAgent(Agent):
    """Agent that classifies, scores, and ranks text/image chunks."""

    def reorder(
        self,
        question: str,
        text_chunks: List[Dict[str, str]],
        image_chunks: List[Dict[str, str]],
        top_k_text_after_rerank: int,
        top_k_image_after_rerank: int,
    ) -> Tuple[Dict, List[Dict]]:
        # Text branch uses deterministic content scoring to avoid JSON-format collapse.
        text_results = self._score_text_chunks(question, text_chunks)

        image_payload = [chunk["content"] for chunk in image_chunks]
        image_prompt = self._build_image_prompt(question, image_chunks)
        image_raw, image_messages = self.predict(
            image_prompt,
            texts=None,
            images=image_payload or None,
            with_sys_prompt=True,
        )

        image_parsed = self._safe_json_parse(image_raw)
        parsed = {
            "text_results": text_results,
            "image_results": (image_parsed or {}).get("image_results", []),
        }
        normalized = self._normalize_result(parsed, text_chunks, image_chunks)

        # Image fallback: if parser fails or no relevant image, keep top retrieval images as weak evidence.
        if not any(i.get("relevance") == "relevant" for i in normalized.get("image_results", [])):
            normalized["image_results"] = self._fallback_image_results(image_chunks)

        ranked = self._compute_ranking(normalized, text_chunks, image_chunks, top_k_text_after_rerank, top_k_image_after_rerank)

        debug_info = {
            "raw_text_response": "rule-based-text-scoring",
            "raw_image_response": image_raw,
            "parse_ok_text": True,
            "parse_ok_image": bool(image_parsed),
            "text_fallback_used": False,
            "image_fallback_used": not any(i.get("relevance") == "relevant" for i in self._normalize_result(parsed, text_chunks, image_chunks).get("image_results", [])),
            "normalized": normalized,
            "relevant_text": len([t for t in normalized.get("text_results", []) if t.get("relevance") == "relevant"]),
            "relevant_image": len([i for i in normalized.get("image_results", []) if i.get("relevance") == "relevant"]),
        }
        try:
            import json

            print("[Reorder][raw][text]", "rule-based-text-scoring")
            print("[Reorder][raw][image]", image_raw)
            print("[Reorder][normalized]", json.dumps(debug_info, ensure_ascii=True))
        except Exception:
            pass
        ranked["debug"] = debug_info
        messages = []
        if isinstance(image_messages, list):
            messages.extend(image_messages)
        return ranked, messages

    def _format_text_chunk(self, chunk: Dict[str, str], max_len: int = 600) -> str:
        content = chunk.get("content", "")
        if len(content) > max_len:
            content = content[:max_len] + "..."
        return f"[{chunk.get('id')}] {content}"

    def _build_image_prompt(
        self,
        question: str,
        image_chunks: List[Dict[str, str]],
    ) -> str:
        image_list = "\n".join([f"- image_{idx}: attached image #{idx+1}" for idx, _ in enumerate(image_chunks)])
        if not image_list:
            image_list = "- None"
        return (
            "Task: Label relevance and support scores for IMAGE chunks only.\n"
            f"Question: {question}\n\n"
            "Image chunks (id: order note):\n"
            f"{image_list}\n\n"
            "Use the attached images as primary evidence. If uncertain but likely related, prefer score 1 instead of discarding.\n"
            "Output JSON only, no markdown or explanations outside JSON.\n"
            "Set text_results to an empty array. Do not output coordinates/ranking/relations.\n"
            "For each image item use: {id: string, relevance: 'relevant'|'irrelevant', support_score: 0|1|2|3, support_reason: string}.\n"
            "If relevance is 'irrelevant', set support_score=0 and support_reason=\"\".\n"
            "Prefer sparse output and keep support_reason short (<= 12 words).\n"
            "Return exact schema: {\"text_results\": [], \"image_results\": [...]}\n"
        )

    def _safe_json_parse(self, text: str) -> Dict:
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            pass

        # Try outermost braces
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                pass

        # Regex-extract flat object snippets even if root braces are unbalanced
        objs = []
        for m in re.finditer(r"\{[^{}]*\}", text):
            snippet = m.group(0)
            try:
                obj = json.loads(snippet)
                objs.append(obj)
            except Exception:
                continue
        if objs:
            text_items = [o for o in objs if str(o.get("id", "")).startswith("text_")]
            image_items = [o for o in objs if str(o.get("id", "")).startswith("image_")]
            return {"text_results": text_items, "image_results": image_items}

        return {}

    def _normalize_result(
        self,
        result: Dict,
        text_chunks: List[Dict[str, str]],
        image_chunks: List[Dict[str, str]],
    ) -> Dict:
        result = result or {}
        text_results = self._normalize_chunk_results(result.get("text_results", []), text_chunks)
        image_results = self._normalize_chunk_results(result.get("image_results", []), image_chunks)
        return {
            "text_results": text_results,
            "image_results": image_results,
        }

    def _normalize_chunk_results(self, raw_results: List[Dict], chunks: List[Dict[str, str]]) -> List[Dict]:
        raw_map = {item.get("id"): item for item in raw_results if item.get("id") is not None}
        normalized = []
        for chunk in chunks:
            item = dict(raw_map.get(chunk["id"], {}))
            item_id = chunk["id"]

            raw_relevance = str(item.get("relevance", "irrelevant")).lower()
            if raw_relevance.startswith("ir("):
                raw_relevance = "irrelevant"
            raw_score = item.get("support_score", 0)
            try:
                score = int(raw_score)
            except Exception:
                score = 0
            reason = item.get("support_reason", "")
            if not isinstance(reason, str):
                reason = str(reason) if reason is not None else ""

            relevance = raw_relevance if raw_relevance in ("relevant", "irrelevant") else None
            if relevance is None:
                if score >= 2 and reason:
                    relevance = "relevant"
                elif reason and "no relevant" not in reason.lower():
                    relevance = "relevant"
                    if score <= 0:
                        score = 1
                else:
                    relevance = "irrelevant"

            score = max(0, min(3, score))
            if relevance == "irrelevant":
                score = 0
                reason = ""

            normalized.append(
                {
                    "id": item_id,
                    "relevance": relevance,
                    "support_score": score,
                    "support_reason": reason,
                }
            )
        return normalized

    def _compute_ranking(
        self,
        normalized: Dict,
        text_chunks: List[Dict[str, str]],
        image_chunks: List[Dict[str, str]],
        top_k_text_after_rerank: int,
        top_k_image_after_rerank: int,
    ) -> Dict:
        text_results = normalized.get("text_results", [])
        image_results = normalized.get("image_results", [])

        text_order = {chunk["id"]: idx for idx, chunk in enumerate(text_chunks)}
        image_order = {chunk["id"]: idx for idx, chunk in enumerate(image_chunks)}

        def rank_items(items, order_map):
            relevant_items = [i for i in items if i.get("relevance") == "relevant"]
            return [
                i["id"]
                for i in sorted(
                    relevant_items,
                    key=lambda x: (-x.get("support_score", 0), order_map.get(x.get("id"), 1e9)),
                )
            ]

        text_ranked_ids = rank_items(text_results, text_order)
        image_ranked_ids = rank_items(image_results, image_order)

        def select_top(ranked_ids, k):
            if k is None or k <= 0:
                return ranked_ids
            return ranked_ids[:k]

        ranking = {
            "text_ranked_ids": text_ranked_ids,
            "image_ranked_ids": image_ranked_ids,
            "selected_text_ids": select_top(text_ranked_ids, top_k_text_after_rerank),
            "selected_image_ids": select_top(image_ranked_ids, top_k_image_after_rerank),
        }

        return {
            "text_results": text_results,
            "image_results": image_results,
            "ranking": ranking,
            "relations": {"recommended_sets": [], "conflict_pairs": [], "weak_evidence": []},
        }

    def _score_text_chunks(self, question: str, text_chunks: List[Dict[str, str]]) -> List[Dict]:
        # Deterministic content-based scoring for text chunks.
        q_terms = self._extract_terms(question)
        scored = []
        for chunk in text_chunks:
            cid = chunk.get("id", "")
            content = str(chunk.get("content", "")).lower()
            hits = sum(1 for term in q_terms if term in content)
            score = 0
            if hits >= 3:
                score = 3
            elif hits == 2:
                score = 2
            elif hits == 1:
                score = 1
            scored.append((cid, hits, score))

        # Guarantee at least a few candidates are available for downstream answer agents.
        top_by_hits = sorted(scored, key=lambda x: (-x[1], x[0]))
        selected_ids = {cid for cid, hits, _ in top_by_hits[:4] if hits > 0}
        if not selected_ids:
            selected_ids = {cid for cid, _, _ in top_by_hits[:2]}

        results = []
        score_map = {cid: score for cid, _, score in scored}
        hit_map = {cid: hits for cid, hits, _ in scored}
        for chunk in text_chunks:
            cid = chunk.get("id", "")
            if cid in selected_ids:
                s = max(1, score_map.get(cid, 1))
                h = hit_map.get(cid, 0)
                results.append(
                    {
                        "id": cid,
                        "relevance": "relevant",
                        "support_score": s,
                        "support_reason": f"keyword overlap={h}",
                    }
                )
            else:
                results.append(
                    {
                        "id": cid,
                        "relevance": "irrelevant",
                        "support_score": 0,
                        "support_reason": "",
                    }
                )
        return results

    def _fallback_image_results(self, image_chunks: List[Dict[str, str]]) -> List[Dict]:
        # Keep a small weak-evidence set when image parser fails.
        keep = {chunk.get("id", "") for chunk in image_chunks[:2]}
        results = []
        for chunk in image_chunks:
            cid = chunk.get("id", "")
            if cid in keep:
                results.append(
                    {
                        "id": cid,
                        "relevance": "relevant",
                        "support_score": 1,
                        "support_reason": "image fallback by retrieval order",
                    }
                )
            else:
                results.append(
                    {
                        "id": cid,
                        "relevance": "irrelevant",
                        "support_score": 0,
                        "support_reason": "",
                    }
                )
        return results

    def _extract_terms(self, question: str) -> List[str]:
        terms = re.findall(r"[a-zA-Z0-9%]+", str(question).lower())
        stop = {
            "the", "is", "are", "was", "were", "what", "which", "who", "whom", "whose", "how",
            "from", "this", "that", "with", "for", "and", "or", "among", "into", "their", "there",
            "report", "according", "question", "compared", "between",
        }
        dedup = []
        for t in terms:
            if len(t) < 2:
                continue
            if t in stop:
                continue
            if t not in dedup:
                dedup.append(t)
        return dedup[:20]

    def _ensure_rank(self, rank_ids: List[str], results: List[Dict]) -> List[str]:
        if rank_ids:
            return [rid for rid in rank_ids if any(rid == r.get("id") and r.get("relevance") == "relevant" for r in results)]
        sorted_results = sorted(
            [r for r in results if r.get("relevance") == "relevant"],
            key=lambda x: (x.get("support_score", 0), x.get("support_reason", "")),
            reverse=True,
        )
        return [r["id"] for r in sorted_results]

    def _ensure_selection(self, selected: List[str], ranked: List[str], limit: int) -> List[str]:
        if selected:
            return selected[:limit] if limit else selected
        if limit <= 0:
            return ranked
        return ranked[:limit]
