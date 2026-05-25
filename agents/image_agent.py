from typing import Dict, List

from agents.base_agent import Agent


class ImageAgent(Agent):
    refusal_message = "I cannot answer the question based on the image information."

    def answer(self, question: str, image_chunks: List[Dict[str, str]], reorder_result: Dict = None):
        if reorder_result is None:
            return self._answer_without_reorder(question, image_chunks)

        selected_ids = reorder_result.get("ranking", {}).get("selected_image_ids", []) or []
        meta = {item.get("id"): item for item in reorder_result.get("image_results", []) if item.get("id")}
        image_map = {chunk["id"]: chunk["content"] for chunk in image_chunks}

        selected = []
        for cid in selected_ids:
            path = image_map.get(cid)
            if path is None:
                continue
            info = meta.get(cid, {})
            relevance = str(info.get("relevance", "irrelevant")).lower()
            try:
                score = int(info.get("support_score", 0))
            except Exception:
                score = 0
            if relevance != "relevant" or score <= 0:
                continue
            selected.append(
                {
                    "id": cid,
                    "score": max(0, min(3, score)),
                    "reason": info.get("support_reason", ""),
                    "content": path,
                }
            )

        if not selected:
            return self.refusal_message, []

        ordered = sorted(selected, key=lambda x: x["score"], reverse=True)
        prompt = self._build_prompt(question, ordered, reorder_result.get("relations", {}))
        images_payload = [item["content"] for item in ordered]
        response, messages, _ = self.predict(prompt, texts=None, images=images_payload, with_sys_prompt=True)
        return response, messages

    def _answer_without_reorder(self, question: str, image_chunks: List[Dict[str, str]]):
        if not image_chunks:
            return self.refusal_message, []

        selected = []
        for chunk in image_chunks:
            cid = chunk.get("id")
            path = chunk.get("content")
            if cid is None or path is None:
                continue
            selected.append(
                {
                    "id": cid,
                    "score": 2,
                    "reason": "retrieval-selected",
                    "content": path,
                }
            )

        if not selected:
            return self.refusal_message, []

        prompt = self._build_prompt(question, selected, {})
        images_payload = [item["content"] for item in selected]
        response, messages, _ = self.predict(prompt, texts=None, images=images_payload, with_sys_prompt=True)
        return response, messages

    def _build_prompt(self, question: str, ordered_chunks: List[Dict[str, str]], relations: Dict) -> str:
        evidence_lines = "\n".join(
            [f"- {c['id']} (score={c['score']}, reason={c['reason']}); image order matches listed sequence." for c in ordered_chunks]
        )
        recommended_sets = relations.get("recommended_sets", [])
        conflict_pairs = relations.get("conflict_pairs", [])
        weak_evidence = relations.get("weak_evidence", [])
        relation_lines = [
            "Recommended sets:" + (" none" if not recommended_sets else ""),
        ]
        for item in recommended_sets:
            relation_lines.append(f"  * members={item.get('members', [])}, reason={item.get('reason', '')}")
        relation_lines.append("Conflict pairs:" + (" none" if not conflict_pairs else ""))
        for item in conflict_pairs:
            relation_lines.append(
                f"  * {item.get('id1', '')} vs {item.get('id2', '')}, reason={item.get('reason', '')}"
            )
        relation_lines.append("Weak evidence:" + (" none" if not weak_evidence else ""))
        for item in weak_evidence:
            relation_lines.append(f"  * {item.get('id', '')}: {item.get('reason', '')}")
        relation_block = "\n".join(relation_lines)

        return (
            f"Question: {question}\n"
            "Use the selected images only. Prioritize higher scores (3 > 2 > 1) and confirm details directly from the images/OCR; do not rely solely on the provided reasons.\n"
            "If evidence conflicts, explain briefly. If you cannot answer using the selected images, reply exactly with the refusal message.\n"
            "Selected image summary (order matches attached images):\n"
            f"{evidence_lines}\n\n"
            "Relations:\n"
            f"{relation_block}"
        )
