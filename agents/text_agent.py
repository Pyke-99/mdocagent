from typing import Dict, List

from agents.base_agent import Agent


class TextAgent(Agent):
    refusal_message = "I cannot answer the question based on the text information."

    def answer(self, question: str, text_chunks: List[Dict[str, str]], reorder_result: Dict):
        selected_ids = reorder_result.get("ranking", {}).get("selected_text_ids", []) or []
        meta = {item.get("id"): item for item in reorder_result.get("text_results", []) if item.get("id")}
        text_map = {chunk["id"]: chunk["content"] for chunk in text_chunks}

        selected = []
        for cid in selected_ids:
            content = text_map.get(cid)
            if content is None:
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
                    "content": content,
                }
            )

        if not selected:
            return self.refusal_message, []

        ordered = sorted(selected, key=lambda x: x["score"], reverse=True)
        prompt = self._build_prompt(question, ordered, reorder_result.get("relations", {}))
        texts_payload = [self._format_text(item) for item in ordered]
        response, messages = self.predict(prompt, texts=texts_payload, images=None, with_sys_prompt=True)
        return response, messages

    def _format_text(self, item: Dict[str, str]) -> str:
        return f"[{item['id']}] (score={item['score']}) {item['content']}"

    def _build_prompt(self, question: str, ordered_chunks: List[Dict[str, str]], relations: Dict) -> str:
        evidence_lines = "\n".join(
            [f"- {c['id']} (score={c['score']}, reason={c['reason']})" for c in ordered_chunks]
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
            "Use the selected text chunks below. Prioritize higher scores (3 > 2 > 1). Verify claims directly against the raw text; do not rely solely on the provided reasons.\n"
            "If evidence conflicts, explain briefly. If you cannot answer using the selected text, reply exactly with the refusal message.\n"
            "Selected text summary (in order):\n"
            f"{evidence_lines}\n\n"
            "Relations:\n"
            f"{relation_block}"
        )
