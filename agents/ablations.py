from agents.mdoc_agent import MDocAgent


class MDAi(MDocAgent):
    def __init__(self, config):
        super().__init__(config)
    
    def predict(self, sample, question, texts, images):
        text_chunks = [{"id": f"text_{idx}", "content": txt} for idx, txt in enumerate(texts or [])]
        image_chunks = [{"id": f"image_{idx}", "content": img} for idx, img in enumerate(images or [])]

        reorder_result, _ = self.reorder_agent.reorder(
            question,
            text_chunks,
            image_chunks,
            top_k_text_after_rerank=self.config.top_k_text_after_rerank,
            top_k_image_after_rerank=self.config.top_k_image_after_rerank,
        )
        selected_image_ids = reorder_result.get("ranking", {}).get("selected_image_ids", [])
        selected_image_chunks = [chunk for chunk in image_chunks if chunk["id"] in selected_image_ids]

        image_response, _ = self.image_agent.answer(question, selected_image_chunks, reorder_result)
        if image_response == getattr(self.image_agent, "refusal_message", ""):
            return image_response, []

        all_messages = "Question:\n" + question + "\n"
        all_messages += "Image Agent:\n" + image_response + "\n"
        final_ans, final_messages = self.sum(all_messages)
        return final_ans, final_messages
    
class MDAt(MDocAgent):
    def __init__(self, config):
        super().__init__(config)
    
    def predict(self, sample, question, texts, images):
        text_chunks = [{"id": f"text_{idx}", "content": txt} for idx, txt in enumerate(texts or [])]
        image_chunks = [{"id": f"image_{idx}", "content": img} for idx, img in enumerate(images or [])]

        reorder_result, _ = self.reorder_agent.reorder(
            question,
            text_chunks,
            image_chunks,
            top_k_text_after_rerank=self.config.top_k_text_after_rerank,
            top_k_image_after_rerank=self.config.top_k_image_after_rerank,
        )
        selected_text_ids = reorder_result.get("ranking", {}).get("selected_text_ids", [])
        selected_text_chunks = [chunk for chunk in text_chunks if chunk["id"] in selected_text_ids]

        text_response, _ = self.text_agent.answer(question, selected_text_chunks, reorder_result)
        if text_response == getattr(self.text_agent, "refusal_message", ""):
            return text_response, []

        all_messages = "Question:\n" + question + "\n"
        all_messages += "Text Agent:\n" + text_response + "\n"
        final_ans, final_messages = self.sum(all_messages)
        return final_ans, final_messages
    
class MDAs(MDocAgent):
    def __init__(self, config):
        super().__init__(config)
    
    def predict(self, sample, question, texts, images):
        text_chunks = [{"id": f"text_{idx}", "content": txt} for idx, txt in enumerate(texts or [])]
        image_chunks = [{"id": f"image_{idx}", "content": img} for idx, img in enumerate(images or [])]

        reorder_result, _ = self.reorder_agent.reorder(
            question,
            text_chunks,
            image_chunks,
            top_k_text_after_rerank=self.config.top_k_text_after_rerank,
            top_k_image_after_rerank=self.config.top_k_image_after_rerank,
        )

        selected_text_ids = reorder_result.get("ranking", {}).get("selected_text_ids", [])
        selected_image_ids = reorder_result.get("ranking", {}).get("selected_image_ids", [])
        selected_text_chunks = [chunk for chunk in text_chunks if chunk["id"] in selected_text_ids]
        selected_image_chunks = [chunk for chunk in image_chunks if chunk["id"] in selected_image_ids]

        text_response, _ = self.text_agent.answer(question, selected_text_chunks, reorder_result)
        image_response, _ = self.image_agent.answer(question, selected_image_chunks, reorder_result)

        text_refusal = getattr(self.text_agent, "refusal_message", "")
        image_refusal = getattr(self.image_agent, "refusal_message", "")
        if text_response == text_refusal and image_response == image_refusal:
            return "Insufficient information to answer the question.", []

        all_messages = "Question:\n" + question + "\n"
        all_messages += "Text Agent:\n" + text_response + "\n"
        all_messages += "Image Agent:\n" + image_response + "\n"
        final_ans, final_messages = self.sum(all_messages)
        return final_ans, final_messages