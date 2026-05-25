from dataclasses import asdict

from agents.cross_modal_alignment_agent import CrossModalAlignmentAgent
from agents.evidence_selection_agent import EvidenceSelectionAgent
from agents.final_answer_agent import FinalAnswerAgent
from agents.question_analysis_agent import QuestionAnalysisAgent


class AgentRunner:
    def __init__(self, stage_models, agent4_input_mode: str = "rewritten_only"):
        self.agent1 = QuestionAnalysisAgent(stage_models["agent1"]) if "agent1" in stage_models else None
        self.agent2 = EvidenceSelectionAgent(stage_models["agent2"])
        self.agent3 = CrossModalAlignmentAgent(stage_models["agent3"])
        self.agent4 = FinalAnswerAgent(stage_models["agent4"])
        self.agent4_input_mode = str(agent4_input_mode or "rewritten_only")

    def _merge_token_usage(self, *usages):
        if not usages:
            return None
        merged = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        found = False
        for usage in usages:
            if not isinstance(usage, dict):
                continue
            for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                if key in usage and isinstance(usage[key], int):
                    merged[key] += usage[key]
                    found = True
        if not found:
            return None
        if merged["total_tokens"] == 0:
            merged["total_tokens"] = merged["prompt_tokens"] + merged["completion_tokens"]
        return merged

    def run(self, question, texts, images, agent1_override=None):
        text_chunks = [{"id": f"text_{idx}", "content": txt} for idx, txt in enumerate(texts or [])]
        image_chunks = [{"id": f"image_{idx}", "content": img} for idx, img in enumerate(images or [])]
        text_map = {item["id"]: item["content"] for item in text_chunks}
        image_map = {item["id"]: item["content"] for item in image_chunks}

        if agent1_override is not None:
            # Use externally injected agent1 output (e.g., from RouteAgent2)
            out1_dict = agent1_override
            token1 = None
        elif self.agent1 is not None:
            out1, token1 = self.agent1.run(question)
            out1_dict = asdict(out1)
        else:
            # Skip agent1; use minimal default output
            out1_dict = {
                "question_type": "unknown",
                "task_operation": "answer",
                "answer_target": "general",
                "key_constraints": [],
                "must_check_slots": [],
                "time_constraint": "",
                "scope_constraint": "",
                "modality_hint": "text_and_image",
            }
            token1 = None

        out2, token2 = self.agent2.run(question, out1_dict, text_chunks, image_chunks)
        out2_dict = asdict(out2)

        out3, token3 = self.agent3.run(question, out1_dict, out2_dict, text_map, image_map)
        out3_dict = asdict(out3)

        out4, token4 = self.agent4.run(
            question,
            out1_dict,
            out3_dict,
            text_chunks_by_id=text_map,
            image_chunks_by_id=image_map,
            input_mode=self.agent4_input_mode,
        )
        out4_dict = asdict(out4)

        trace = {
            "agent1": out1_dict,
            "agent2": out2_dict,
            "agent3": out3_dict,
            "agent4": out4_dict,
        }
        return out4_dict, trace, self._merge_token_usage(token1, token2, token3, token4)
