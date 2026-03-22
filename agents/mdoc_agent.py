
from tqdm import tqdm
import importlib
import json
import torch
import os
import logging
from typing import Dict, Any, List, Tuple

from agents.multi_agent_system import MultiAgentSystem
from agents.base_agent import Agent
from agents.reflection_agent import EvidencePack
from mydatasets.base_dataset import BaseDataset


logger = logging.getLogger(__name__)


class MDocAgent(MultiAgentSystem):
    def __init__(self, config):
        super().__init__(config)
        # Reflection loop configuration (hard-capped at 3 as requested)
        max_iter = getattr(config, "reflection_max_iterations", 3)
        self.reflection_max_iterations = min(3, int(max_iter)) if max_iter else 3
        # Toggle: whether to run the reflection loop inside the summarizing agent
        # 在配置中设置 config.enable_reflection = False 即可完全关闭反思逻辑
        self.enable_reflection: bool = getattr(config, "enable_reflection", False)
        # Collect per-sample reflection outputs for optional JSON dumping (only used when reflection is enabled)
        self.reflection_records: List[Dict[str, Any]] = []

    @staticmethod
    def _slice_text_from_focus(
        evidence: EvidencePack, text_refs: List[str]
    ) -> List[str]:
        """
        Turn textual references like 'Tc:2' or 'Tq:7' into actual snippets.
        - Tc:i -> i-th non-empty line from evidence.text_critical
        - Tq:j -> j-th entry from evidence.text_segments
        """
        tc_lines: List[str] = []
        if evidence.text_critical:
            for line in evidence.text_critical.splitlines():
                if line.strip():
                    tc_lines.append(line.strip())

        snippets: List[str] = []
        for ref in text_refs or []:
            if not isinstance(ref, str):
                continue
            ref = ref.strip()
            if ref.startswith("Tc:"):
                try:
                    idx = int(ref.split("Tc:")[1])
                    if 0 <= idx < len(tc_lines):
                        snippets.append(tc_lines[idx])
                except Exception:
                    continue
            elif ref.startswith("Tq:"):
                try:
                    idx = int(ref.split("Tq:")[1])
                    if 0 <= idx < len(evidence.text_segments):
                        snippets.append(evidence.text_segments[idx])
                except Exception:
                    continue
        return snippets

    @staticmethod
    def _slice_images_from_focus(
        evidence: EvidencePack, image_refs: List[str]
    ) -> List[str]:
        """
        Turn image references like 'Iq:1(page=3)' into a subset of evidence.image_paths.
        Currently we only care about the Iq:<index> portion and ignore the page metadata.
        """
        selected: List[str] = []
        for ref in image_refs or []:
            if not isinstance(ref, str):
                continue
            # Strip page metadata if present
            core = ref.split("(", 1)[0].strip()
            if core.startswith("Iq:"):
                try:
                    idx = int(core.split("Iq:")[1])
                    if 0 <= idx < len(evidence.image_paths):
                        selected.append(evidence.image_paths[idx])
                except Exception:
                    continue
        return selected

    def _rerun_text_agent(
        self,
        text_agent: Agent,
        question: str,
        bad_claims: List[Dict[str, Any]],
        focus_text_snippets: List[str],
    ) -> Tuple[str, List[Any]]:
        """
        Re-run the Text agent using focused text snippets and bad claims.
        The agent is required to output JSON:
        {
            "answer_t": "...",
            "citations": [...]
        }
        """
        evidence_block = "\n\n".join(focus_text_snippets) if focus_text_snippets else ""
        prompt = (
            question
            + "\n\nYou are revising the TEXT-only answer with focused evidence.\n"
            "Use ONLY the following text evidence snippets and the list of bad claims\n"
            "to correct factual errors and remove unsupported content.\n"
            f"Text evidence snippets:\n{evidence_block}\n\n"
            "Bad claims (JSON):\n"
            f"{json.dumps(bad_claims, ensure_ascii=False)}\n\n"
            "Return ONLY a JSON object of the form:\n"
            '{\"answer_t\": \"...\", \"citations\": [ ... ]}.\n'
        )
        raw, messages = text_agent.predict(
            prompt, texts=focus_text_snippets or None, images=None, with_sys_prompt=True
        )
        answer_t = raw
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = raw[start : end + 1]
                parsed = json.loads(json_str)
                answer_t = parsed.get("answer_t", raw)
        except Exception:
            # Fall back to raw if parsing fails
            pass
        return answer_t, messages

    def _rerun_image_agent(
        self,
        image_agent: Agent,
        question: str,
        bad_claims: List[Dict[str, Any]],
        focus_images: List[str],
    ) -> Tuple[str, List[Any]]:
        """
        Re-run the Image agent using focused image slices and bad claims.
        The agent is required to output JSON:
        {
            "answer_i": "...",
            "citations": [...]
        }
        """
        prompt = (
            question
            + "\n\nYou are revising the IMAGE-grounded answer with focused image evidence.\n"
            "Use ONLY the provided image snippets and the bad claims list to correct\n"
            "factual errors and remove unsupported content.\n"
            "Bad claims (JSON):\n"
            f"{json.dumps(bad_claims, ensure_ascii=False)}\n\n"
            "Return ONLY a JSON object of the form:\n"
            '{\"answer_i\": \"...\", \"citations\": [ ... ]}.\n'
        )
        raw, messages = image_agent.predict(
            prompt, texts=None, images=focus_images or None, with_sys_prompt=True
        )
        answer_i = raw
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = raw[start : end + 1]
                parsed = json.loads(json_str)
                answer_i = parsed.get("answer_i", raw)
        except Exception:
            # Fall back to raw if parsing fails
            pass
        return answer_i, messages

    def predict(self, question, texts, images):
        # General agent (aG)
        general_agent = self.agents[-1]
        general_response, messages = general_agent.predict(
            question, texts, images, with_sys_prompt=True
        )

        # Critical information from the General agent (Tc/Ic)
        critical_info = general_agent.self_reflect(
            prompt=general_agent.config.agent.critical_prompt, add_to_message=False
        )

        start_index = critical_info.find("{")
        end_index = critical_info.find("}") + 1
        critical_info_json = critical_info[start_index:end_index]
        text_reflection = ""
        image_reflection = ""
        try:
            critical_parsed = json.loads(critical_info_json)
            text_reflection = critical_parsed.get("text", "")
            image_reflection = critical_parsed.get("image", "")
        except Exception as e:
            print(e)

        # Build EvidencePack ONCE per predict call
        evidence_pack = EvidencePack(
            question=question,
            text_segments=texts or [],
            image_paths=images or [],
            general_answer=general_response,
            text_critical=text_reflection,
            image_critical=image_reflection,
        )

        text_agent = self.agents[1]
        image_agent = self.agents[0]

        # ===== Initial Text & Image agents run =====
        base_relect_prompt = "\nYou may use the given clue:\n"

        text_prompt_suffix = base_relect_prompt + text_reflection
        text_response, text_messages = text_agent.predict(
            question + text_prompt_suffix,
            texts=texts,
            images=None,
            with_sys_prompt=True,
        )

        image_prompt_suffix = base_relect_prompt + image_reflection
        image_response, image_messages = image_agent.predict(
            question + image_prompt_suffix,
            texts=None,
            images=images,
            with_sys_prompt=True,
        )

        # Aggregated messages for the summarizing agent
        def build_sum_input(
            general_ans: str, text_ans: str, image_ans: str
        ) -> str:
            msg = "General Agent:\n" + general_ans + "\n"
            msg += "Text Agent:\n" + text_ans + "\n"
            msg += "Image Agent:\n" + image_ans + "\n"
            msg += (
                "Instruction:\n"
                "Resolve conflicts strictly based on focus evidence.\n"
                "Remove unsupported claims.\n"
                "Prefer cited evidence when available.\n"
            )
            return msg

        sum_input = build_sum_input(general_response, text_response, image_response)
        final_ans, final_messages = self.sum(sum_input)

        # 如果关闭了反思开关，直接返回总结结果（旧行为：只用 summarizing agent，不跑 reflection loop）
        if not self.enable_reflection:
            return final_ans, final_messages

        # ===== Reflection loop INSIDE summarizing agent =====
        max_iters = max(1, int(self.reflection_max_iterations))
        reflection_history: List[Dict[str, Any]] = []

        for iteration in range(max_iters):
            reflection_result = self.sum_agent.reflect(final_ans, evidence_pack)

            overall_status = reflection_result.get("overall_status", "PASS")
            bad_claims = reflection_result.get("bad_claims", []) or []
            dispatch_plan = reflection_result.get("dispatch_plan", {}) or {}
            focus_pack = reflection_result.get("focus_pack", {}) or {}

            needs_text_rerun = bool(dispatch_plan.get("needs_text_rerun", False))
            needs_image_rerun = bool(dispatch_plan.get("needs_image_rerun", False))

            num_bad = len(bad_claims)

            # Logging
            logger.info(
                "Reflection iteration %d: bad_claims=%d, text_rerun=%s, image_rerun=%s",
                iteration + 1,
                num_bad,
                needs_text_rerun,
                needs_image_rerun,
            )

            # Record this iteration for later inspection
            reflection_history.append(
                {
                    "iteration": iteration + 1,
                    "overall_status": overall_status,
                    "bad_claims": bad_claims,
                    "dispatch_plan": dispatch_plan,
                    "focus_pack": focus_pack,
                    "current_summary": final_ans,
                }
            )

            # 1) 如果所有声明都通过（overall_status == PASS），直接收敛
            # 2) 如果没有 bad_claim 或者不需要任何 rerun，也没必要继续空转
            if (
                overall_status == "PASS"
                or num_bad == 0
                or (not needs_text_rerun and not needs_image_rerun)
            ):
                break

            if iteration + 1 >= max_iters:
                # Reached loop limit, stop with latest answer
                break

            # Build focus-based evidence slices
            focus_text_refs = focus_pack.get("text_refs") or []
            focus_image_refs = focus_pack.get("image_refs") or []

            focus_text_snippets = self._slice_text_from_focus(
                evidence_pack, focus_text_refs
            )
            focus_images = self._slice_images_from_focus(
                evidence_pack, focus_image_refs
            )

            # Optionally re-run Text agent
            if needs_text_rerun:
                new_text_ans, _ = self._rerun_text_agent(
                    text_agent, question, bad_claims, focus_text_snippets
                )
                text_response = new_text_ans

            # Optionally re-run Image agent
            if needs_image_rerun:
                new_image_ans, _ = self._rerun_image_agent(
                    image_agent, question, bad_claims, focus_images
                )
                image_response = new_image_ans

            # Re-run summarizing agent with updated answers and fixed instruction
            sum_input = build_sum_input(general_response, text_response, image_response)
            final_ans, final_messages = self.sum(sum_input)

        # Persist per-sample reflection info for offline inspection
        try:
            self.reflection_records.append(
                {
                    "question": question,
                    "general_answer": general_response,
                    "reflection_history": reflection_history,
                    "final_answer": final_ans,
                }
            )
        except Exception:
            # Do not crash prediction if logging fails
            pass

        return final_ans, final_messages
