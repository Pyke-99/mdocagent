from models.base_model import BaseModel
from mydatasets.base_dataset import BaseDataset
import os
from typing import Dict, Union, Any, List
import json
import pandas as pd
from tqdm import tqdm
import re
import importlib
import logging

from agents.reflection_agent import EvidencePack


logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, config, model=None):
        self.config = config
        self.messages = None
        if model is not None:
            self.model:BaseModel = model
        else:
            module = importlib.import_module(self.config.model.module_name)
            model_class = getattr(module, self.config.model.class_name)
            print("Create model: ", self.config.model.class_name)
            self.model = model_class(self.config.model)
    
    def clean_messages(self):
        self.messages = None
        
    def _predict(self, question, texts=None, images=None, add_to_message = False):
        if not self.config.agent.use_text:
            texts = None
        if not self.config.agent.use_image:
            images = None
        generated_ans, messages = self.model.predict(question, texts, images, self.messages)
        if add_to_message:
            self.messages = messages
        return generated_ans, messages
    
    def predict(self, question, texts=None, images=None, with_sys_prompt=True):
        if with_sys_prompt:
            question = self.config.agent.system_prompt + question
        return self._predict(question, texts, images, add_to_message = True)
    
    def self_reflect(self, prompt=None, add_to_message = True):
        if prompt is None:
            self_reflect_prompt = self.config.agent.self_reflect_prompt
        else:
            self_reflect_prompt = prompt
        
        generated_ans, messages = self._predict(question = self_reflect_prompt)
        if add_to_message:
            self.messages = messages
        
        return generated_ans
    
    def eval(self, question, answer, gt):
        prompt = self.config.agent.eval_system_prompt.format(question=question, answer=answer, gt=gt)
        try:
            generated_ans, _ = self.model.predict(prompt)
            result = extract_evaluation_metrics(generated_ans)
            return result
        except Exception as e:
            print(f"Error evaluating answer: {str(e)}")
            return {"binary_correctness": 0}
    
    def eval_dataset(self, dataset: BaseDataset):
        samples, ans_path = dataset.load_latest_results()
        if self.config.truncate_len:
            samples = samples[:self.config.truncate_len]
        samples_with_answer = []
        for sample in tqdm(samples):
            try:
                question = sample[dataset.config.question_key]
                answer = sample[self.config.ans_key]
                gt = sample[dataset.config.gt_key]
                result = self.eval(question, answer, gt)
                sample['binary_correctness'] = result.get('binary_correctness', None)
                samples_with_answer.append(sample)
            except Exception as e:
                print(f"Error evaluating sample: {str(e)}")
                
        ans_file_path_name = ans_path[:-5]+"_results.json"
        with open(ans_file_path_name, "w") as file:
            json.dump(samples_with_answer, file, indent=4)
            
        samples_with_answer = pd.DataFrame(samples_with_answer)
        path = os.path.join(dataset.config.result_dir,"results.txt")
        with open(path, "a") as file:
            file.write("\nEvaluation Results Summary:\n")
            file.write(f"Result file: {ans_path}\n")
            file.write(f"Average Binary Correctness: {samples_with_answer['binary_correctness'].mean():.3f}\n")
        
        print(f"Save results to {path}.")

    def reflect(self, candidate_answer: str, evidence: EvidencePack) -> Dict[str, Any]:
        """
        Reflection on a candidate summary answer using the SAME model instance.

        The model is instructed to:
        1) Extract atomic factual claims from `candidate_answer`.
        2) Align each claim against the EvidencePack:
           - question (q)
           - retrieved text segments (Tq)
           - retrieved image segments (Iq)
           - critical text/image info (Tc/Ic)
        3) Return a STRICT JSON object with the schema:
           {
             "overall_status": "PASS" | "FAIL",
             "bad_claims": [
               {"claim": "...", "status": "...", "reason": "..."}
             ],
             "dispatch_plan": {
               "needs_text_rerun": true/false,
               "needs_image_rerun": true/false
             },
             "focus_pack": {
               "text_refs": ["Tc:2", "Tq:7"],
               "image_refs": ["Iq:1(page=3)", "Iq:4(page=6)"]
             }
           }

        If parsing fails, we conservatively return a PASS with empty bad_claims.
        """
        lines: List[str] = []
        lines.append(
            "You are a strict factual reflection module for a multi-agent MDocAgent pipeline. "
            "You must ONLY return JSON in the exact schema described below."
        )
        lines.append(
            "Your JSON schema MUST be:\n"
            "{\n"
            '  "overall_status": "PASS" | "FAIL",\n'
            '  "bad_claims": [\n'
            '    {\n'
            '      "claim": "atomic factual claim text",\n'
            '      "status": "SUPPORTED" | "INSUFFICIENT" | "CONFLICT",\n'
            '      "reason": "short natural language explanation of the label"\n'
            "    }\n"
            "  ],\n"
            '  "dispatch_plan": {\n'
            '    "needs_text_rerun": true | false,\n'
            '    "needs_image_rerun": true | false\n'
            "  },\n"
            '  "focus_pack": {\n'
            '    "text_refs": ["Tc:0", "Tc:1", "Tq:3", ...],\n'
            '    "image_refs": ["Iq:0(page=1)", "Iq:2(page=3)", ...]\n'
            "  }\n"
            "}\n"
            "Rules (BE STRICT):\n"
            "- A claim is SUPPORTED only if it is directly and unambiguously entailed by Tc/Tq/Iq; mere plausibility is NOT enough.\n"
            "- If the evidence does not clearly support the claim, you MUST label it as INSUFFICIENT (do NOT guess or assume).\n"
            "- If the evidence contradicts the claim (numbers, categories, comparisons, counts, time ranges, etc.), label it as CONFLICT.\n"
            "- If the user question expects a specific number, comparison, group name, or time span and the candidate answer omits it or only gives vague language, treat that claim as INSUFFICIENT.\n"
            "- If the candidate answer says there is \"not enough information\" or that the image/text does not provide information, but Tc/Tq/Iq actually contain relevant information, then that claim is CONFLICT.\n"
            "- overall_status MUST be \"PASS\" if and only if there are NO bad claims "
            "(i.e., no claim labeled as INSUFFICIENT or CONFLICT).\n"
            "- A bad claim is any claim whose status is INSUFFICIENT or CONFLICT.\n"
            "- If ALL claims are SUPPORTED, overall_status MUST be \"PASS\" and bad_claims SHOULD be an empty list.\n"
            "- focus_pack.text_refs can reference:\n"
            "    * Tc:i  -> the i-th textual slice from critical text info (Tc)\n"
            "    * Tq:j  -> the j-th retrieved text segment (Tq)\n"
            "- focus_pack.image_refs can reference:\n"
            "    * Iq:k(page=p) -> the k-th retrieved image segment (Iq) and optional page metadata.\n"
            "Return STRICTLY one JSON object and nothing else."
        )

        # Describe evidence
        lines.append(f"\nQuestion (q): {evidence.question}")

        if evidence.text_critical:
            lines.append("\nCritical text info (Tc), indexed by line:")
            for idx, seg in enumerate(evidence.text_critical.splitlines()):
                seg_str = seg.strip()
                if not seg_str:
                    continue
                lines.append(f"[Tc:{idx}] {seg_str}")

        if evidence.image_critical:
            lines.append("\nCritical image info (Ic):")
            # 防止非字符串类型导致 join 报错
            lines.append(str(evidence.image_critical))

        lines.append("\nRetrieved text segments (Tq):")
        if evidence.text_segments:
            for idx, seg in enumerate(evidence.text_segments):
                lines.append(f"[Tq:{idx}] {seg}")
        else:
            lines.append("No text segments available.")

        lines.append("\nRetrieved image segments (Iq):")
        if evidence.image_paths:
            for idx, img in enumerate(evidence.image_paths):
                lines.append(
                    f"[Iq:{idx}] Image path or identifier: {img}. Page information may be provided separately."
                )
        else:
            lines.append("No image segments available.")

        lines.append("\nCandidate summary answer (aS0) to be fact-checked:")
        # 同样保证为字符串
        lines.append(str(candidate_answer))

        lines.append(
            "\nYour tasks:\n"
            "1) Extract atomic factual claims from aS0.\n"
            "2) For each claim, compare against the EvidencePack (q, Tq, Iq, Tc/Ic).\n"
            "3) Label each claim as SUPPORTED, INSUFFICIENT, or CONFLICT and explain briefly in 'reason'.\n"
            "4) Decide whether text and/or image agents should be re-run and set dispatch_plan accordingly.\n"
            "5) Select the most relevant evidence references for focus_pack.text_refs and focus_pack.image_refs.\n"
            "Respond with STRICT JSON only."
        )

        prompt = "\n".join(lines)

        try:
            raw_output, _ = self.model.predict(
                prompt,
                texts=evidence.text_segments,
                images=evidence.image_paths,
                history=None,
            )
        except Exception as e:
            logger.warning("Reflection call failed, defaulting to PASS: %s", e)
            return {
                "overall_status": "PASS",
                "bad_claims": [],
                "dispatch_plan": {
                    "needs_text_rerun": False,
                    "needs_image_rerun": False,
                },
                "focus_pack": {"text_refs": [], "image_refs": []},
            }

        json_str = raw_output
        try:
            start = raw_output.find("{")
            end = raw_output.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = raw_output[start : end + 1]
            parsed: Dict[str, Any] = json.loads(json_str)
        except Exception as e:
            logger.warning(
                "Failed to parse reflection JSON, defaulting to PASS. Error: %s, raw: %r",
                e,
                raw_output[:500],
            )
            return {
                "overall_status": "PASS",
                "bad_claims": [],
                "dispatch_plan": {
                    "needs_text_rerun": False,
                    "needs_image_rerun": False,
                },
                "focus_pack": {"text_refs": [], "image_refs": []},
            }

        # Normalise fields to required structure
        overall_status = str(parsed.get("overall_status", "")).upper()

        bad_claims = parsed.get("bad_claims") or []
        if not isinstance(bad_claims, list):
            bad_claims = []

        normalized_bad_claims: List[Dict[str, Any]] = []
        for c in bad_claims:
            if not isinstance(c, dict):
                continue
            claim_text = c.get("claim", "") or c.get("text", "")
            status_raw = str(c.get("status", "")).upper()
            if status_raw not in {"SUPPORTED", "INSUFFICIENT", "CONFLICT"}:
                status_raw = "INSUFFICIENT"
            reason = c.get("reason", "")
            # 只有 INSUFFICIENT / CONFLICT 才算真正的 bad claim
            if status_raw in {"INSUFFICIENT", "CONFLICT"}:
                normalized_bad_claims.append(
                    {
                        "claim": claim_text,
                        "status": status_raw,
                        "reason": reason,
                    }
                )

        dispatch_plan = parsed.get("dispatch_plan") or {}
        needs_text_rerun = bool(dispatch_plan.get("needs_text_rerun", False))
        needs_image_rerun = bool(dispatch_plan.get("needs_image_rerun", False))

        focus_pack = parsed.get("focus_pack") or {}
        text_refs = focus_pack.get("text_refs") or []
        image_refs = focus_pack.get("image_refs") or []
        if not isinstance(text_refs, list):
            text_refs = []
        if not isinstance(image_refs, list):
            image_refs = []

        # 根据是否存在 bad claims 决定是否需要强制修正 overall_status
        has_issue = len(normalized_bad_claims) > 0
        if overall_status not in {"PASS", "FAIL"}:
            # 模型没按规范给出 overall_status，就由我们来根据 bad_claims 决定
            overall_status = "FAIL" if has_issue else "PASS"
        else:
            # 如果模型说 PASS 但我们检测到问题，则强制改为 FAIL；
            # 如果模型说 FAIL 即使没有 bad_claims，也保留 FAIL，不再强制改成 PASS
            if overall_status == "PASS" and has_issue:
                overall_status = "FAIL"

        return {
            "overall_status": overall_status,
            "bad_claims": normalized_bad_claims,
            "dispatch_plan": {
                "needs_text_rerun": needs_text_rerun,
                "needs_image_rerun": needs_image_rerun,
            },
            "focus_pack": {
                "text_refs": text_refs,
                "image_refs": image_refs,
            },
        }

def extract_evaluation_metrics(eval_str: str) -> Dict[str, Union[float, int]]:
    import re
    # 1. 尝试直接提取 JSON
    try:
        start_index = eval_str.find('{') 
        end_index = eval_str.rfind('}') + 1 
        if start_index != -1 and end_index > start_index:
            json_str = eval_str[start_index:end_index]
            metrics = json.loads(json_str)
            if 'binary_correctness' in metrics:
                return {'binary_correctness': int(metrics.get('binary_correctness', 0))}
    except Exception:
        pass
    # 2. 尝试正则提取 JSON
    try:
        match = re.search(r'\{[^\}]*?"binary_correctness"\s*:\s*[01][^\}]*?\}', eval_str)
        if match:
            metrics = json.loads(match.group(0))
            return {'binary_correctness': int(metrics.get('binary_correctness', 0))}
    except Exception:
        pass
    # 3. 关键词判断
    s = eval_str.lower()
    if any(x in s for x in ["yes", "correct", "对", "正确", "是", "right"]):
        return {'binary_correctness': 1}
    if any(x in s for x in ["no", "wrong", "错误", "错", "不对", "不正确", "false"]):
        return {'binary_correctness': 0}
    # 4. 默认
    return {'binary_correctness': 0}