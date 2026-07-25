import json
import os
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from agents.final_answer_agent import FinalAnswerAgent
from agents.route_agent2 import RouteAgent2
from agents.route_agent3 import RouteAgent3
from agents.route_gate_agent import RouteGateAgent
from mydatasets.base_dataset import BaseDataset
from pipeline.model_registry import ModelRegistry


class RouteHybridPipeline:
    KEEP_FIELDS = [
        "doc_id",
        "doc_type",
        "question",
        "answer",
        "evidence_pages",
        "evidence_sources",
        "answer_format",
        "retrieval-query",
        "retrieval-key",
        "qwen_retrieval-key",
        "qwen_retrieval-query",
        "text-index-path-question",
        "text-top-10-question",
        "text-top-10-question_score",
        "image-top-10-question",
        "image-top-10-question_score",
    ]

    def __init__(self, config, gate_model_cfg, route_model_cfg, final_model_cfg):
        self.config = config
        self.registry = ModelRegistry()

        gate_key = self.registry.build_model_key(gate_model_cfg)
        route_key = self.registry.build_model_key(route_model_cfg)
        final_key = self.registry.build_model_key(final_model_cfg)

        self.gate_model = self.registry.get_or_create(gate_key, gate_model_cfg)
        self.route_model = self.registry.get_or_create(route_key, route_model_cfg)
        self.final_model = self.registry.get_or_create(final_key, final_model_cfg)

        self.gate_agent = RouteGateAgent(self.gate_model)
        self.agent2 = RouteAgent2(self.route_model)
        self.agent3 = RouteAgent3(self.route_model)
        self.agent4 = FinalAnswerAgent(self.final_model)

    def _has_valid_answer(self, sample: Dict) -> bool:
        if self.config.ans_key not in sample:
            return False
        value = sample.get(self.config.ans_key)
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
        return True

    def _normalize_token_usage_entry(self, usage):
        if not isinstance(usage, dict):
            return None

        if any(k in usage for k in ["prompt_tokens", "completion_tokens", "total_tokens"]):
            return {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            }

        if any(k in usage for k in ["input", "output", "total"]):
            input_tokens = int(usage.get("input", 0) or 0)
            output_tokens = int(usage.get("output", 0) or 0)
            total_tokens = int(usage.get("total", input_tokens + output_tokens) or 0)
            return {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": total_tokens,
            }

        return None

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
            normalized = self._normalize_token_usage_entry(usage)
            if not normalized:
                continue
            for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                merged[key] += normalized[key]
            found = True
        if not found:
            return None
        if merged["total_tokens"] == 0:
            merged["total_tokens"] = merged["prompt_tokens"] + merged["completion_tokens"]
        return merged

    def _build_output_views(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        keep_keys = set(self.KEEP_FIELDS + [self.config.ans_key])
        result_samples = []
        analysis_samples = []
        token_usage_samples = []

        for sample in samples:
            result_item = {k: sample.get(k) for k in self.KEEP_FIELDS if k in sample}
            result_item[self.config.ans_key] = sample.get(self.config.ans_key)
            result_samples.append(result_item)

            analysis_samples.append(
                {
                    "doc_id": sample.get("doc_id"),
                    "question": sample.get("question"),
                    "analysis_fields": {k: v for k, v in sample.items() if k not in keep_keys},
                }
            )

            token_usage_samples.append(
                {
                    "doc_id": sample.get("doc_id"),
                    "question": sample.get("question"),
                    "token_usage": sample.get("token_usage"),
                }
            )

        return result_samples, analysis_samples, token_usage_samples

    def _dump_all_outputs(self, dataset: BaseDataset, samples: List[Dict], sample_no=None):
        result_samples, _, token_usage_samples = self._build_output_views(samples)
        result_path = dataset.dump_reults(result_samples)

        base, _ = os.path.splitext(result_path)
        token_usage_path = base + "_token_usage.json"
        with open(token_usage_path, "w") as f:
            json.dump(token_usage_samples, f, indent=4)

        if sample_no is None:
            print(f"Save final results to {result_path}.")
            print(f"Save token usage to {token_usage_path}.")
        else:
            print(f"Save {sample_no} results to {result_path}.")
            print(f"Save {sample_no} token usage to {token_usage_path}.")

    def _build_text_image_chunks(self, texts, images):
        text_chunks = [{"id": f"text_{idx}", "content": txt} for idx, txt in enumerate(texts or [])]
        image_chunks = [{"id": f"image_{idx}", "content": img} for idx, img in enumerate(images or [])]
        return text_chunks, image_chunks

    def _route_toggle(self, name: str, default: bool = True) -> bool:
        route_cfg = getattr(self.config, "route_hybrid", None)
        if route_cfg is None:
            return default
        return bool(getattr(route_cfg, f"{name}_enabled", default))

    def _force_multi_agent_path(self) -> bool:
        route_cfg = getattr(self.config, "route_hybrid", None)
        if route_cfg is None:
            return False
        return bool(getattr(route_cfg, "force_multi_agent_path", False))

    def _build_passthrough_agent2_output(self, gate_dict: dict, text_chunks: List[Dict], image_chunks: List[Dict]) -> Dict:
        candidate_evidence = []
        for chunk in text_chunks:
            candidate_evidence.append(
                {
                    "text": str(chunk.get("content", ""))[:300],
                    "type": "direct",
                }
            )
        for chunk in image_chunks:
            candidate_evidence.append(
                {
                    "text": "[image evidence] " + str(chunk.get("id", "")),
                    "type": "background",
                }
            )

        return {
            "question_type": "simple" if gate_dict.get("route", "complex") == "simple" else "multi-hop",
            "entities": [],
            "target": "",
            "constraints": gate_dict.get("key_signals", []),
            "requires_multi_hop": gate_dict.get("route", "complex") == "complex",
            "requires_table": any("table" in str(s).lower() for s in gate_dict.get("key_signals", [])),
            "requires_image": bool(image_chunks),
            "candidate_evidence": candidate_evidence,
            "reason": "agent2_ablation_passthrough",
        }

    def _build_passthrough_agent3_output(self, agent2_output: dict, text_chunks: List[Dict], image_chunks: List[Dict]) -> Dict:
        selected_text_chunks = [str(chunk.get("id", "")).strip() for chunk in text_chunks if str(chunk.get("id", "")).strip()]
        selected_image_chunks = [str(chunk.get("id", "")).strip() for chunk in image_chunks if str(chunk.get("id", "")).strip()]
        has_both_modalities = bool(selected_text_chunks and selected_image_chunks)
        response_mode = "multimodal-joint" if has_both_modalities else "text-first" if selected_text_chunks else "image-first"

        aligned_evidence = []
        pair_count = min(len(selected_text_chunks), len(selected_image_chunks))
        for idx in range(pair_count):
            aligned_evidence.append(
                {
                    "text_part": selected_text_chunks[idx],
                    "image_part": selected_image_chunks[idx],
                    "relation": "support" if idx == 0 else "compare",
                }
            )

        return {
            "aligned_evidence": aligned_evidence,
            "evidence_groups": [
                {
                    "group": "retrieval_passthrough",
                    "supports": "direct evidence for final answering",
                }
            ],
            "conflicts": [],
            "missing_info": [],
            "ready_for_answer": bool(selected_text_chunks or selected_image_chunks),
            "final_text_chunks": selected_text_chunks,
            "final_image_chunks": selected_image_chunks,
            "cross_modal_relations": aligned_evidence,
            "alignment_points": [
                "agent3 disabled: deterministic alignment passthrough",
                f"final_text_chunks={selected_text_chunks}",
                f"final_image_chunks={selected_image_chunks}",
            ],
            "alignment_instructions": [
                "answer only from selected chunks",
                "state uncertainty when evidence is partial",
            ],
            "response_mode": response_mode,
        }

    def _build_minimal_agent1(self, question: str, gate_out: Dict, text_chunks: List[Dict], image_chunks: List[Dict]) -> Dict:
        if image_chunks and text_chunks:
            modality_hint = "multimodal-joint"
        elif image_chunks:
            modality_hint = "image-first"
        else:
            modality_hint = "text-first"

        return {
            "task_type": gate_out.get("route", "simple"),
            "question_type": gate_out.get("route", "simple"),
            "task_operation": "answer",
            "answer_target": "direct_answer",
            "hard_constraints": gate_out.get("key_signals", []),
            "workflow_guidance": "simple route: answer directly from selected evidence",
            "target_variable": "final answer",
            "answer_type": "short_text",
            "key_constraints": gate_out.get("key_signals", []),
            "time_constraint": "",
            "scope_constraint": "",
            "comparison_axes": [],
            "calc_requirements": [],
            "modality_hint": modality_hint,
            "risk_flags": [],
            "must_check_slots": [],
            "reasoning_focus": [],
            "forbidden_shortcuts": [],
            "answer_style": "concise",
        }

    def _build_simple_agent3(self, text_chunks: List[Dict], image_chunks: List[Dict], gate_out: Dict) -> Dict:
        final_text_chunks = [chunk["id"] for chunk in text_chunks]
        final_image_chunks = [chunk["id"] for chunk in image_chunks]
        if not final_text_chunks and not final_image_chunks:
            final_text_chunks = [chunk["id"] for chunk in text_chunks[:1]]

        return {
            "aligned_evidence": [],
            "evidence_groups": [],
            "conflicts": [],
            "missing_info": [],
            "ready_for_answer": bool(final_text_chunks or final_image_chunks),
            "final_text_chunks": final_text_chunks,
            "final_image_chunks": final_image_chunks,
            "cross_modal_relations": [],
            "rewritten_evidence": [],
            "alignment_points": [f"simple route from gate={gate_out.get('route', 'simple')}"] if gate_out else [],
            "alignment_instructions": ["answer directly from the selected evidence and keep it concise"],
            "response_mode": "multimodal-joint" if final_text_chunks and final_image_chunks else "text-first" if final_text_chunks else "image-first",
        }

    def predict_dataset(self, dataset: BaseDataset, resume_path=None):
        samples = dataset.load_data(use_retreival=True)
        if resume_path:
            assert os.path.exists(resume_path)
            with open(resume_path, "r") as f:
                samples = json.load(f)

        if self.config.truncate_len:
            samples = samples[: self.config.truncate_len]

        agent2_enabled = self._route_toggle("agent2", True)
        agent3_enabled = self._route_toggle("agent3", True)
        force_multi_agent_path = self._force_multi_agent_path()

        sample_no = 0
        for sample in tqdm(samples):
            if resume_path and self._has_valid_answer(sample):
                continue

            question, texts, images = dataset.load_sample_retrieval_data(sample)
            text_chunks, image_chunks = self._build_text_image_chunks(texts, images)

            gate_out, gate_token_usage = self.gate_agent.run(question, texts, images)
            gate_dict = self.gate_agent.to_dict(gate_out)

            if force_multi_agent_path:
                gate_dict = dict(gate_dict)
                gate_dict["route"] = "complex"
                gate_dict["reason"] = str(gate_dict.get("reason", ""))
                if gate_dict["reason"]:
                    gate_dict["reason"] += " | force_multi_agent_path=true"
                else:
                    gate_dict["reason"] = "force_multi_agent_path=true"

            if gate_dict.get("route", "complex") == "simple":
                agent1_like = self._build_minimal_agent1(question, gate_dict, text_chunks, image_chunks)
                agent3_like = self._build_simple_agent3(text_chunks, image_chunks, gate_dict)
                final_out = self.agent4.run(
                    question,
                    agent1_like,
                    agent3_like,
                    agent2_output={},
                    text_chunks_by_id={c["id"]: c["content"] for c in text_chunks},
                    image_chunks_by_id={c["id"]: c["content"] for c in image_chunks},
                    input_mode=getattr(self.config, "agent4_input_mode", "rewritten_only"),
                )
                final_answer = final_out.final_answer
                token_usage = self._merge_token_usage(gate_token_usage, final_out.token_usage)
                trace = {
                    "mode": "route_hybrid",
                    "route": "simple",
                    "route_gate": gate_dict,
                    "route_ablation": {
                        "agent2_enabled": agent2_enabled,
                        "agent3_enabled": agent3_enabled,
                        "force_multi_agent_path": force_multi_agent_path,
                    },
                    "agent1": agent1_like,
                    "agent3": agent3_like,
                    "agent4": self.agent4.to_dict(final_out),
                }
            else:
                agent1_like = self._build_minimal_agent1(question, gate_dict, text_chunks, image_chunks)
                if agent2_enabled:
                    out2, out2_token_usage = self.agent2.run(question, texts, images)
                else:
                    out2 = self._build_passthrough_agent2_output(gate_dict, text_chunks, image_chunks)
                    out2_token_usage = None
                out2_dict = out2 if isinstance(out2, dict) else self.agent2.to_dict(out2)
                if agent3_enabled:
                    out3, out3_token_usage = self.agent3.run(question, out2_dict, texts, images)
                else:
                    out3 = self._build_passthrough_agent3_output(out2_dict, text_chunks, image_chunks)
                    out3_token_usage = None
                out3_dict = out3 if isinstance(out3, dict) else self.agent3.to_dict(out3)
                final_out = self.agent4.run(
                    question,
                    agent1_like,
                    out3_dict,
                    agent2_output=out2_dict,
                    text_chunks_by_id={c["id"]: c["content"] for c in text_chunks},
                    image_chunks_by_id={c["id"]: c["content"] for c in image_chunks},
                    input_mode=getattr(self.config, "agent4_input_mode", "rewritten_only"),
                )
                final_answer = final_out.final_answer
                token_usage = self._merge_token_usage(gate_token_usage, out2_token_usage, out3_token_usage, final_out.token_usage)
                trace = {
                    "mode": "route_hybrid",
                    "route": "complex",
                    "route_gate": gate_dict,
                    "route_ablation": {
                        "agent2_enabled": agent2_enabled,
                        "agent3_enabled": agent3_enabled,
                        "force_multi_agent_path": force_multi_agent_path,
                    },
                    "agent1": agent1_like,
                    "agent2": out2_dict,
                    "agent3": out3_dict,
                    "agent4": self.agent4.to_dict(final_out),
                }

            sample[self.config.ans_key] = final_answer
            sample["token_usage"] = token_usage
            if getattr(self.config, "save_trace", True):
                sample[self.config.ans_key + "_trace"] = trace

            torch.cuda.empty_cache()
            sample_no += 1
            if sample_no % self.config.save_freq == 0:
                self._dump_all_outputs(dataset, samples, sample_no=sample_no)

        self._dump_all_outputs(dataset, samples)
