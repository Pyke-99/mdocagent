import json
import os
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from mydatasets.base_dataset import BaseDataset
from pipeline.agent_runner import AgentRunner
from pipeline.model_registry import ModelRegistry


class MainPipeline:
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

    def __init__(self, config, stage_model_cfgs: Dict[str, object]):
        self.config = config
        self.registry = ModelRegistry()
        self.stage_models = {}
        for stage_name, model_cfg in stage_model_cfgs.items():
            # Use registry key based on model config so stages with identical configs share one instance.
            model_key = self.registry.build_model_key(model_cfg)
            self.stage_models[stage_name] = self.registry.get_or_create(model_key, model_cfg)
        self.runner = AgentRunner(
            self.stage_models,
            agent4_input_mode=getattr(config, "agent4_input_mode", "rewritten_only"),
        )

    def predict(self, question, texts, images, agent1_override=None):
        out4, trace, token_usage = self.runner.run(question, texts, images, agent1_override=agent1_override)
        return out4.get("final_answer"), trace, token_usage

    def _has_valid_answer(self, sample: Dict) -> bool:
        if self.config.ans_key not in sample:
            return False
        value = sample.get(self.config.ans_key)
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
        return True

    def _predict_with_runtime_fallbacks(self, question, texts, images, agent1_override=None):
        attempts = [
            (texts, images, "full"),
            ((texts or [])[:4], (images or [])[:2], "reduced"),
            ((texts or [])[:2], [], "text_only"),
            ([], (images or [])[:1], "image_only"),
        ]

        last_error = None
        for cur_texts, cur_images, tag in attempts:
            try:
                answer, trace = self.predict(question, cur_texts, cur_images, agent1_override=agent1_override)
                if answer is None:
                    answer = "Insufficient information to answer the question."
                if not isinstance(trace, dict):
                    trace = {}
                if tag != "full":
                    trace["runtime_fallback"] = tag
                return answer, trace
            except RuntimeError as e:
                last_error = e
                print(f"RuntimeError during {tag} attempt: {e}")
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                continue

        fallback_trace = {
            "runtime_error": str(last_error) if last_error else "unknown runtime error",
            "runtime_fallback": "failed_all_attempts",
        }
        return "Insufficient information to answer the question.", fallback_trace

    def _slot_names(self, slots) -> List[str]:
        out = []
        for s in slots or []:
            if isinstance(s, dict):
                name = str(s.get("slot_name", s.get("name", ""))).strip()
                if name:
                    out.append(name)
            else:
                text = str(s).strip()
                if text:
                    out.append(text)
        return out

    def _build_handoff_entry(self, sample: Dict) -> Dict:
        trace_key = self.config.ans_key + "_trace"
        trace = sample.get(trace_key, {}) if isinstance(sample.get(trace_key, {}), dict) else {}

        a1 = trace.get("agent1", {}) if isinstance(trace.get("agent1", {}), dict) else {}
        a2 = trace.get("agent2", {}) if isinstance(trace.get("agent2", {}), dict) else {}
        a3 = trace.get("agent3", {}) if isinstance(trace.get("agent3", {}), dict) else {}

        a1_slots = self._slot_names(a1.get("must_check_slots", []))
        key_constraints = a1.get("key_constraints", []) if isinstance(a1.get("key_constraints", []), list) else []
        time_constraint = str(a1.get("time_constraint", "") or "").strip()
        scope_constraint = str(a1.get("scope_constraint", "") or "").strip()
        answer_type = str(a1.get("answer_type", "") or "").strip()
        modality_hint = str(a1.get("modality_hint", "") or "").strip()

        extras = []
        if key_constraints:
            extras.append(f"key_constraints={key_constraints}")
        if time_constraint:
            extras.append(f"time_constraint={time_constraint}")
        if scope_constraint:
            extras.append(f"scope_constraint={scope_constraint}")
        if answer_type:
            extras.append(f"answer_type={answer_type}")
        if modality_hint:
            extras.append(f"modality_hint={modality_hint}")

        a1_text = (
            f"Task type={a1.get('question_type', '')}; operation={a1.get('task_operation', '')}; "
            f"target={a1.get('answer_target', '')}; soft slots={a1_slots}."
        )
        if extras:
            a1_text += " " + "; ".join(extras) + "."

        decisions = a2.get("chunk_decisions", []) if isinstance(a2.get("chunk_decisions", []), list) else []
        use_text = [d.get("chunk_id") for d in decisions if isinstance(d, dict) and d.get("decision") == "use" and d.get("modality") == "text"]
        use_image = [d.get("chunk_id") for d in decisions if isinstance(d, dict) and d.get("decision") == "use" and d.get("modality") == "image"]
        a2_text = (
            f"Selected text chunks={use_text}; selected image chunks={use_image}; "
            f"selection summary={a2.get('selection_summary', {})}."
        )

        a3_text = (
            f"Final text chunks={a3.get('final_text_chunks', [])}; final image chunks={a3.get('final_image_chunks', [])}; "
            f"cross-modal relations={a3.get('cross_modal_relations', [])}; "
            f"alignment points={a3.get('alignment_points', [])}; response mode={a3.get('response_mode', '')}."
        )

        return {
            "doc_id": sample.get("doc_id"),
            "question": sample.get("question"),
            "agent1_to_agent2_nl": a1_text,
            "agent2_to_agent3_nl": a2_text,
            "agent3_to_agent4_nl": a3_text,
        }

    def _build_output_views(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        keep_keys = set(self.KEEP_FIELDS + [self.config.ans_key])

        result_samples = []
        analysis_samples = []
        handoff_samples = []
        token_usage_samples = []

        for sample in samples:
            result_item = {k: sample.get(k) for k in self.KEEP_FIELDS if k in sample}
            result_item[self.config.ans_key] = sample.get(self.config.ans_key)
            result_samples.append(result_item)

            analysis_item = {
                "doc_id": sample.get("doc_id"),
                "question": sample.get("question"),
                "analysis_fields": {k: v for k, v in sample.items() if k not in keep_keys},
            }
            analysis_samples.append(analysis_item)

            handoff_samples.append(self._build_handoff_entry(sample))

            token_usage_samples.append(
                {
                    "doc_id": sample.get("doc_id"),
                    "question": sample.get("question"),
                    "token_usage": sample.get("token_usage"),
                }
            )

        return result_samples, analysis_samples, handoff_samples, token_usage_samples

    def _dump_all_outputs(self, dataset: BaseDataset, samples: List[Dict], sample_no=None):
        result_samples, analysis_samples, handoff_samples, token_usage_samples = self._build_output_views(samples)
        result_path = dataset.dump_reults(result_samples)

        base, _ = os.path.splitext(result_path)
        analysis_path = base + "_analysis.json"
        handoff_path = base + "_agent_handoff_nl.json"
        token_usage_path = base + "_token_usage.json"

        with open(analysis_path, "w") as f:
            json.dump(analysis_samples, f, indent=4)
        with open(handoff_path, "w") as f:
            json.dump(handoff_samples, f, indent=4)
        with open(token_usage_path, "w") as f:
            json.dump(token_usage_samples, f, indent=4)

        if sample_no is None:
            print(f"Save final results to {result_path}.")
        else:
            print(f"Save {sample_no} results to {result_path}.")
        print(f"Save analysis to {analysis_path}.")
        print(f"Save agent handoff NL to {handoff_path}.")
        print(f"Save token usage to {token_usage_path}.")

    def predict_dataset(self, dataset: BaseDataset, resume_path=None):
        samples = dataset.load_data(use_retreival=True)
        if resume_path:
            assert os.path.exists(resume_path)
            with open(resume_path, "r") as f:
                samples = json.load(f)

        if self.config.truncate_len:
            samples = samples[: self.config.truncate_len]

        sample_no = 0
        for sample in tqdm(samples):
            if resume_path and self._has_valid_answer(sample):
                continue

            question, texts, images = dataset.load_sample_retrieval_data(sample)
            final_answer, trace = self._predict_with_runtime_fallbacks(question, texts, images)

            sample[self.config.ans_key] = final_answer
            if getattr(self.config, "save_trace", True):
                sample[self.config.ans_key + "_trace"] = trace

            torch.cuda.empty_cache()
            sample_no += 1
            if sample_no % self.config.save_freq == 0:
                self._dump_all_outputs(dataset, samples, sample_no=sample_no)

        self._dump_all_outputs(dataset, samples)
