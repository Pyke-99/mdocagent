import json
import os
import re

import torch
from tqdm import tqdm

from mydatasets.base_dataset import BaseDataset
from pipeline.model_registry import ModelRegistry


class SingleAgentPipeline:
    EMPTY_ANSWER_FALLBACK = "Model returned an empty answer."
    RUNTIME_FAILED_FALLBACK = "All runtime fallback attempts failed; unable to produce an answer."

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

    def __init__(self, config, model_cfg):
        self.config = config
        self.registry = ModelRegistry()
        model_key = f"single_agent:{model_cfg.class_name}:{getattr(model_cfg, 'model_id', getattr(model_cfg, 'model', ''))}"
        self.model = self.registry.get_or_create(model_key, model_cfg)

    def _has_valid_answer(self, sample):
        if self.config.ans_key not in sample:
            return False
        value = sample.get(self.config.ans_key)
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
        return True

    def _predict_single(self, question, texts, images):
        prompt = (
            "Answer the question using only the provided text and image evidence. "
            "Return a complete concise answer in one sentence. "
            "Do not explain. "
            "Do not copy snippets or table headers. "
            "If the question asks who, return the correct person or entity. "
            "If it asks when or during which years, return the exact date or year range. "
            "If it asks how many, return all requested counts. "
            "Avoid fragmentary output and avoid trailing citation-like noise.\n\n"
            "Question:\n"
            + str(question)
        )
        answer, _, token_usage = self.model.predict(prompt, texts=texts or None, images=images or None)
        return str(answer).strip() if answer is not None else "", token_usage

    def _needs_one_based_counting(self, question, answer):
        q = str(question or "").lower()
        a = str(answer or "")
        if not a.strip():
            return False
        if not re.search(r"\b\d+\b", a):
            return False

        key_phrases = [
            "what number",
            "which number",
            "lineage number",
            "number king",
            "number was he",
            "ordinal",
            "rank",
        ]
        return any(k in q for k in key_phrases)

    def _normalize_one_based_counting(self, question, answer, texts, images):
        answer = str(answer or "").strip()
        if not self._needs_one_based_counting(question, answer):
            return answer, {"applied": False, "reason": "not_applicable"}

        prompt = (
            "You are validating a final QA answer. "
            "Rule: all counting, ordinal, and rank outputs must be 1-based (first=1). "
            "If the draft uses zero-based indexing, convert it to the correct 1-based value. "
            "Keep all non-count facts unchanged. "
            "Return only the final corrected answer sentence.\n\n"
            "Question:\n"
            + str(question)
            + "\n\nDraft answer:\n"
            + answer
        )

        try:
            revised, _, _ = self.model.predict(prompt, texts=texts or None, images=images or None)
            revised = str(revised or "").strip()
            if revised and revised != answer and not self._is_placeholder_or_refusal(revised):
                return revised, {
                    "applied": True,
                    "reason": "normalized_to_one_based_counting",
                    "before": answer,
                    "after": revised,
                }
        except Exception as e:
            return answer, {"applied": False, "reason": f"normalize_error:{e}"}

        return answer, {"applied": False, "reason": "no_change"}

    def _is_placeholder_or_refusal(self, text):
        s = str(text or "").strip().lower()
        if not s:
            return True
        markers = [
            "insufficient information",
            "insufficient evidence",
            "cannot answer",
            "can't answer",
            "unable to answer",
            "unable to determine",
            "not enough information",
            "no relevant evidence",
            "no supportable content",
            "provided evidence is insufficient",
            "all runtime fallback attempts failed",
            "model returned an empty answer",
            "no valid answer generated",
            "evidence is insufficient",
            "信息不足",
            "证据不足",
            "无法回答",
        ]
        return any(marker in s for marker in markers)

    def _apply_chunk_limits(self, texts, images):
        text_limit = getattr(self.config, "single_agent_text_chunks", None)
        image_limit = getattr(self.config, "single_agent_image_chunks", None)

        try:
            text_limit = None if text_limit is None else int(text_limit)
        except Exception:
            text_limit = None
        try:
            image_limit = None if image_limit is None else int(image_limit)
        except Exception:
            image_limit = None

        capped_texts = list(texts or [])
        capped_images = list(images or [])
        if text_limit is not None and text_limit >= 0:
            capped_texts = capped_texts[:text_limit]
        if image_limit is not None and image_limit >= 0:
            capped_images = capped_images[:image_limit]
        return capped_texts, capped_images

    def _predict_with_runtime_fallbacks(self, question, texts, images):
        texts, images = self._apply_chunk_limits(texts, images)
        ans, token_usage = self._predict_single(question, texts, images)
        trace = {"runtime_fallback": "full"}
        if ans and not self._is_placeholder_or_refusal(ans):
            return ans, trace, token_usage

        return ans or "No valid answer generated.", trace, token_usage

    def _build_output_views(self, samples):
        keep_keys = set(self.KEEP_FIELDS + [self.config.ans_key])

        result_samples = []
        analysis_samples = []
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

            token_usage_item = {
                "doc_id": sample.get("doc_id"),
                "question": sample.get("question"),
                "token_usage": sample.get("token_usage"),
            }
            token_usage_samples.append(token_usage_item)

        return result_samples, analysis_samples, token_usage_samples

    def _dump_all_outputs(self, dataset: BaseDataset, samples, sample_no=None):
        result_samples, analysis_samples, token_usage_samples = self._build_output_views(samples)
        result_path = dataset.dump_reults(result_samples)

        base, _ = os.path.splitext(result_path)
        analysis_path = base + "_analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis_samples, f, indent=4)

        token_usage_path = base + "_token_usage.json"
        with open(token_usage_path, "w") as f:
            json.dump(token_usage_samples, f, indent=4)

        if sample_no is None:
            print(f"Save final results to {result_path}.")
        else:
            print(f"Save {sample_no} results to {result_path}.")
        print(f"Save analysis to {analysis_path}.")
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
            final_answer, trace, token_usage = self._predict_with_runtime_fallbacks(question, texts, images)

            sample[self.config.ans_key] = final_answer
            sample["token_usage"] = token_usage
            if getattr(self.config, "save_trace", True):
                sample[self.config.ans_key + "_trace"] = {
                    "mode": "single_agent",
                    "single_agent_model": getattr(self.config, "single_agent_model", "qwen2vl"),
                    **trace,
                }

            torch.cuda.empty_cache()
            sample_no += 1
            if sample_no % self.config.save_freq == 0:
                self._dump_all_outputs(dataset, samples, sample_no=sample_no)

        self._dump_all_outputs(dataset, samples)
