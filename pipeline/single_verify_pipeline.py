import json
import os

import torch
from tqdm import tqdm

from agents.verifier_agent import VerifierAgent
from mydatasets.base_dataset import BaseDataset
from pipeline.single_agent_pipeline import SingleAgentPipeline


class SingleVerifyPipeline:
    KEEP_FIELDS = SingleAgentPipeline.KEEP_FIELDS

    def __init__(self, config, single_model_cfg, verifier_model_cfg):
        self.config = config
        self.single = SingleAgentPipeline(config, single_model_cfg)
        # Reuse single registry cache for model dedup when configs are identical.
        verifier_key = self.single.registry.build_model_key(verifier_model_cfg)
        verifier_model = self.single.registry.get_or_create(verifier_key, verifier_model_cfg)
        self.verifier = VerifierAgent(verifier_model)

        verify_cfg = getattr(config, "single_verify", None)
        self.max_major_revision_rounds = int(getattr(verify_cfg, "max_major_revision_rounds", 1)) if verify_cfg else 1
        self.abstain_answer = (
            str(getattr(verify_cfg, "abstain_answer", "Insufficient evidence to answer."))
            if verify_cfg
            else "Insufficient evidence to answer."
        )

    def _has_valid_answer(self, sample):
        return self.single._has_valid_answer(sample)

    def _build_output_views(self, samples):
        keep_keys = set(self.KEEP_FIELDS + [self.config.ans_key])

        result_samples = []
        analysis_samples = []
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

        return result_samples, analysis_samples

    def _dump_all_outputs(self, dataset: BaseDataset, samples, sample_no=None):
        result_samples, analysis_samples = self._build_output_views(samples)
        result_path = dataset.dump_reults(result_samples)

        base, _ = os.path.splitext(result_path)
        analysis_path = base + "_analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis_samples, f, indent=4)

        if sample_no is None:
            print(f"Save final results to {result_path}.")
        else:
            print(f"Save {sample_no} results to {result_path}.")
        print(f"Save analysis to {analysis_path}.")

    def _build_evidence_ids(self, texts, images):
        return [f"text_{i}" for i in range(len(texts or []))] + [f"image_{i}" for i in range(len(images or []))]

    def _light_revise_answer(self, question, texts, images, previous_answer, issues, revision_instruction):
        prompt = (
            "You are lightly revising an existing document QA answer. "
            "Do not answer from scratch. "
            "Only fix wording, formatting, repeated tokens, or minor missing qualifications. "
            "Keep the core factual answer unchanged unless the listed issues explicitly require a minimal correction. "
            "Use only the provided chunks. "
            "Do not add unsupported facts. "
            "Return only one concise final answer sentence.\n\n"
            f"Question:\n{question}\n\n"
            f"Previous answer:\n{previous_answer}\n\n"
            f"Issues:\n{json.dumps(issues or [], ensure_ascii=True)}\n\n"
            f"Revision instruction:\n{revision_instruction or ''}"
        )
        try:
            revised, _, _ = self.verifier.model.predict(prompt, texts=texts or None, images=images or None)
            revised = str(revised or "").strip()
            if revised:
                return revised
        except Exception:
            pass
        return str(previous_answer or "").strip()

    def _single_revision_answer(
        self,
        question,
        texts,
        images,
        previous_answer,
        issues,
        missing_requirements,
        unsupported_claims,
        revision_instruction,
    ):
        revised_question = (
            str(question or "")
            + "\n\n[Revision request]\n"
            + "Fix the previous answer according to the verifier feedback.\n"
            + "Use only provided chunks. If evidence is insufficient, say it cannot be determined.\n"
            + "Previous answer: "
            + str(previous_answer or "")
            + "\nIssues: "
            + json.dumps(issues or [], ensure_ascii=True)
            + "\nMissing requirements: "
            + json.dumps(missing_requirements or [], ensure_ascii=True)
            + "\nUnsupported claims: "
            + json.dumps(unsupported_claims or [], ensure_ascii=True)
            + "\nInstruction: "
            + str(revision_instruction or "")
        )
        return self.single._predict_single(revised_question, texts, images)

    def _run_single_verify(self, question, texts, images):
        capped_texts, capped_images = self.single._apply_chunk_limits(texts, images)
        evidence = self._build_evidence_ids(capped_texts, capped_images)

        initial_answer, initial_trace = self.single._predict_with_runtime_fallbacks(question, capped_texts, capped_images)
        single_calls = 1
        verifier_calls = 0
        revision_calls = 0
        
        current_answer = initial_answer
        final_answer = str(initial_answer or "").strip() or self.abstain_answer
        revised_answer = ""
        trace_data = {}

        for round_idx in range(self.max_major_revision_rounds + 1):
            verifier_out = self.verifier.run(question, capped_texts, capped_images, current_answer, evidence)
            verifier_calls += 1
            verdict = verifier_out.verdict
            pass_reason = verifier_out.pass_reason
            issues = verifier_out.issues
            claim_evidence_map = verifier_out.claim_evidence_map
            missing_requirements = verifier_out.missing_requirements
            format_issues = verifier_out.format_issues
            revision_instruction = verifier_out.revision_instruction
            unsupported_claims = [
                c for c in (claim_evidence_map or []) if str(c.get("support_status", "")).lower() == "unsupported"
            ]
            
            # 记录当前轮次的验证结果，用于 trace（记录第一次的，或最后一次的）
            if round_idx == 0:
                trace_data.update({
                    "verifier_verdict": verdict,
                    "pass_reason": str(pass_reason or ""),
                    "issues": issues,
                    "claim_evidence_map": claim_evidence_map,
                    "missing_requirements": missing_requirements,
                    "format_issues": format_issues,
                    "revision_instruction": revision_instruction,
                })

            if verdict == "pass":
                final_answer = str(current_answer or "").strip() or self.abstain_answer
                break
            elif verdict == "minor_revise":
                revised_answer = self._light_revise_answer(
                    question,
                    capped_texts,
                    capped_images,
                    current_answer,
                    issues,
                    revision_instruction,
                )
                revision_calls += 1
                final_answer = revised_answer or (str(current_answer or "").strip() or self.abstain_answer)
                break
            elif verdict == "major_revise":
                if round_idx < self.max_major_revision_rounds:
                    revised_answer = self._single_revision_answer(
                        question,
                        capped_texts,
                        capped_images,
                        current_answer,
                        issues,
                        missing_requirements,
                        unsupported_claims,
                        revision_instruction,
                    )
                    revision_calls += 1
                    current_answer = revised_answer or (str(current_answer or "").strip() or self.abstain_answer)
                    final_answer = current_answer
                else:
                    final_answer = str(current_answer or "").strip() or self.abstain_answer
            else:
                final_answer = self.abstain_answer
                break

        total_calls = single_calls + verifier_calls + revision_calls
        trace_data.update({
            "mode": "single_verify",
            "single_agent_model": getattr(self.config, "single_agent_model", "qwen2vl"),
            "evidence": evidence,
            "initial_answer": str(initial_answer or ""),
            "revised_answer": str(revised_answer or ""),
            "final_answer": str(final_answer or ""),
            "single_calls": single_calls,
            "verifier_calls": verifier_calls,
            "revision_calls": revision_calls,
            "total_calls": total_calls,
            "single_trace": initial_trace,
            "max_major_revision_rounds": self.max_major_revision_rounds,
            "rounds_run": round_idx + 1,
        })

        return final_answer, trace_data

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
            final_answer, trace = self._run_single_verify(question, texts, images)

            sample[self.config.ans_key] = final_answer
            if getattr(self.config, "save_trace", True):
                sample[self.config.ans_key + "_trace"] = trace

            torch.cuda.empty_cache()
            sample_no += 1
            if sample_no % self.config.save_freq == 0:
                self._dump_all_outputs(dataset, samples, sample_no=sample_no)

        self._dump_all_outputs(dataset, samples)
