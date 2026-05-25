"""
Route-Analysis-Answer-Verify (RAAV) Pipeline.

A training-free document QA architecture with four agents:
- Route Agent: lightweight routing (single vs multi)
- Analysis Agent: analyzes complex questions (multi path only)
- Answer Agent: unified answer generator
- Verify Agent: verifies answers (multi path only)

This pipeline does not modify existing architectures.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from agents.analysis_raav_agent import AnalysisRaavAgent
from agents.answer_raav_agent import AnswerRaavAgent
from agents.route_raav_agent import RouteRaavAgent
from agents.verify_raav_agent import VerifyRaavAgent
from mydatasets.base_dataset import BaseDataset
from pipeline.model_registry import ModelRegistry
from schemas.raav_schema import RAAvTraceEntry


class RouteAnalysisAnswerVerifyPipeline:
    """RAAV Pipeline: Route -> Analysis -> Answer -> Verify."""

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
        """
        Initialize RAAV pipeline.

        Args:
            config: Configuration object with pipeline settings
            model_cfg: Model configuration (typically qwen3)
        """
        self.config = config
        self.registry = ModelRegistry()

        # All agents share the same model for simplicity
        model_key = self.registry.build_model_key(model_cfg)
        self.model = self.registry.get_or_create(model_key, model_cfg)

        # Initialize agents
        self.route_agent = RouteRaavAgent(self.model)
        self.analysis_agent = AnalysisRaavAgent(self.model)
        self.answer_agent = AnswerRaavAgent(self.model)
        self.verify_agent = VerifyRaavAgent(self.model)

        # Configuration
        self.verify_single_path = getattr(config, "verify_single_path", False)
        self.max_revision_rounds = getattr(config, "max_revision_rounds", 1)
        self.enable_minor_revise = getattr(config, "enable_minor_revise", True)
        self.enable_conservative_fallback = getattr(config, "enable_conservative_fallback", True)
        self.abstain_answer = getattr(
            config,
            "abstain_answer",
            "Based on the provided chunks, there is not enough evidence to determine the answer.",
        )

    def _has_valid_answer(self, sample: Dict) -> bool:
        """Check if sample already has a valid answer."""
        if self.config.ans_key not in sample:
            return False
        value = sample.get(self.config.ans_key)
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
        return True

    def _build_output_views(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Build output views for results and analysis."""
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
                    "token_usage": sample.get(self.config.ans_key + "_trace", {}).get("token_usage", {}),
                }
            )

        return result_samples, analysis_samples, token_usage_samples

    def _dump_all_outputs(self, dataset: BaseDataset, samples: List[Dict], sample_no=None):
        """Dump all outputs to files."""
        result_samples, analysis_samples, token_usage_samples = self._build_output_views(samples)
        result_path = dataset.dump_reults(result_samples)

        base, _ = os.path.splitext(result_path)
        analysis_path = base + "_analysis.json"
        token_usage_path = base + "_token_usage.json"

        with open(analysis_path, "w") as f:
            json.dump(analysis_samples, f, indent=4, ensure_ascii=False)
        with open(token_usage_path, "w") as f:
            json.dump(token_usage_samples, f, indent=4, ensure_ascii=False)

        if sample_no is None:
            print(f"Save final results to {result_path}.")
        else:
            print(f"Save {sample_no} results to {result_path}.")
        print(f"Save analysis to {analysis_path}.")
        print(f"Save token usage to {token_usage_path}.")

    def _apply_minor_revise(self, answer: str, issues: List[str]) -> str:
        """
        Apply lightweight revisions for minor_revise verdict.

        Allowed revisions:
        - Remove duplicate words
        - Remove extra spaces
        - Fix light format issues
        - Fix light phrasing issues

        Prohibited revisions:
        - Names, numbers, dates, locations
        - Object relationships, counts, comparisons
        - Core conclusions

        Args:
            answer: The answer string
            issues: List of issues

        Returns:
            Revised answer
        """
        revised = answer

        # Simple cleanup
        revised = " ".join(revised.split())  # Remove extra spaces
        revised = revised.replace("  ", " ")  # Remove double spaces

        # Remove common duplicate patterns (very conservative)
        # Only remove if it's obvious repetition of a simple word
        words = revised.split()
        if len(words) > 1:
            seen_idx = {}
            for i, word in enumerate(words):
                lower_word = word.lower().strip(".,;:")
                if lower_word and len(lower_word) < 4:  # Very short words only
                    if lower_word in seen_idx and i - seen_idx[lower_word] <= 2:
                        # Likely duplicate, but be very conservative
                        pass
                    else:
                        seen_idx[lower_word] = i

        return revised

    def _is_rejection_answer(self, answer: str) -> bool:
        """Check whether an answer is a rejection-style response."""
        text = str(answer or "").lower()
        rejection_markers = [
            "cannot be determined",
            "can't be determined",
            "not specified",
            "not enough evidence",
            "insufficient evidence",
            "no relevant evidence",
            "unable to determine",
            "unable to answer",
            "does not explicitly",
            "not explicitly stated",
        ]
        return any(marker in text for marker in rejection_markers)

    def _has_severe_verify_issue(self, issues: List[str]) -> bool:
        """Detect severe verification issues that should prevent conservative fallback."""
        text = " ".join([str(x).lower() for x in issues or []])
        severe_markers = [
            "wrong number",
            "incorrect number",
            "wrong count",
            "incorrect count",
            "wrong score",
            "incorrect score",
            "wrong metric",
            "incorrect metric",
            "wrong value",
            "incorrect value",
            "wrong entity",
            "unsupported",
            "contradict",
            "conflict",
            "missing required",
            "does not answer",
            "not answer",
            "extra unsupported",
            "hallucinated",
        ]
        return any(marker in text for marker in severe_markers)

    def _normalize_final_answer(self, answer: str) -> str:
        """Normalize final answer to avoid Python repr lists; keep concise."""
        text = str(answer or "").strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                import ast

                value = ast.literal_eval(text)
                if isinstance(value, list):
                    return ", ".join(str(x) for x in value)
            except Exception:
                pass
        return text

    def predict_single(self, question: str, chunks: List[str]) -> Tuple[str, Dict]:
        """
        Single path: Question -> Answer Agent -> [Optional] Verify Agent.

        Args:
            question: Question string
            chunks: List of chunks

        Returns:
            (answer, trace)
        """
        # Direct answer
        answer_out = self.answer_agent.run(question, chunks)
        initial_answer = answer_out.answer

        trace = RAAvTraceEntry(
            mode="raav",
            route="single",
            question_type="simple",
            initial_answer=initial_answer,
            final_answer=initial_answer,
            final_action="single_return",
        )

        # Collect token usage
        token_usage = {"answer_agent": answer_out.token_usage}

        # Optional: verify single path if enabled
        if self.verify_single_path:
            verify_out = self.verify_agent.run(
                question,
                chunks,
                initial_answer,
                answer_out.used_evidence,
            )
            trace.verify_verdict = verify_out.verdict
            token_usage["verify_agent"] = verify_out.token_usage

            if verify_out.verdict == "pass":
                trace.final_action = "single_return"
            elif verify_out.verdict == "minor_revise":
                revised = self._apply_minor_revise(initial_answer, verify_out.issues)
                trace.revised_answer = revised
                trace.final_answer = revised
                trace.final_action = "minor_revise"
            else:
                # For single path, abstain on major_revise/abstain
                trace.final_answer = initial_answer
                trace.final_action = "single_return"

        # normalize final answer formatting
        trace.final_answer = self._normalize_final_answer(trace.final_answer)
        trace_dict = trace.to_dict()
        trace_dict["token_usage"] = token_usage
        return trace.final_answer, trace_dict

    def predict_multi(self, question: str, chunks: List[str], route_result: Dict) -> Tuple[str, Dict]:
        """
        Multi path: Question -> Route -> Analysis -> Answer -> Verify -> [Optional Revision].

        Args:
            question: Question string
            chunks: List of chunks
            route_result: Route agent output

        Returns:
            (answer, trace)
        """
        trace = RAAvTraceEntry(
            mode="raav",
            route="multi",
            question_type=route_result.get("question_type", ""),
        )

        # Step 1: Analysis
        analysis_out = self.analysis_agent.run(question, chunks, route_result)
        analysis_dict = self.analysis_agent.to_dict(analysis_out)

        # Debug trace: analysis requirements and evidence count
        trace.analysis_requirements = analysis_dict.get("question_requirements", [])
        trace.analysis_key_evidence_count = len(analysis_dict.get("key_evidence", []))

        # Step 2: Answer
        answer_out = self.answer_agent.run(question, chunks, analysis_dict)
        initial_answer = answer_out.answer
        trace.initial_answer = initial_answer

        # Step 3: Verify
        verify_out = self.verify_agent.run(
            question,
            chunks,
            initial_answer,
            answer_out.used_evidence,
            analysis_dict,
        )
        trace.verify_verdict = verify_out.verdict
        trace.verify_issues = verify_out.issues

        # Collect token usage
        token_usage = {
            "analysis_agent": analysis_out.token_usage,
            "answer_agent": answer_out.token_usage,
            "verify_agent": verify_out.token_usage,
        }

        # Step 4: Handle verdict
        if verify_out.verdict == "pass":
            trace.final_answer = initial_answer
            trace.final_action = "multi_pass"
            trace_dict = trace.to_dict()
            trace_dict["token_usage"] = token_usage
            return initial_answer, trace_dict

        elif verify_out.verdict == "minor_revise":
            if self.enable_minor_revise:
                revised = self._apply_minor_revise(initial_answer, verify_out.issues)
                trace.revised_answer = revised
                trace.final_answer = revised
                trace.final_action = "minor_revise"
                trace_dict = trace.to_dict()
                trace_dict["token_usage"] = token_usage
                return revised, trace_dict
            else:
                trace.final_answer = initial_answer
                trace.final_action = "minor_revise"
                trace_dict = trace.to_dict()
                trace_dict["token_usage"] = token_usage
                return initial_answer, trace_dict

        elif verify_out.verdict == "major_revise":
            # Attempt revision if within budget
            if self.max_revision_rounds >= 1:
                revision_answer_out = self.answer_agent.run_revision(
                    question,
                    chunks,
                    analysis_dict,
                    initial_answer,
                    verify_out.issues,
                    verify_out.revision_instruction,
                )
                revised_answer = revision_answer_out.answer
                trace.revised_answer = revised_answer
                token_usage["revision_answer_agent"] = revision_answer_out.token_usage

                # Re-verify
                re_verify_out = self.verify_agent.run(
                    question,
                    chunks,
                    revised_answer,
                    revision_answer_out.used_evidence,
                    analysis_dict,
                )
                trace.verify_verdict = re_verify_out.verdict
                trace.verify_issues = re_verify_out.issues
                token_usage["re_verify_agent"] = re_verify_out.token_usage

                if re_verify_out.verdict in {"pass", "minor_revise"}:
                    final_answer = revised_answer
                    if re_verify_out.verdict == "minor_revise" and self.enable_minor_revise:
                        final_answer = self._apply_minor_revise(revised_answer, re_verify_out.issues)
                    final_answer = self._normalize_final_answer(final_answer)
                    trace.final_answer = final_answer
                    trace.final_action = "major_revise_success"
                    trace_dict = trace.to_dict()
                    trace_dict["token_usage"] = token_usage
                    return final_answer, trace_dict

                if re_verify_out.verdict == "abstain":
                    if (
                        self.enable_conservative_fallback
                        and revised_answer
                        and not self._is_rejection_answer(revised_answer)
                        and not self._has_severe_verify_issue(re_verify_out.issues)
                    ):
                        final = self._normalize_final_answer(revised_answer)
                        trace.final_answer = final
                        trace.final_action = "fallback_conservative_answer"
                        trace_dict = trace.to_dict()
                        trace_dict["token_usage"] = token_usage
                        return final, trace_dict

                    trace.final_answer = self.abstain_answer
                    trace.final_action = "abstain"
                    trace_dict = trace.to_dict()
                    trace_dict["token_usage"] = token_usage
                    return self.abstain_answer, trace_dict

                # Still major_revise but revision budget exhausted.
                if (
                    self.enable_conservative_fallback
                    and revised_answer
                    and not self._is_rejection_answer(revised_answer)
                    and not self._has_severe_verify_issue(re_verify_out.issues)
                ):
                    final = self._normalize_final_answer(revised_answer)
                    trace.final_answer = final
                    trace.final_action = "fallback_conservative_answer"
                    trace_dict = trace.to_dict()
                    trace_dict["token_usage"] = token_usage
                    return final, trace_dict

                trace.final_answer = self.abstain_answer
                trace.final_action = "abstain"
                trace_dict = trace.to_dict()
                trace_dict["token_usage"] = token_usage
                return self.abstain_answer, trace_dict

            # No revision budget

            if (
                self.enable_conservative_fallback
                and initial_answer
                and not self._is_rejection_answer(initial_answer)
                and not self._has_severe_verify_issue(verify_out.issues)
            ):
                final = self._normalize_final_answer(initial_answer)
                trace.final_answer = final
                trace.final_action = "fallback_conservative_answer"
                trace_dict = trace.to_dict()
                trace_dict["token_usage"] = token_usage
                return final, trace_dict

            trace.final_answer = self.abstain_answer
            trace.final_action = "abstain"
            trace_dict = trace.to_dict()
            trace_dict["token_usage"] = token_usage
            return self.abstain_answer, trace_dict

        else:  # abstain
            if (
                self.enable_conservative_fallback
                and initial_answer
                and not self._is_rejection_answer(initial_answer)
                and not self._has_severe_verify_issue(verify_out.issues)
            ):
                final = self._normalize_final_answer(initial_answer)
                trace.final_answer = final
                trace.final_action = "fallback_conservative_answer"
                trace_dict = trace.to_dict()
                trace_dict["token_usage"] = token_usage
                return final, trace_dict

            trace.final_answer = self.abstain_answer
            trace.final_action = "abstain"
            trace_dict = trace.to_dict()
            trace_dict["token_usage"] = token_usage
            return self.abstain_answer, trace_dict

    def predict(
        self, question: str, chunks: List[str]
    ) -> Tuple[str, Dict]:
        """
        End-to-end prediction.

        Args:
            question: Question string
            chunks: List of chunks

        Returns:
            (final_answer, trace)
        """
        # Route
        route_result, route_token_usage = self.route_agent.run(question, chunks)
        route_dict = self.route_agent.to_dict(route_result)

        if route_dict["route"] == "single":
            answer, trace = self.predict_single(question, chunks)
        else:
            answer, trace = self.predict_multi(question, chunks, route_dict)

        # Add token usage to trace
        trace["token_usage"] = route_token_usage
        return answer, trace

    def predict_dataset(self, dataset: BaseDataset, resume_path=None):
        """
        Predict on entire dataset.

        Args:
            dataset: BaseDataset instance
            resume_path: Optional path to resume from
        """
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
            # Note: RAAV currently only uses texts, not images
            final_answer, trace = self.predict(question, texts or [])

            sample[self.config.ans_key] = final_answer
            if getattr(self.config, "save_trace", True):
                sample[self.config.ans_key + "_trace"] = trace

            torch.cuda.empty_cache()
            sample_no += 1
            if sample_no % self.config.save_freq == 0:
                self._dump_all_outputs(dataset, samples, sample_no=sample_no)

        self._dump_all_outputs(dataset, samples)
