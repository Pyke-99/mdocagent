import json
import re
from dataclasses import asdict
from typing import Dict, List, Tuple

from prompts.verifier_prompts import VERIFIER_PROMPT
from schemas.verifier_schema import VerifierOutput


class VerifierAgent:
    ALLOWED_VERDICTS = {"pass", "minor_revise", "major_revise", "abstain"}

    def __init__(self, model):
        self.model = model

    def run(self, question: str, texts: List[str], images: List[str], answer: str, evidence: List[str]) -> VerifierOutput:
        chunk_records = self._build_chunk_records(texts, images)
        payload = {
            "question": str(question or ""),
            "chunks": chunk_records,
            "answer": str(answer or ""),
            "evidence": [str(e) for e in (evidence or [])],
        }
        prompt = VERIFIER_PROMPT + "\n\nInput:\n" + json.dumps(payload, ensure_ascii=True)

        raw = ""
        try:
            raw, _, _ = self.model.predict(prompt, texts=texts or None, images=images or None)
        except Exception:
            raw = ""

        parsed = self._safe_json_parse(raw)
        heuristic = self._heuristic_assess(question, answer, chunk_records)

        if parsed:
            normalized = self._normalize(parsed)
            normalized = self._fill_from_heuristic(normalized, heuristic)
            return self._enforce_constraints(normalized, heuristic)

        fallback = self._fill_from_heuristic(self._fallback_verdict(answer), heuristic)
        return self._enforce_constraints(fallback, heuristic)

    def to_dict(self, output: VerifierOutput) -> Dict:
        return asdict(output)

    def _normalize(self, parsed: Dict) -> VerifierOutput:
        verdict = str(parsed.get("verdict", "abstain")).strip().lower()
        if verdict not in self.ALLOWED_VERDICTS:
            verdict = "abstain"

        pass_reason = str(parsed.get("pass_reason", "")).strip()

        issues_raw = parsed.get("issues", [])
        if isinstance(issues_raw, list):
            issues = [str(x).strip() for x in issues_raw if str(x).strip()]
        elif isinstance(issues_raw, str) and issues_raw.strip():
            issues = [issues_raw.strip()]
        else:
            issues = []

        claim_map_raw = parsed.get("claim_evidence_map", [])
        claim_evidence_map = self._normalize_claim_map(claim_map_raw)

        missing_req_raw = parsed.get("missing_requirements", [])
        if isinstance(missing_req_raw, list):
            missing_requirements = [str(x).strip() for x in missing_req_raw if str(x).strip()]
        elif isinstance(missing_req_raw, str) and missing_req_raw.strip():
            missing_requirements = [missing_req_raw.strip()]
        else:
            missing_requirements = []

        format_raw = parsed.get("format_issues", [])
        if isinstance(format_raw, list):
            format_issues = [str(x).strip() for x in format_raw if str(x).strip()]
        elif isinstance(format_raw, str) and format_raw.strip():
            format_issues = [format_raw.strip()]
        else:
            format_issues = []

        revision_instruction = str(parsed.get("revision_instruction", "")).strip()
        return VerifierOutput(
            verdict=verdict,
            pass_reason=pass_reason,
            issues=issues,
            claim_evidence_map=claim_evidence_map,
            missing_requirements=missing_requirements,
            format_issues=format_issues,
            revision_instruction=revision_instruction,
        )

    def _fallback_verdict(self, answer: str) -> VerifierOutput:
        text = str(answer or "").strip()
        if not text:
            return VerifierOutput(
                verdict="abstain",
                pass_reason="",
                issues=["empty_answer"],
                claim_evidence_map=[],
                missing_requirements=[],
                format_issues=[],
                revision_instruction="",
            )

        refusal_markers = [
            "insufficient information",
            "cannot answer",
            "not enough information",
            "unable to determine",
            "信息不足",
            "无法回答",
        ]
        low = text.lower()
        if any(m in low for m in refusal_markers):
            return VerifierOutput(
                verdict="abstain",
                pass_reason="",
                issues=["insufficient_evidence"],
                claim_evidence_map=[],
                missing_requirements=[],
                format_issues=[],
                revision_instruction="",
            )

        return VerifierOutput(
            verdict="minor_revise",
            pass_reason="",
            issues=["fallback_non_structured_verification"],
            claim_evidence_map=[],
            missing_requirements=[],
            format_issues=[],
            revision_instruction="Refine wording and ensure every core claim is evidence-supported.",
        )

    def _build_chunk_records(self, texts: List[str], images: List[str]) -> List[Dict[str, str]]:
        records = []
        for idx, txt in enumerate(texts or []):
            records.append({"evidence_id": f"text_{idx}", "modality": "text", "content": str(txt or "")})
        for idx, path in enumerate(images or []):
            records.append({"evidence_id": f"image_{idx}", "modality": "image", "content": str(path or "")})
        return records

    def _heuristic_assess(self, question: str, answer: str, chunk_records: List[Dict[str, str]]) -> Dict:
        q = str(question or "")
        a = str(answer or "").strip()
        requirements = self._extract_requirements(q)
        missing_requirements = self._check_missing_requirements(q, a, requirements)
        format_issues = self._detect_format_issues(q, a)

        claims = self._split_claims(a)
        claim_map = []
        supported_count = 0
        unsupported_count = 0
        for claim in claims:
            evidence_id, evidence_text, support_status = self._find_best_support(claim, chunk_records)
            if support_status in {"direct", "partial"}:
                supported_count += 1
            else:
                unsupported_count += 1
            claim_map.append(
                {
                    "claim": claim,
                    "evidence_id": evidence_id,
                    "evidence_text": evidence_text,
                    "support_status": support_status,
                }
            )

        issues = []
        if not a:
            issues.append("empty_answer")
        if missing_requirements:
            issues.append("missing_key_requirements")
        if format_issues:
            issues.append("format_pollution_detected")
        if unsupported_count > 0:
            issues.append("unsupported_claims_detected")
        if claims and supported_count == 0:
            issues.append("no_supported_core_claim")

        if not a:
            verdict = "abstain"
        elif missing_requirements:
            verdict = "major_revise"
        elif claims and supported_count == 0:
            verdict = "abstain"
        elif unsupported_count > 0:
            verdict = "major_revise"
        elif format_issues:
            verdict = "minor_revise"
        else:
            verdict = "pass"

        pass_reason = ""
        if verdict == "pass":
            pass_reason = "Core claims are evidence-supported; no key requirements missing; no format anomalies detected."

        revision_instruction = ""
        if verdict == "minor_revise":
            revision_instruction = "Clean formatting artifacts and keep the same evidence-supported core facts."
        elif verdict == "major_revise":
            revision_instruction = "Fix unsupported or missing key requirements and ensure all core claims are grounded in chunks only."

        return {
            "verdict": verdict,
            "pass_reason": pass_reason,
            "issues": issues,
            "claim_evidence_map": claim_map,
            "missing_requirements": missing_requirements,
            "format_issues": format_issues,
            "revision_instruction": revision_instruction,
            "supported_claim_count": supported_count,
            "unsupported_claim_count": unsupported_count,
        }

    def _fill_from_heuristic(self, model_out: VerifierOutput, heuristic: Dict) -> VerifierOutput:
        out = VerifierOutput(
            verdict=model_out.verdict,
            pass_reason=model_out.pass_reason,
            issues=list(model_out.issues or []),
            claim_evidence_map=list(model_out.claim_evidence_map or []),
            missing_requirements=list(model_out.missing_requirements or []),
            format_issues=list(model_out.format_issues or []),
            revision_instruction=model_out.revision_instruction,
        )

        if not out.claim_evidence_map:
            out.claim_evidence_map = list(heuristic.get("claim_evidence_map", []))
        if not out.missing_requirements:
            out.missing_requirements = list(heuristic.get("missing_requirements", []))
        if not out.format_issues:
            out.format_issues = list(heuristic.get("format_issues", []))
        if not out.issues:
            out.issues = list(heuristic.get("issues", []))
        if out.verdict == "pass" and not out.pass_reason:
            out.pass_reason = str(heuristic.get("pass_reason", "")).strip()
        if out.verdict in {"minor_revise", "major_revise"} and not out.revision_instruction:
            out.revision_instruction = str(heuristic.get("revision_instruction", "")).strip()
        return out

    def _enforce_constraints(self, output: VerifierOutput, heuristic: Dict) -> VerifierOutput:
        verdict = output.verdict if output.verdict in self.ALLOWED_VERDICTS else "abstain"

        claim_map = output.claim_evidence_map or []
        unsupported = [c for c in claim_map if str(c.get("support_status", "")).lower() == "unsupported"]
        has_support = any(str(c.get("support_status", "")).lower() in {"direct", "partial"} for c in claim_map)
        missing_requirements = output.missing_requirements or []
        format_issues = output.format_issues or []
        issues = output.issues or []

        # Hard downgrade: pass cannot have empty reasons/maps and cannot ignore strict failures.
        if verdict == "pass":
            if not output.pass_reason.strip() or not claim_map:
                verdict = "major_revise"
                issues.append("invalid_pass_without_reason_or_evidence_map")
            elif missing_requirements:
                verdict = "major_revise"
                issues.append("missing_key_requirements")
            elif format_issues:
                verdict = "minor_revise"
                issues.append("format_pollution_detected")
            elif unsupported:
                verdict = "major_revise"
                issues.append("unsupported_claims_detected")
            elif not has_support:
                verdict = "abstain"
                issues.append("no_supported_core_claim")

        # Missing key requirements must be major.
        if missing_requirements and verdict in {"pass", "minor_revise"}:
            verdict = "major_revise"
            if "missing_key_requirements" not in issues:
                issues.append("missing_key_requirements")

        # Visible format pollution cannot pass.
        if format_issues and verdict == "pass":
            verdict = "minor_revise"
            if "format_pollution_detected" not in issues:
                issues.append("format_pollution_detected")

        # No evidence support cannot pass.
        if not has_support and verdict == "pass":
            verdict = "abstain"
            if "no_supported_core_claim" not in issues:
                issues.append("no_supported_core_claim")

        # If non-pass, issues must be non-empty.
        if verdict != "pass" and not issues:
            issues = list(heuristic.get("issues", [])) or ["verification_requires_revision_or_abstain"]

        pass_reason = output.pass_reason.strip() if verdict == "pass" else ""
        if verdict == "pass" and not pass_reason:
            pass_reason = str(heuristic.get("pass_reason", "")).strip() or "All core claims are supported by provided evidence."

        revision_instruction = output.revision_instruction.strip()
        if verdict in {"minor_revise", "major_revise"} and not revision_instruction:
            revision_instruction = str(heuristic.get("revision_instruction", "")).strip()
            if not revision_instruction:
                revision_instruction = (
                    "Revise the answer to satisfy missing requirements, remove formatting pollution, and keep only evidence-supported claims."
                )

        return VerifierOutput(
            verdict=verdict,
            pass_reason=pass_reason,
            issues=list(dict.fromkeys([str(x).strip() for x in issues if str(x).strip()])),
            claim_evidence_map=claim_map,
            missing_requirements=missing_requirements,
            format_issues=format_issues,
            revision_instruction=revision_instruction,
        )

    def _normalize_claim_map(self, raw_value) -> List[Dict[str, str]]:
        out = []
        if not isinstance(raw_value, list):
            return out
        for item in raw_value:
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim", "")).strip()
            evidence_id = str(item.get("evidence_id", "")).strip()
            evidence_text = str(item.get("evidence_text", "")).strip()
            support_status = str(item.get("support_status", "")).strip().lower()
            if support_status not in {"direct", "partial", "unsupported"}:
                support_status = "unsupported"
            if not claim:
                continue
            out.append(
                {
                    "claim": claim,
                    "evidence_id": evidence_id,
                    "evidence_text": evidence_text,
                    "support_status": support_status,
                }
            )
        return out

    def _extract_requirements(self, question: str) -> List[str]:
        q = str(question or "").lower()
        reqs = []
        if "when and" in q or "which years" in q or "during which years" in q:
            reqs.append("time")
        if "play" in q or "film" in q or "role" in q or "what" in q:
            reqs.append("target_entity_or_event")
        if "maximum power" in q or " power " in q:
            reqs.append("power")
        if "torque" in q:
            reqs.append("torque")
        if "how many of each" in q:
            reqs.append("count_each_type")
        if "how many" in q and "of each" not in q:
            reqs.append("count")
        if "what number" in q or "lineage number" in q or ("which" in q and "number" in q):
            reqs.append("ordinal_number")
        if q.strip().startswith("who"):
            reqs.append("person_name")
        if "from" in q and "to" in q:
            reqs.append("range_match")
        if "who won" in q or "winner" in q:
            reqs.append("winner")
        if "which couples" in q:
            reqs.append("couples")
        return list(dict.fromkeys(reqs))

    def _check_missing_requirements(self, question: str, answer: str, requirements: List[str]) -> List[str]:
        a = str(answer or "").strip()
        low = a.lower()
        missing = []

        def has_time(text: str) -> bool:
            if re.search(r"\b(1[0-9]{3}|20[0-9]{2}|\d{4})\b", text):
                return True
            if re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", text):
                return True
            return False

        for req in requirements:
            ok = True
            if req == "time":
                ok = has_time(a)
            elif req == "target_entity_or_event":
                ok = len(re.findall(r"[A-Z][a-z]+", a)) >= 1 or len(a.split()) >= 3
            elif req == "power":
                ok = "power" in low or "kw" in low or "hp" in low
            elif req == "torque":
                ok = "torque" in low or "n" in low and "m" in low
            elif req == "count_each_type":
                ok = len(re.findall(r"\b\d+[\d,]*\b", a)) >= 2
            elif req == "count":
                ok = bool(re.search(r"\b\d+[\d,]*\b", a))
            elif req == "ordinal_number":
                ok = bool(re.search(r"\b\d+(st|nd|rd|th)\b", low) or re.search(r"\b(first|second|third|fourth|fifth)\b", low))
            elif req == "person_name":
                ok = len(re.findall(r"[A-Z][a-z]+", a)) >= 1
            elif req == "range_match":
                ok = bool(re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", a)) and ("to" in low or "-" in a or "from" in low)
            elif req == "winner":
                ok = "won" in low or "winner" in low
            elif req == "couples":
                ok = " and " in low or "," in a

            if not ok:
                missing.append(req)

        return list(dict.fromkeys(missing))

    def _detect_format_issues(self, question: str, answer: str) -> List[str]:
        q = str(question or "").lower()
        a = str(answer or "")
        low = a.lower()
        issues = []

        if re.search(r"\b([A-Za-z]{3,})\1\b", a):
            issues.append("concatenated_repeated_token")
        if re.search(r"\b(\w+)\s+\1\b", low):
            issues.append("adjacent_duplicate_word")
        if re.search(r"\(\d+,\d+\),\(\d+,\d+\)", a):
            issues.append("coordinate_or_ocr_pollution")
        if "|" in a and len(a.split("|")) >= 3:
            issues.append("table_row_residue")
        if re.search(r"\s{2,}", a):
            issues.append("abnormal_spacing")

        complex_q = any(k in q for k in ["when and", "which years", "how many of each", "maximum power and torque", "what year", "what number"])
        if complex_q and low.strip() in {"yes", "no"}:
            issues.append("overly_short_binary_answer_for_complex_question")

        return list(dict.fromkeys(issues))

    def _split_claims(self, answer: str) -> List[str]:
        a = str(answer or "").strip()
        if not a:
            return []
        parts = re.split(r"[;\n]+", a)
        claims = []
        for part in parts:
            p = part.strip(" .")
            if not p:
                continue
            if ", and " in p:
                sub = [x.strip(" .") for x in p.split(", and ") if x.strip(" .")]
                claims.extend(sub)
            elif " and " in p and len(p.split()) > 8:
                sub = [x.strip(" .") for x in p.split(" and ") if x.strip(" .")]
                claims.extend(sub)
            else:
                claims.append(p)
        if not claims:
            claims = [a]
        return claims[:10]

    def _find_best_support(self, claim: str, chunk_records: List[Dict[str, str]]) -> Tuple[str, str, str]:
        claim_tokens = self._tokenize(claim)
        best = ("", "", 0.0)
        for row in chunk_records:
            if row.get("modality") != "text":
                continue
            content = str(row.get("content", ""))
            c_tokens = self._tokenize(content)
            if not claim_tokens or not c_tokens:
                continue
            overlap = len(set(claim_tokens) & set(c_tokens)) / max(1, len(set(claim_tokens)))
            if overlap > best[2]:
                best = (str(row.get("evidence_id", "")), content[:220], overlap)

        evidence_id, evidence_text, score = best
        if score >= 0.55:
            status = "direct"
        elif score >= 0.25:
            status = "partial"
        else:
            status = "unsupported"
            evidence_id = ""
            evidence_text = ""
        return evidence_id, evidence_text, status

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in re.findall(r"[A-Za-z0-9]+", str(text or "").lower()) if len(t) >= 3]

    def _safe_json_parse(self, text: str):
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass

        fenced = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        for candidate in fenced:
            try:
                return json.loads(candidate)
            except Exception:
                continue

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
        return None
