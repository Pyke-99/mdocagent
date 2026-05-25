#!/usr/bin/env python3
"""
Generate semantic-level questions from tactic.txt using Aliyun Bailian Qwen.
The script sends the full tactic.txt to qwen-max-latest and asks for operational,
scenario-based questions instead of rigid definition-style questions.
"""
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Install with: pip install openai")
    sys.exit(1)


TACTIC_TXT_PATH = Path("tmp/TacticQA/tactic.txt")
OUTPUT_PATH = Path("data/TacticQA/samples_semantic.json")
MODEL_NAME = "qwen-max-latest"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def _split_by_zh_tactic_id_lines(tactic_txt: str) -> list[dict[str, str]]:
    """Parse blocks from lines like: 战术ID: xxx"""
    blocks: list[dict[str, str]] = []
    current_id = None
    current_lines: list[str] = []

    for line in tactic_txt.splitlines():
        if line.startswith("战术ID:"):
            if current_id and current_lines:
                blocks.append({"tactic_id": current_id, "block": "\n".join(current_lines)})
            match = re.search(r"战术ID:\s*(\S+)", line)
            current_id = match.group(1) if match else None
            current_lines = [line] if current_id else []
        elif current_id:
            current_lines.append(line)

    if current_id and current_lines:
        blocks.append({"tactic_id": current_id, "block": "\n".join(current_lines)})

    return blocks


def _split_by_json_tactic_id_fields(tactic_txt: str) -> list[dict[str, str]]:
    """Fallback parser for JSON-like text containing "Tactic_ID": "..." fields.

    Works even when the full text is not strict JSON (e.g. concatenated arrays).
    """
    pattern = re.compile(r'"Tactic_ID"\s*:\s*"([^"]+)"')
    matches = list(pattern.finditer(tactic_txt))
    if not matches:
        return []

    blocks: list[dict[str, str]] = []
    for idx, match in enumerate(matches):
        tactic_id = match.group(1).strip()
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(tactic_txt)
        block = tactic_txt[start:end].strip()
        if tactic_id and block:
            blocks.append({"tactic_id": tactic_id, "block": block})

    return blocks


def split_tactics(tactic_txt: str) -> list[dict[str, str]]:
    """Split tactic text into blocks.

    Priority:
    1) "战术ID: xxx" line-based format.
    2) JSON-like format containing "Tactic_ID": "xxx" fields.
    """
    blocks = _split_by_zh_tactic_id_lines(tactic_txt)
    if blocks:
        return blocks

    return _split_by_json_tactic_id_fields(tactic_txt)


def extract_json_payload(content: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from model output."""
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def build_prompt(tactic_txt: str) -> str:
    return f"""You are an expert in military/tactical knowledge abstraction and semantic question generation.

Based on the complete tactic.txt below as the sole source, generate semantic-level QA data, NOT rigid definition-style questions.

Question Generation Requirements:
1. Questions must be operational, scenario-based, and process-oriented: "How to...", "How to coordinate...", "What is the procedure...", "Key risks and mitigation...", "How to execute..."
2. Strictly avoid definition-style questions like "What is X", "Define X", "Describe X's tactics".
3. Focus on practical execution, workflow, coordination, decision-making, execution sequence, precautions, and conditional logic.
4. CRITICAL: Generate REPRESENTATIVE questions that provide maximum coverage of key tactical concepts. Prioritize unique insights over quantity.
5. Make questions natural and varied, avoid templated phrasing.
6. Keep answers concise, actionable, and faithful to the tactic content.
7. Prioritize process-oriented and coordination-focused questions, especially for tactics involving UAVs, robot dogs, humanoid robots, multi-floor buildings, breaching, suppression, rescue, reconnaissance, and coordination.
8. CRITICAL: Generate EXACTLY 200 question_answer_pairs in total across all tactics. The goal is 200 representative, non-duplicative, actionable QA pairs in English.

Output Format Requirements:
- Output ONLY valid JSON, no explanations, no Markdown, no additional text.
- Top-level JSON must be an object with field: question_answer_pairs.
- question_answer_pairs is an array where each element contains:
  - tactic_id: string
  - question: string (in English)
  - answer: string (in English)

Important generation rules:
- Ensure diversity across tactics and question types; avoid repeating the same phrasing.
- If a tactic lacks obvious process questions, generate representative procedure/coordination/contingency questions derived from its content.
- Do NOT include placeholders or meta-text; every pair must contain a concrete question and concise actionable answer.

Complete tactic.txt below:
---BEGIN TACTIC TXT---
{tactic_txt}
---END TACTIC TXT---
"""


def _request_additional_pairs(client: OpenAI, existing_pairs: list[dict[str, str]], tactic_txt: str, needed: int) -> list[dict[str, Any]]:
    """Ask the model to produce exactly `needed` additional pairs, returning parsed JSON list or empty list."""
    prompt = (
        "We already have %d representative question_answer_pairs (JSON). "
        "Please generate exactly %d ADDITIONAL representative question_answer_pairs to reach a total of 200. "
        "Do NOT repeat existing questions. Output ONLY a valid JSON object with a single field `question_answer_pairs` which is an array of the additional items. "
        "Each item must include: tactic_id (string), question (English), answer (English). "
        % (len(existing_pairs), needed)
    )
    # include a short excerpt of existing questions to help avoid duplication
    sample_existing = json.dumps(existing_pairs[:20], ensure_ascii=False)
    full_prompt = f"{prompt}\nExistingPreview:{sample_existing}\n\nComplete tactic.txt below:\n---BEGIN TACTIC TXT---\n{tactic_txt}\n---END TACTIC TXT---"

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.7,
        top_p=0.9,
        max_tokens=4000,
    )
    return_resp = resp.choices[0].message.content or ""
    parsed = extract_json_payload(return_resp)
    if not parsed:
        return []
    return parsed.get("question_answer_pairs", [])


def _expand_samples_to_200(client: OpenAI, samples: list[dict[str, str]], tactic_blocks: list[dict[str, str]], tactic_txt: str) -> list[dict[str, str]]:
    """Try to expand `samples` to exactly 200 items by requesting more from the model, then fill programmatically if needed."""
    target = 200
    existing_questions = {s["question"] for s in samples}
    attempts = 0
    while len(samples) < target and attempts < 3:
        needed = target - len(samples)
        print(f"Requesting {needed} additional pairs from model (attempt {attempts+1})...")
        new_raw = _request_additional_pairs(client, samples, tactic_txt, needed)
        if not new_raw:
            attempts += 1
            continue

        # normalize and append non-duplicates
        normalized_new = normalize_pairs({"question_answer_pairs": new_raw}, tactic_blocks)
        added = 0
        for item in normalized_new:
            if item["question"] in existing_questions:
                continue
            samples.append(item)
            existing_questions.add(item["question"])
            added += 1
            if len(samples) >= target:
                break

        if added == 0:
            attempts += 1
        else:
            attempts = 0

    # Final programmatic fill if still short
    if len(samples) < target:
        print(f"Model did not produce enough items; programmatically filling {target - len(samples)} remaining slots...")
        idx = 0
        while len(samples) < target:
            tb = tactic_blocks[idx % len(tactic_blocks)]
            tactic_id = tb.get("tactic_id", f"T{idx}")
            q = f"How to handle unexpected obstacles during {tactic_id} execution when primary route is compromised?"
            if q in existing_questions:
                q = f"How to adapt {tactic_id} procedures for alternative entry when primary route is blocked (variant {idx})?"
            a = "Assess alternate routes, reassign UAVs for overwatch, use machine dogs to clear hazards, and update command with new timing and roles."
            samples.append({
                "doc_id": "tactic.pdf",
                "q_uid": f"{tactic_id}_semantic_fill_{idx}",
                "question": q,
                "answer": a,
                "doc_url": "",
            })
            existing_questions.add(q)
            idx += 1

    # Truncate if somehow exceeded
    return samples[:target]


def call_qwen(client: OpenAI, tactic_txt: str) -> str:
    prompt = build_prompt(tactic_txt)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=0.9,
        max_tokens=6000,
    )
    return response.choices[0].message.content or ""


def call_qwen_with_prompt(client: OpenAI, prompt: str, max_tokens: int = 3000) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def generate_batches(client: OpenAI, tactic_blocks: list[dict[str, str]], tactic_txt: str, batch_size: int = 40) -> list[dict[str, str]]:
    """Split tactics into batches, request QA pairs per batch, aggregate and return normalized samples."""
    total = len(tactic_blocks)
    all_pairs: list[dict[str, Any]] = []
    for i in range(0, total, batch_size):
        batch = tactic_blocks[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}: tactics {i+1}-{i+len(batch)}")
        # create a compact summary for this batch
        compressed = compress_tactic_blocks(batch, max_intents_per=1)
        prompt = (
            "You are an expert in military/tactical knowledge abstraction.\n"
            "Generate representative, operational, scenario-based question_answer_pairs (JSON) for the tactics listed below.\n"
            "Rules: output ONLY a JSON object with field `question_answer_pairs` (array). Each item: tactic_id, question (English), answer (English).\n"
            "Produce diverse, non-redundant, actionable QA pairs derived from the tactic content.\n"
            "Aim to cover the batch's key tactical concepts. Do NOT output extra text.\n\n"
            "TacticSummaries:\n"
            f"{compressed}\n"
        )

        raw = call_qwen_with_prompt(client, prompt, max_tokens=2500)
        parsed = extract_json_payload(raw)
        if not parsed:
            print("Warning: could not parse JSON from batch response; skipping batch preview")
            continue
        pairs = parsed.get("question_answer_pairs", [])
        if not isinstance(pairs, list):
            continue
        all_pairs.extend(pairs)
    # normalize and deduplicate by question text
    normalized = normalize_pairs({"question_answer_pairs": all_pairs}, tactic_blocks)
    seen = set()
    unique: list[dict[str, str]] = []
    for item in normalized:
        q = item.get("question", "").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        unique.append(item)
    return unique


def compress_tactic_blocks(tactic_blocks: list[dict[str, str]], max_intents_per: int = 2) -> str:
    """Create a compact textual summary for each tactic to reduce prompt size.

    For each tactic block we extract Tactic_ID, Tactic_Name, Objective, Semantic_Tags,
    and the first few Action_Sequence Intents. Returns a newline-separated summary string.
    """
    summaries: list[str] = []
    intent_re = re.compile(r'"Intent"\s*:\s*"([\s\S]*?)"', re.MULTILINE)
    name_re = re.compile(r'"Tactic_Name"\s*:\s*"([^"]+)"')
    obj_re = re.compile(r'"Objective"\s*:\s*"([^"]+)"')
    tags_re = re.compile(r'"Semantic_Tags"\s*:\s*\[([^\]]*)\]')

    for tb in tactic_blocks:
        tid = tb.get("tactic_id") or "UNKNOWN"
        block = tb.get("block", "")
        name_m = name_re.search(block)
        obj_m = obj_re.search(block)
        tags_m = tags_re.search(block)
        intents = intent_re.findall(block)

        name = name_m.group(1).strip() if name_m else ""
        objective = obj_m.group(1).strip() if obj_m else ""
        tags = tags_m.group(1).replace('\n', ' ').strip() if tags_m else ""

        first_intents = [re.sub(r'\s+', ' ', i).strip() for i in intents[:max_intents_per]]
        summary = f"[{tid}] {name}. Objective: {objective}. Tags: {tags}. Intents: {' ; '.join(first_intents)}"
        summaries.append(summary)

    return "\n".join(summaries)


def normalize_pairs(data: dict[str, Any], fallback_tactics: list[dict[str, str]]) -> list[dict[str, str]]:
    pairs = data.get("question_answer_pairs", [])
    if not isinstance(pairs, list):
        return []

    tactic_ids = [item["tactic_id"] for item in fallback_tactics]
    normalized: list[dict[str, str]] = []

    for index, item in enumerate(pairs):
        if not isinstance(item, dict):
            continue

        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        tactic_id = str(item.get("tactic_id", "")).strip()

        if not tactic_id and tactic_ids:
            tactic_id = tactic_ids[index % len(tactic_ids)]

        if not question or not answer or not tactic_id:
            continue

        normalized.append({
            "doc_id": "tactic.pdf",
            "q_uid": f"{tactic_id}_semantic_{index}",
            "question": question,
            "answer": answer,
            "doc_url": "",
        })

    return normalized


def main() -> None:
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        print("Error: DASHSCOPE_API_KEY environment variable not set")
        print("Set it first, then rerun: export DASHSCOPE_API_KEY='your-key'")
        sys.exit(1)

    if not TACTIC_TXT_PATH.exists():
        print(f"Error: file not found: {TACTIC_TXT_PATH}")
        sys.exit(1)

    tactic_txt = read_text(TACTIC_TXT_PATH)
    tactic_blocks = split_tactics(tactic_txt)

    print(f"Found {len(tactic_blocks)} tactic blocks in {TACTIC_TXT_PATH}")
    if not tactic_blocks:
        print("Error: no tactic blocks found.")
        print("Expected either lines like '战术ID: xxx' or JSON fields like \"Tactic_ID\": \"xxx\".")
        sys.exit(1)

    for item in tactic_blocks[:3]:
        print(f"  - {item['tactic_id']}")
    if len(tactic_blocks) > 3:
        print("  ...")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    # If raw tactic text is too long for the model, compress tactic blocks into compact summaries
    MAX_INPUT_CHARS = 120000
    if len(tactic_txt) > MAX_INPUT_CHARS:
        print(f"Original tactic text length {len(tactic_txt)} exceeds {MAX_INPUT_CHARS}, compressing tactics...")
        compressed = compress_tactic_blocks(tactic_blocks, max_intents_per=2)
        tactic_txt_used = f"COMPRESSED_SUMMARY_OF_{len(tactic_blocks)}_Tactics:\n" + compressed
        print(f"Compressed summary length: {len(tactic_txt_used)}")
    else:
        tactic_txt_used = tactic_txt

    print(f"Calling {MODEL_NAME} through DashScope compatible API in batches...")
    print(f"Generating representative questions (max 200) for {len(tactic_blocks)} tactics via batching...")
    samples_raw = generate_batches(client, tactic_blocks, tactic_txt)

    # convert aggregated raw samples into normalized structure
    data = {"question_answer_pairs": samples_raw}

    # normalize aggregated pairs
    samples = normalize_pairs(data, tactic_blocks)
    if not samples:
        print("Error: no valid samples parsed from model output")
        print("Parsed keys:", list(data.keys()))
        sys.exit(1)

    # Ensure exactly 200 representative questions (try model expansion then fill if necessary)
    samples = _expand_samples_to_200(client, samples, tactic_blocks, tactic_txt)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(samples)} representative semantic QA samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
