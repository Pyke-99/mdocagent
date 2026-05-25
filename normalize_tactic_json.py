#!/usr/bin/env python3
"""
Normalize tactic JSON files into RAG QA JSON list.

Usage:
  python normalize_tactic_json.py --input ./raw_jsons --output ./tactic_rag_qa.json --dataset-name TacticQA

Supports:
 - input path is a single .json file or a directory containing .json files
 - file content may be list[dict] or {"data": [...]} or single dict

Output: JSON list with fields in order: doc_id, q_uid, question, answer, doc_url, text-index-path-question
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any


def load_items_from_file(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if path.is_dir():
        for fp in sorted(path.glob('*.json')):
            try:
                data = json.load(open(fp, 'r', encoding='utf-8'))
            except Exception as e:
                print(f"Skip {fp}: load error {e}")
                continue
            if isinstance(data, list):
                items.extend(data)
            elif isinstance(data, dict):
                if 'data' in data and isinstance(data['data'], list):
                    items.extend(data['data'])
                else:
                    items.append(data)
    else:
        try:
            data = json.load(open(path, 'r', encoding='utf-8'))
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}")
        if isinstance(data, list):
            items.extend(data)
        elif isinstance(data, dict):
            if 'data' in data and isinstance(data['data'], list):
                items.extend(data['data'])
            else:
                items.append(data)

    return items


def _to_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, (list, tuple)):
        return '，'.join(str(x) for x in val)
    return str(val)


def normalize_item(item: Dict[str, Any], dataset_name: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    tid = item.get('Tactic_ID') or item.get('id') or item.get('Tactic_Name', 'unknown')
    doc_id = f"{tid}.json"
    tactic_name = item.get('Tactic_Name', '').strip()

    # helpers
    def make_entry(q_uid: str, question: str, answer: str) -> Dict[str, Any]:
        return {
            'doc_id': doc_id,
            'q_uid': q_uid,
            'question': question,
            'answer': answer,
            'doc_url': "",
            'text-index-path-question': f".ragatouille/colbert/indexes/{dataset_name}-question-{tid}.json",
        }

    # 1 objective
    q_uid = f"{tid}_objective"
    question = f"{tactic_name} 的任务目标是什么？"
    answer = _to_str(item.get('Objective', ''))
    out.append(make_entry(q_uid, question, answer))

    # 2 description
    q_uid = f"{tid}_description"
    question = f"{tactic_name} 的战术描述是什么？"
    answer = _to_str(item.get('Description', ''))
    out.append(make_entry(q_uid, question, answer))

    # 3 mission phase
    q_uid = f"{tid}_mission_phase"
    question = f"{tactic_name} 属于哪个任务阶段？"
    answer = _to_str(item.get('Mission_Phase', ''))
    out.append(make_entry(q_uid, question, answer))

    # 4 tactic type
    q_uid = f"{tid}_tactic_type"
    question = f"{tactic_name} 属于什么战术类型？"
    answer = _to_str(item.get('Tactic_Type', ''))
    out.append(make_entry(q_uid, question, answer))

    # 5 environment
    q_uid = f"{tid}_environment"
    question = f"{tactic_name} 适用于什么环境？"
    answer = _to_str(item.get('Applicable_Environment', ''))
    out.append(make_entry(q_uid, question, answer))

    # 6 tags
    q_uid = f"{tid}_tags"
    question = f"{tactic_name} 的语义标签有哪些？"
    tags = item.get('Semantic_Tags') or []
    answer = '，'.join(str(x) for x in tags) if isinstance(tags, (list, tuple)) else _to_str(tags)
    out.append(make_entry(q_uid, question, answer))

    # 7 execution time
    q_uid = f"{tid}_execution_time"
    question = f"{tactic_name} 的预计执行时间是多少？"
    exec_time = item.get('Execution_Time')
    if exec_time is None:
        answer = ""
    else:
        answer = f"{exec_time}秒"
    out.append(make_entry(q_uid, question, answer))

    # 8 credibility
    q_uid = f"{tid}_credibility"
    question = f"{tactic_name} 的可信度评分是多少？"
    answer = _to_str(item.get('Credibility', ''))
    out.append(make_entry(q_uid, question, answer))

    # 9 action sequence steps
    steps = item.get('Action_Sequence') or []
    if isinstance(steps, dict):
        # sometimes a mapping
        steps = [steps]
    for s in steps:
        step_no = s.get('Step') if 'Step' in s else s.get('step')
        try:
            step_display = int(step_no)
        except Exception:
            step_display = step_no if step_no is not None else ''

        q_uid = f"{tid}_step_{step_display}"
        question = f"{tactic_name} 的第 {step_display} 步行动是什么？"
        intent = _to_str(s.get('Intent', ''))
        visual = s.get('Visual_Aids') or []
        if isinstance(visual, str):
            visual_txt = visual
        else:
            visual_txt = '；'.join(str(x) for x in visual)
        answer = intent
        if visual_txt:
            answer = f"{intent} {visual_txt}"
        out.append(make_entry(q_uid, question, answer))

    # 10 visual overall
    q_uid = f"{tid}_visual_overall"
    question = f"{tactic_name} 的整体视觉辅助提示是什么？"
    overall = item.get('Visual_Aid_Overall') or []
    if isinstance(overall, str):
        overall_txt = overall
    else:
        overall_txt = '\n'.join(str(x) for x in overall)
    out.append(make_entry(q_uid, question, overall_txt))

    return out


def convert(input_path: str, output_path: str, dataset_name: str):
    p = Path(input_path)
    items = load_items_from_file(p)
    all_qas: List[Dict[str, Any]] = []
    for it in items:
        qas = normalize_item(it, dataset_name)
        all_qas.extend(qas)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_qas, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(all_qas)} QA entries to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Input file or directory')
    parser.add_argument('--output', '-o', required=True, help='Output JSON path')
    parser.add_argument('--dataset-name', '-d', default='TacticQA', help='Dataset name used in index path')
    args = parser.parse_args()
    convert(args.input, args.output, args.dataset_name)


if __name__ == '__main__':
    main()
