#!/usr/bin/env python3
"""
Simple evaluation script for PaperTab results JSON.
Computes Exact Match (EM) and token-level F1 between `answer` and generated `ans_*` key.
Saves detailed per-sample results to a JSON file alongside printing a summary.

Usage:
  python scripts/eval_results.py \
    --results results/PaperTab/ptab_route3_qwen3/2026-04-30-01-07.json \
    --ans-key ans_ptab_route3_qwen3

"""
import argparse
import json
import os
import re
import string
from datetime import datetime

ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
PUNC = re.compile(r"[{}]".format(re.escape(string.punctuation)))


def normalize(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = text.strip()
    text = PUNC.sub("", text)
    text = ARTICLES.sub(" ", text)
    text = " ".join(text.split())
    return text


def exact_match(a, b):
    return int(normalize(a) == normalize(b))


def f1_score(a, b):
    a_tokens = normalize(a).split()
    b_tokens = normalize(b).split()
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    common = 0
    from collections import Counter
    a_count = Counter(a_tokens)
    b_count = Counter(b_tokens)
    for tok in a_count:
        common += min(a_count[tok], b_count.get(tok, 0))
    if common == 0:
        return 0.0
    precision = common / len(a_tokens)
    recall = common / len(b_tokens)
    return 2 * precision * recall / (precision + recall)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="Path to results JSON")
    p.add_argument("--ans-key", required=True, help="Answer key in JSON (e.g. ans_ptab_route3_qwen3)")
    p.add_argument("--out", default=None, help="Optional output path for per-sample eval JSON")
    args = p.parse_args()

    with open(args.results, 'r') as f:
        samples = json.load(f)

    total = 0
    evaluated = 0
    em_sum = 0
    f1_sum = 0.0
    details = []

    for i, s in enumerate(samples):
        gold = s.get('answer')
        pred = s.get(args.ans_key)
        if gold is None:
            # skip if no gold label
            continue
        total += 1
        if pred is None:
            details.append({
                'index': i,
                'doc_id': s.get('doc_id'),
                'question': s.get('question'),
                'gold': gold,
                'pred': pred,
                'em': 0,
                'f1': 0.0,
            })
            continue
        evaluated += 1
        em = exact_match(pred, gold)
        f1 = f1_score(pred, gold)
        em_sum += em
        f1_sum += f1
        details.append({
            'index': i,
            'doc_id': s.get('doc_id'),
            'question': s.get('question'),
            'gold': gold,
            'pred': pred,
            'em': em,
            'f1': f1,
        })

    em_rate = (em_sum / evaluated * 100) if evaluated else 0.0
    mean_f1 = (f1_sum / evaluated * 100) if evaluated else 0.0

    summary = {
        'results_file': args.results,
        'ans_key': args.ans_key,
        'timestamp': datetime.now().isoformat(),
        'total_with_gold': total,
        'evaluated_with_pred': evaluated,
        'exact_match_%': round(em_rate, 2),
        'mean_f1_%': round(mean_f1, 2),
        'missing_predictions': total - evaluated,
    }

    print('Evaluation summary:')
    for k, v in summary.items():
        print(f" - {k}: {v}")

    out_path = args.out
    if out_path is None:
        dirname = os.path.dirname(args.results)
        base = os.path.basename(args.results).rsplit('.', 1)[0]
        out_path = os.path.join(dirname, base + '_eval.json')

    with open(out_path, 'w') as f:
        json.dump({'summary': summary, 'details': details}, f, indent=4, ensure_ascii=False)

    print('\nPer-sample eval saved to', out_path)


if __name__ == '__main__':
    main()
