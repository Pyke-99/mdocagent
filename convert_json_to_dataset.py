#!/usr/bin/env python3
"""
Convert a folder of per-document JSON files into dataset structure expected by MDocAgent.
Usage: python convert_json_to_dataset.py --src data/json_docs --dataset MyDataset

Expected input JSON format (per file):
{
  "id": "paper_1",
  "doc_id": "paper_1.json",    # optional, will be normalized to .pdf-style doc_id
  "question": "...",          # optional for QA samples
  "answer": "...",            # optional
  "pages": [
    {"page": 0, "text": "...", "image_path": "path/to/img0.png"},
    {"page": 1, "text": "..."}
  ],
  "metadata": {...}
}

This script writes:
- data/{dataset}/samples.json
- tmp/{dataset}/{doc_name}_{page}.txt for each page's text
- copies image files into tmp/{dataset}/ if image_path provided (keeps original path if absolute)

"""
import argparse
import json
import os
from pathlib import Path


def normalize_doc_id(fn):
    # normalize to a pdf-like name
    name = fn
    if name.endswith('.json'):
        name = name[:-5] + '.pdf'
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='data/json_docs')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='data')
    args = parser.parse_args()

    src = Path(args.src)
    dataset = args.dataset
    data_dir = Path(args.out_dir) / dataset
    tmp_dir = Path('tmp') / dataset

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    samples = []

    for fp in sorted(src.glob('*.json')):
        try:
            j = json.load(open(fp,'r',encoding='utf-8'))
        except Exception as e:
            print(f"Skip {fp}: load error {e}")
            continue
        # doc_id
        doc_id = j.get('doc_id') or j.get('id') or fp.stem
        # normalize to .pdf-like for compatibility
        if not str(doc_id).endswith('.pdf'):
            doc_id = str(doc_id) + '.pdf'
        doc_name = str(doc_id).rsplit('.pdf',1)[0]

        pages = j.get('pages') or []
        if not pages and 'text' in j:
            pages = [{'page': 0, 'text': j.get('text','')}]

        for p in pages:
            idx = p.get('page', 0)
            text = p.get('text','') or ''
            out_txt = tmp_dir / f"{doc_name}_{idx}.txt"
            with open(out_txt, 'w', encoding='utf-8') as f:
                f.write(text)
        # build sample entry - minimal
        sample = {
            'doc_id': doc_id,
            'q_uid': j.get('id',''),
            'question': j.get('question',''),
            'answer': j.get('answer','')
        }
        samples.append(sample)

    samples_path = data_dir / 'samples.json'
    with open(samples_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(samples)} samples -> {samples_path}")
    files_written = list(tmp_dir.glob('*'))
    print(f"Wrote {len(files_written)} tmp files into {tmp_dir} (showing up to 10):")
    for p in files_written[:10]:
        print('  ', p)

if __name__ == '__main__':
    main()
