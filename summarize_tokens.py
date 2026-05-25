import json
import statistics

file_path = '/root/MDocAgent/results/TacticQA/tacticqa_classic_all2/2026-05-13-09-35_token_usage.json'
with open(file_path, 'r') as f:
    data = json.load(f)

samples = []
if isinstance(data, list):
    for entry in data:
        if 'token_usage' in entry:
            usage = entry['token_usage']
            samples.append({
                'prompt_tokens': usage['prompt_tokens'],
                'completion_tokens': usage['completion_tokens'],
                'total_tokens': usage['total_tokens'],
                'question': entry.get('question', 'N/A')
            })

if not samples:
    print("No samples found.")
    exit()

p_tokens = [s['prompt_tokens'] for s in samples]
c_tokens = [s['completion_tokens'] for s in samples]
t_tokens = [s['total_tokens'] for s in samples]

def stats(lst):
    return min(lst), max(lst), statistics.median(lst)

p_min, p_max, p_med = stats(p_tokens)
c_min, c_max, c_med = stats(c_tokens)
t_min, t_max, t_med = stats(t_tokens)

over_15k = len([p for p in p_tokens if p > 15000])
under_5k = len([p for p in p_tokens if p < 5000])

sorted_by_p = sorted(samples, key=lambda x: x['prompt_tokens'])
smallest_5 = sorted_by_p[:5]
largest_5 = sorted_by_p[-5:]

print(f"Prompt Tokens: Min={p_min}, Max={p_max}, Median={p_med}")
print(f"Completion Tokens: Min={c_min}, Max={c_max}, Median={c_med}")
print(f"Total Tokens: Min={t_min}, Max={t_max}, Median={t_med}")
print(f"Prompt > 15000: {over_15k}")
print(f"Prompt < 5000: {under_5k}")

print("\nSmallest 5 by Prompt Tokens:")
for s in smallest_5:
    print(f"- {s['prompt_tokens']}: {s['question']}")

print("\nLargest 5 by Prompt Tokens:")
for s in reversed(largest_5):
    print(f"- {s['prompt_tokens']}: {s['question']}")
