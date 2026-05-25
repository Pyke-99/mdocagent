#!/usr/bin/env python3
from pathlib import Path
import re

p = Path('tmp/TacticQA/tactic.txt')
text = p.read_text(encoding='utf-8')
lines = text.splitlines()
cleaned = []
for line in lines:
    if 'JSON Swiss' in line:
        continue
    if '本地生成' in line:
        continue
    # remove lines that are just 'Page N' or contain only whitespace and 'Page'
    if re.match(r"^\s*Page\s*\d+\s*$", line):
        continue
    # remove lines that are only long runs of spaces (page footer spacing)
    if line.strip() == '' and cleaned and cleaned[-1].strip() == '':
        # avoid consecutive blank lines
        continue
    cleaned.append(line)

# ensure single blank line after opening '[' if present
if cleaned and cleaned[0].strip() == '[' and len(cleaned) > 1 and cleaned[1].strip() == '':
    # remove the empty line after [
    del cleaned[1]

p.write_text('\n'.join(cleaned) + '\n', encoding='utf-8')
print('Cleaned', p)
