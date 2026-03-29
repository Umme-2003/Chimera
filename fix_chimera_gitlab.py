#!/usr/bin/env python
"""Fix syntax error in chimera_gitlab.py - backslash in f-string expression"""

import re

with open('chimera_gitlab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace {"\n".join(...) with {chr(10).join(...)
# The problematic pattern is inside f-string expressions
content = re.sub(
    r'\{"\\n"\.join\(f"',
    r'{chr(10).join(f"',
    content
)

with open('chimera_gitlab.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed: Replaced {\"\\n\".join with {chr(10).join")
