#!/usr/bin/env python
"""Commit only core files to GitLab."""

import os
import subprocess
import sys

os.chdir(r'C:\Users\Dell\Downloads\project-chimera-gitlab')

CORE_FILES = [
    '.env.example',
    '.gitignore',
    'requirements.txt',
    'knowledge_base.txt',
    '.gitlab-ci.yml',
    'chimera_core.py',
    'chimera_gitlab.py',
    'chimera.py',
    'main.py',
]

def run(cmd):
    print(f">>> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"stderr: {result.stderr}", file=sys.stderr)
    return result.returncode == 0

# Clear staging
run('git reset HEAD')

# Add core files one by one
for f in CORE_FILES:
    if os.path.exists(f):
        run(f'git add "{f}"')
    else:
        print(f"WARNING: {f} does not exist")

# Add .gitlab directory
if os.path.exists('.gitlab'):
    run('git add .gitlab/')

# Check what will be committed
print("\n" + "="*50)
print("Staged for commit:")
run('git diff --cached --name-only')

# Commit
commit_msg = """feat: Project Chimera - 4-agent security remediation

Core implementation:
- Vulnerability Scanner with RAG analysis
- Remediation Engineer with AI fixes
- Test Runner with self-healing validation
- Report & MR Creator with secure push"""

print("\n" + "="*50)
print("Committing...")
with open('.commit_msg.txt', 'w') as f:
    f.write(commit_msg)
run('git commit -F .commit_msg.txt')
os.remove('.commit_msg.txt')

# Push
print("\n" + "="*50)
print("Pushing to GitLab...")
run('git push -u origin main')

print("\nDone!")
