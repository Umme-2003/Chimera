#!/usr/bin/env python
"""Push Project Chimera to GitLab (core files only, excluding Tier-1)."""

import os
import subprocess
import sys

# Files/directories to push (CORE only)
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

# Directories to include
CORE_DIRS = [
    '.gitlab',
]

# Files to EXCLUDE (Tier-1 modules and docs)
EXCLUDE_FILES = [
    'chimera_sustainability.py',
    'chimera_confidence.py',
    'chimera_security_debt.py',
    'README.md',
    'SUBMISSION.md',
    'CHANGELOG.md',
    'COMPARISON_ANALYSIS.md',
    'TESTING_GUIDE.md',
    'WINNING_STRATEGY.md',
    'vulnerable_code.py',
    'vulnerable_code_corrected.py',
    'vulnerable_sql.py',
    'demo_features.py',
    'demo_vulnerabilities.py',
    'test_full_flow.py',
    'fix_chimera_gitlab.py',
    'run_demo.py',
    'architect.py',
    'app.py',  # Optional - might include if it exists
]

def run_cmd(cmd, description=""):
    """Run a git command and print output."""
    if description:
        print(f"\n>>> {description}")
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and "warning" not in result.stderr.lower():
        print(f"STDERR: {result.stderr}", file=sys.stderr)
    return result.returncode == 0

def main():
    os.chdir(r'C:\Users\Dell\Downloads\project-chimera-gitlab')

    print("=" * 60)
    print("PUSHING PROJECT CHIMERA TO GITLAB")
    print("=" * 60)
    print(f"\nCore files to push: {len(CORE_FILES)}")
    print(f"Core directories: {len(CORE_DIRS)}")

    # Step 1: Reset staging area
    run_cmd('git reset', "Resetting staging area")

    # Step 2: Remove venv files that might be cached
    run_cmd('git rm -r --cached venv 2>nul || echo "venv not cached"', "Removing venv from cache")
    run_cmd('git rm -r --cached __pycache__ 2>nul || echo "__pycache__ not cached"', "Removing __pycache__ from cache")

    # Step 3: Add core files that exist
    print("\n>>> Adding core files...")
    added = []
    for f in CORE_FILES:
        if os.path.exists(f):
            if run_cmd(f'git add "{f}"'):
                added.append(f)
                print(f"  + Added: {f}")
        else:
            print(f"  - Missing: {f}")

    # Step 4: Add core directories
    for d in CORE_DIRS:
        if os.path.exists(d):
            if run_cmd(f'git add "{d}"'):
                added.append(d)
                print(f"  + Added directory: {d}")

    print(f"\nTotal items staged: {len(added)}")

    # Step 5: Check status
    run_cmd('git status --short', "Git status")

    # Step 6: Commit
    commit_msg = '''feat: Project Chimera - 4-agent security remediation system

Core implementation featuring:
- Vulnerability Scanner with RAG-powered analysis
- Remediation Engineer with AI-generated fixes
- Test Runner with self-healing validation (venv isolation, framework detection)
- Report & MR Creator with secure credential handling

Key capabilities:
- Multi-framework test detection (Django/Flask/Python)
- Exponential backoff retry logic
- Clean metrics tracking
- GitLab Duo flow integration

Ready for GitLab Duo @mention testing.
'''
    print("\n>>> Committing...")
    # Write commit message to temp file to avoid escaping issues
    with open('.git_commit_msg.txt', 'w') as f:
        f.write(commit_msg)
    run_cmd('git commit -F .git_commit_msg.txt', "Committing changes")
    os.remove('.git_commit_msg.txt')

    # Step 7: Push
    print("\n" + "=" * 60)
    print("PUSHING TO GITLAB...")
    print("=" * 60)
    if run_cmd('git push -u origin main', "Pushing to GitLab"):
        print("\n✅ SUCCESS! Pushed to GitLab.")
    else:
        print("\n❌ PUSH FAILED. Check error messages above.")

if __name__ == '__main__':
    main()
