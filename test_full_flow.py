#!/usr/bin/env python3
"""
Test script to verify full Chimera flow with Security Debt and Confidence Scoring.
"""

import sys
sys.path.insert(0, 'C:/Users/Dell/Downloads/project-chimera-gitlab')

from chimera import run_chimera_orchestration
from chimera_security_debt import calculate_total_security_debt, format_security_debt_report, get_debt_severity_badge
from chimera_confidence import rate_fix_confidence, format_confidence_report, should_auto_merge

# ASCII-safe log callback (no emojis)
def log_callback(msg):
    # Replace problematic unicode characters with ASCII equivalents
    safe_msg = msg.replace('\u2705', '[OK]').replace('\u274c', '[FAIL]').replace('\u26a0', '[WARN]').replace('\ud83d\udd04', '[SYNC]').replace('\u2705', '[OK]').replace('\ud83d\udcca', '[INFO]').replace('\ud83d\udc1b', '[BUG]')
    print(safe_msg)

print("=" * 60)
print("PROJECT CHIMERA - FULL FLOW TEST")
print("=" * 60)
print()

# Run the full orchestration
changed_files, original_codes, corrected_codes = run_chimera_orchestration(
    repo_url='test_repo_demo',
    user_goal='Find and fix hardcoded secrets',
    github_username='test_user',
    log_callback=log_callback
)

print()
print("=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"Files changed: {len(changed_files)}")
print(f"Changed file paths: {changed_files}")
print()

if changed_files:
    # Calculate Security Debt
    print("=" * 60)
    print("SECURITY DEBT CALCULATION")
    print("=" * 60)

    vuln_list = []
    for fp in changed_files:
        vuln_list.append({
            'file': fp,
            'code': corrected_codes.get(fp, ''),
            'file_content': original_codes.get(fp, '')
        })

    debt_report = calculate_total_security_debt(vuln_list)

    print(f"Total Security Debt: ${debt_report['total_debt']:,.2f}")
    print(f"Annual Exposure Cost: ${debt_report['exposure_cost']:,.2f}")
    print(f"Breach Risk: {debt_report['breach_risk']}%")
    print(f"Vulnerability Count: {debt_report['vulnerability_count']}")
    print(f"Severity Badge: {get_debt_severity_badge(debt_report['total_debt'])}")
    print()
    print("Detailed Report:")
    print(format_security_debt_report(debt_report))

    # Calculate Confidence Scoring
    print()
    print("=" * 60)
    print("CONFIDENCE SCORING")
    print("=" * 60)

    for fp in changed_files:
        conf = rate_fix_confidence(
            original_code=original_codes.get(fp, ''),
            fixed_code=corrected_codes.get(fp, ''),
            test_status="SUCCESS",
            test_output="",
            vulnerability_type="hardcoded_secret",
            file_extension=".py"
        )

        print(f"File: {fp}")
        print(f"Confidence Score: {conf['confidence_score']}%")
        print(f"Confidence Level: {conf['confidence_level']}")
        print(f"Auto-merge eligible: {should_auto_merge(conf)}")
        print()
        print(format_confidence_report(conf))
        print()

    # Show before/after code comparison
    print("=" * 60)
    print("CODE COMPARISON")
    print("=" * 60)

    for fp in changed_files:
        print(f"\nFILE: {fp}")
        print("-" * 40)
        print("ORIGINAL:")
        print("-" * 40)
        print(original_codes.get(fp, 'N/A'))
        print()
        print("-" * 40)
        print("CORRECTED:")
        print("-" * 40)
        print(corrected_codes.get(fp, 'N/A'))
        print()
else:
    print("No files were changed.")

print()
print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)
