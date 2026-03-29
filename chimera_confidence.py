# CHIMERA_CONFIDENCE.PY — AI Confidence Scoring for Fixes
# Shows how confident the AI is in each fix

import ast
from typing import Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class ConfidenceFactors:
    """Factors that contribute to fix confidence."""
    syntax_valid: bool = False
    tests_pass: bool = False
    fix_complexity: float = 1.0  # 0.0-1.0, simple = better
    severity_match: bool = False
    risk_level: str = "unknown"


def calculate_fix_complexity(original_code: str, fixed_code: str) -> float:
    """
    Calculate complexity of the fix (0-1, where 1 = simple).
    Simpler fixes are safer.
    """
    original_lines = len([l for l in original_code.split('\n') if l.strip()])
    fixed_lines = len([l for l in fixed_code.split('\n') if l.strip()])

    # Calculate line difference
    line_diff = abs(fixed_lines - original_lines)
    max_lines = max(original_lines, fixed_lines, 1)

    # Complexity decreases with more changes
    complexity = 1.0 - (line_diff / max_lines)
    return max(0.0, min(1.0, complexity))


def validate_syntax(code: str) -> Tuple[bool, str]:
    """Validate Python syntax."""
    try:
        ast.parse(code)
        return True, "Syntax valid"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, str(e)


def classify_severity(vulnerability_type: str) -> Tuple[str, float]:
    """
    Classify vulnerability severity.
    Returns: (severity_level, confidence_weight)
    """
    critical_types = ['sql_injection', 'xss', 'insecure_deserialization', 'hardcoded_secret']
    high_types = ['path_traversal', 'weak_cryptography']
    medium_types = ['missing_input_validation']

    vuln_lower = vulnerability_type.lower()

    if any(v in vuln_lower for v in critical_types):
        return "CRITICAL", 1.0
    elif any(v in vuln_lower for v in high_types):
        return "HIGH", 0.9
    elif any(v in vuln_lower for v in medium_types):
        return "MEDIUM", 0.8
    else:
        return "LOW", 0.7


def rate_fix_confidence(
    original_code: str,
    fixed_code: str,
    test_status: str,
    test_output: str,
    vulnerability_type: str = "unknown",
    file_extension: str = ".py"
) -> Dict[str, Any]:
    """
    Rate the confidence in a fix (0-100%).

    Returns confidence score and detailed breakdown.
    """
    factors = ConfidenceFactors()

    # 1. Syntax validation (20% weight)
    syntax_valid, syntax_msg = validate_syntax(fixed_code)
    factors.syntax_valid = syntax_valid
    syntax_score = 20.0 if syntax_valid else 0.0

    # 2. Test validation (40% weight)
    tests_pass = test_status.startswith("SUCCESS")
    factors.tests_pass = tests_pass
    if tests_pass:
        test_score = 40.0
    elif "TUTORIAL" in test_status or "NO_TESTS" in test_status:
        # Tutorial projects without tests get partial credit
        test_score = 25.0
    else:
        test_score = 0.0

    # 3. Fix complexity (20% weight)
    complexity = calculate_fix_complexity(original_code, fixed_code)
    factors.fix_complexity = complexity
    complexity_score = complexity * 20.0

    # 4. Severity alignment (10% weight)
    severity, severity_weight = classify_severity(vulnerability_type)
    factors.risk_level = severity
    factors.severity_match = severity in ['CRITICAL', 'HIGH']
    severity_score = severity_weight * 10.0

    # 5. Code quality heuristics (10% weight)
    quality_score = 10.0

    # Penalize if fix removes important patterns
    if 'import' in original_code and 'import' not in fixed_code:
        quality_score -= 3.0

    # Bonus for adding security comments
    if '# security' in fixed_code.lower() or '# fix' in fixed_code.lower():
        quality_score += 2.0

    quality_score = max(0, min(10, quality_score))

    # Calculate total score
    total_score = syntax_score + test_score + complexity_score + severity_score + quality_score

    # Determine confidence level
    if total_score >= 90:
        confidence_level = "EXCELLENT"
        badge = "🟢"
        recommendation = "Safe to merge automatically"
    elif total_score >= 75:
        confidence_level = "HIGH"
        badge = "🟡"
        recommendation = "Likely safe, quick review recommended"
    elif total_score >= 60:
        confidence_level = "MEDIUM"
        badge = "🟠"
        recommendation = "Review recommended before merging"
    elif total_score >= 40:
        confidence_level = "LOW"
        badge = "🔴"
        recommendation = "Thorough review required"
    else:
        confidence_level = "CRITICAL"
        badge = "⛔"
        recommendation = "DO NOT MERGE - Manual fix required"

    return {
        "confidence_score": round(total_score, 1),
        "confidence_level": confidence_level,
        "badge": badge,
        "recommendation": recommendation,
        "factors": {
            "syntax_valid": {
                "score": syntax_score,
                "max": 20,
                "passed": syntax_valid,
            },
            "tests_pass": {
                "score": test_score,
                "max": 40,
                "passed": tests_pass,
            },
            "fix_complexity": {
                "score": round(complexity_score, 1),
                "max": 20,
                "complexity_value": round(complexity, 2),
            },
            "severity_alignment": {
                "score": round(severity_score, 1),
                "max": 10,
                "severity": severity,
            },
            "code_quality": {
                "score": round(quality_score, 1),
                "max": 10,
            },
        },
        "details": {
            "vulnerability_type": vulnerability_type,
            "test_status": test_status,
            "syntax_valid": syntax_valid,
            "lines_changed": f"{len(original_code.split(chr(10)))} → {len(fixed_code.split(chr(10)))}",
        }
    }


def format_confidence_report(confidence: Dict[str, Any]) -> str:
    """Format confidence report for display."""
    c = confidence
    factors = c['factors']

    return f"""
## {c['badge']} Fix Confidence: {c['confidence_score']}% ({c['confidence_level']})

**Recommendation:** {c['recommendation']}

### Score Breakdown

| Factor | Score | Status |
|--------|-------|--------|
| Syntax Validation | {factors['syntax_valid']['score']}/{factors['syntax_valid']['max']} | {'✅' if factors['syntax_valid']['passed'] else '❌'} |
| Test Validation | {factors['tests_pass']['score']}/{factors['tests_pass']['max']} | {'✅' if factors['tests_pass']['passed'] else '⚠️'} |
| Fix Complexity | {factors['fix_complexity']['score']}/{factors['fix_complexity']['max']} | Score: {factors['fix_complexity']['complexity_value']} |
| Severity Match | {factors['severity_alignment']['score']}/{factors['severity_alignment']['max']} | Level: {factors['severity_alignment']['severity']} |
| Code Quality | {factors['code_quality']['score']}/{factors['code_quality']['max']} | — |
| **TOTAL** | **{c['confidence_score']}%** | **{c['confidence_level']}** |

### Details
- **Vulnerability:** {c['details']['vulnerability_type']}
- **Test Status:** {c['details']['test_status']}
- **Lines Changed:** {c['details']['lines_changed']}
"""


def should_auto_merge(confidence: Dict[str, Any]) -> bool:
    """Determine if a fix should be auto-merged based on confidence."""
    return confidence['confidence_score'] >= 75 and confidence['confidence_level'] in ['EXCELLENT', 'HIGH']


def get_confidence_color(confidence_score: float) -> str:
    """Get color for confidence score."""
    if confidence_score >= 90:
        return "green"
    elif confidence_score >= 75:
        return "yellow"
    elif confidence_score >= 60:
        return "orange"
    else:
        return "red"
