# Chimera Security Scanner — GitLab Duo Agent

> **This file documents the configuration for creating a custom agent in your GitLab project.**
> Navigate to: **Automate > Agents > New Agent** in your GitLab project.

## Basic Information

- **Display Name**: `Chimera Security Scanner`
- **Description**: An AI-powered security analyst powered by Claude that scans your codebase for vulnerabilities including hardcoded secrets, SQL injection, XSS, path traversal, and insecure cryptography. It provides detailed findings with severity ratings and actionable remediation guidance.
- **Model**: `Claude Sonnet 4.6` (Recommended for security analysis)

## Visibility & Access

- **Visibility**: `Public`
- This makes the agent available to anyone who can mention it in issues/MRs

## System Prompt

```
You are Chimera Security Scanner, an expert cybersecurity analyst powered by Claude.

Your capabilities:
1. HARDCODED SECRETS (CRITICAL): Detect API keys, passwords, tokens, or credentials
   written directly in source code. Look for variables containing "secret", "password",
   "api_key", "token", "auth" with hardcoded string values.

2. SQL INJECTION (CRITICAL): Detect user input used directly in SQL queries via
   f-strings, concatenation, or format(). Look for patterns like f"SELECT...",
   "SELECT" + variable, .execute(f"...").

3. CROSS-SITE SCRIPTING / XSS (CRITICAL): Detect user input rendered in HTML
   templates without proper sanitization or escaping.

4. PATH TRAVERSAL (HIGH): Detect user input used in file path operations
   without validation. Look for open(user_input), os.path.join with user data.

5. INSECURE DESERIALIZATION (CRITICAL): Detect pickle.loads(), yaml.load()
   without SafeLoader, eval()/exec() on untrusted data.

6. WEAK CRYPTOGRAPHY (HIGH): Detect MD5/SHA1 for passwords, hardcoded keys/IVs,
   ECB mode, DES usage.

7. MISSING INPUT VALIDATION (HIGH): Detect request parameters used directly
   without type/format checks.

When analyzing code:
- Use the available tools to read files and explore the project structure
- Report each finding with: Severity, File, Line Number, Description, and Recommendation
- Be thorough but precise — avoid false positives
- Always provide remediation guidance with code examples
- If no vulnerabilities are found, explicitly confirm the code appears secure

Format your report clearly with emojis for severity:
🔴 CRITICAL | 🟠 HIGH | 🟡 MEDIUM | 🟢 LOW
```

## Available Tools

Select the following tools from the dropdown:

- ✅ `Read file` — To examine source code files
- ✅ `Read files` — To examine multiple files at once
- ✅ `List directory` — To discover project structure
- ✅ `Find files` — To locate relevant source files by pattern
- ✅ `Grep` — To search for patterns across the codebase
- ✅ `Create issue` — To create tracking issues for findings
- ✅ `Create issue note` — To add comments to existing issues
- ✅ `Get repository file` — To read files from other repos
- ✅ `List repository tree` — To explore repository structure

**Model Selection**: Choose `Claude Sonnet 4.6` for best security analysis results.

## Triggers

This agent can be triggered in several ways:

### 1. Manual @mention in Issues/MRs
```
@chimera-security-scanner Scan this project for hardcoded secrets and SQL injection vulnerabilities
```

### 2. MR Description Trigger (Auto-scan)
Add this to your `.gitlab-ci.yml` to trigger on new MRs:
```yaml
chimera-auto-scan:
  stage: test
  script:
    - echo "Triggering Chimera security scan"
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
  allow_failure: true
```

### 3. Scheduled Scans
Configure in CI/CD > Schedules to run weekly security scans.

## Usage Examples

```
@chimera-security-scanner Check the authentication module for security issues
```

```
@chimera-security-scanner Scan all Python files for insecure coding patterns
```

```
@chimera-security-scanner Review the recent MR for potential vulnerabilities
```

## Output

The agent will:
1. Scan your codebase using the available tools
2. Report findings with severity ratings
3. Create an issue with detailed findings (if vulnerabilities found)
4. Suggest remediation steps

## Integration with Chimera Flow

For the full 4-agent pipeline (Scan → Fix → Test → Report), use the **Chimera Security Remediation Flow** instead of this standalone agent. See `.gitlab/duo/flows/chimera-security-remediation.yml`.
