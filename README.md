# 🤖 Project Chimera — AI Agent Swarm for Autonomous Security Remediation

> **An autonomous multi-agent pipeline that scans codebases for security vulnerabilities, generates fixes, validates them by running the project's real test suite, and creates a Pull Request — all powered by AI.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)

---

## 🏗️ Architecture

```
Input: Repository URL
        │
        ▼
┌──────────────────┐
│ 🔍 Scanner Agent │  ← Scans all source files for 7 vulnerability types
└────────┬─────────┘
         │ Vulnerability Report (JSON)
         ▼
┌──────────────────────┐
│ 🔧 Remediation Agent │  ← RAG-powered fix generation with knowledge base
└────────┬─────────────┘
         │ Applied fixes
         ▼
┌──────────────────────────────┐
│ 🧪 Test Runner Agent         │  ← Self-healing validator (THE DIFFERENTIATOR)
│   • Detects Django/Flask/Py  │
│   • Creates virtualenv       │
│   • Installs dependencies    │
│   • Runs REAL test suite     │
│   • Auto-reverts on failure  │
└────────┬─────────────────────┘
         │ Validated changes
         ▼
┌──────────────────────┐
│ 📝 Report Agent      │  ← Creates Issue + Pull Request with findings
└──────────────────────┘
```

## 🎯 What Makes This Different

| Feature | Project Chimera | GitHub Copilot | Snyk | CodeQL |
|---------|:-:|:-:|:-:|:-:|
| **Auto-fix vulnerabilities** | ✅ | ❌ | ❌ | ❌ |
| **Run project's real tests** | ✅ | ❌ | ❌ | ❌ |
| **Self-healing (auto-revert)** | ✅ | ❌ | ❌ | ❌ |
| **RAG-powered analysis** | ✅ | ❌ | ❌ | ❌ |
| **Multi-framework support** | ✅ | — | — | ✅ |

**The key differentiator:** Most tools just *find* vulnerabilities. Chimera finds them, *fixes* them, *validates the fixes don't break anything* by running the project's actual test suite, and *auto-reverts* if tests fail. No broken code ever gets committed.

## 🛡️ 7 Vulnerability Categories Detected

| # | Category | Severity | Example |
|---|----------|----------|---------|
| 1 | **Hardcoded Secrets** | 🟠 HIGH | API keys, passwords, tokens in source code |
| 2 | **SQL Injection** | 🔴 CRITICAL | f-strings in SQL queries |
| 3 | **Cross-Site Scripting** | 🟠 HIGH | Unescaped template variables |
| 4 | **Path Traversal** | 🟡 MEDIUM | User input in file paths |
| 5 | **Insecure Deserialization** | 🔴 CRITICAL | `pickle.loads()`, `eval()` |
| 6 | **Weak Cryptography** | 🟡 MEDIUM | MD5/SHA1 for passwords |
| 7 | **Missing Input Validation** | 🟡 MEDIUM | Request params used directly |

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- API key: Groq (free), Google Gemini, or Anthropic Claude

### Setup
```bash
# Clone the repo
git clone https://github.com/Umme-2003/Chimera.git
cd Chimera

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API key (GROQ_API_KEY recommended — free)

# Run the app
streamlit run app.py
```

## 📁 Project Structure

```
├── chimera.py                 # Main engine with LLM scanner + test generator
├── chimera_gitlab.py          # GitLab-adapted orchestration engine
├── chimera_core.py            # Shared core: test runner, framework detection
├── chimera_confidence.py      # Fix confidence scoring system
├── chimera_security_debt.py   # Security debt quantification
├── chimera_sustainability.py  # Compute savings & carbon tracking
├── app.py                     # Streamlit UI for demo
├── knowledge_base.txt         # RAG knowledge base for vulnerability patterns
├── agents/agent.yml           # Agent definition (system prompt + tools)
├── flows/flow.yml             # Multi-agent flow pipeline config
├── requirements.txt           # Python dependencies
└── .env.example               # Environment variable template
```

## 🧪 The Self-Healing Test Runner

This is Chimera's **core innovation**. When AI-generated fixes are applied:

1. **Framework Detection** — Automatically identifies Django, Flask, FastAPI, or generic Python
2. **Environment Setup** — Creates isolated virtualenv, installs project dependencies
3. **Dependency Healing** — Detects and fixes known compatibility issues (e.g., WTForms 3.0+)
4. **Test Execution** — Runs the project's *actual* test suite (pytest, Django test, unittest)
5. **Result Analysis** — Categorizes results: passed, failed, environment error, no tests
6. **Auto-Revert** — If tests fail, ALL changes are automatically reverted. Broken code never ships.

## 📊 Additional Features

- **Security Debt Quantification** — Calculates the "cost" of each vulnerability in developer-hours
- **Fix Confidence Scoring** — Rates each fix (0-100) based on syntax validity, test results, and change complexity
- **Sustainability Metrics** — Tracks compute savings (~70% vs full SAST) and carbon footprint

## 📜 License

MIT License — see [LICENSE](LICENSE)

---

*Built by [@Umme-2003](https://github.com/Umme-2003)*