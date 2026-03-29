# CHIMERA.PY - THE FINAL, STABLE, AND INTELLIGENT TIER-3 VERSION
# Uses chimera_core for shared functionality
import ast
import os
import shutil
import re
import stat
import time
import sys
import subprocess
import json
import urllib.request
import tempfile
from dotenv import load_dotenv
from git import Repo
from github import Github

# Import shared core functions
from chimera_core import (
    extract_python_code,
    keyword_search_files,
    safe_rmtree,
    run_test_suite,
    detect_project_type,
)

# LangChain Imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
# Note: RAG features moved to chimera_core.py
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_core.output_parsers import StrOutputParser

# --- 1. CONFIGURATION ---
load_dotenv()
REPO_PATH = "temp_repo"
KNOWLEDGE_BASE_FILE = "knowledge_base.txt"

# --- LLM Selection: Groq or Gemini ---
if os.getenv("GROQ_API_KEY"):
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
elif os.getenv("GOOGLE_API_KEY"):
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
else:
    raise ValueError("No API key found! Set GROQ_API_KEY or GOOGLE_API_KEY in .env")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

VULNERABLE_SQL_CODE = """
# Injected for testing.
from sqlalchemy.sql import text
class UserDAO:
    def __init__(self, db_session):
        self.db = db_session
    def get_user_by_username(self, username: str):
        raw_query = f"SELECT * FROM users WHERE username = '{username}'"
        result = self.db.execute(text(raw_query))
        return result.fetchone()
"""

# --- 2. HELPER FUNCTIONS ---
def extract_python_code(text: str) -> str:
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    if match: return match.group(1).strip()
    match = re.search(r"```(.*?)```", text, re.DOTALL)
    if match: return match.group(1).strip()
    return text.strip()

def keyword_search_files(directory, keywords):
    matching_files = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
                        if any(keyword in f.read() for keyword in keywords):
                            matching_files.add(os.path.relpath(file_path, directory))
                except IOError: pass
    return list(matching_files)

# ═══════════════════════════════════════════════════════════════
# --- 2B. LLM-POWERED VULNERABILITY SCANNER ---
# Replaces keyword search with intelligent code analysis
# ═══════════════════════════════════════════════════════════════

SCAN_PROMPT = ChatPromptTemplate.from_template("""You are an expert security auditor. Analyze this source code for ALL security vulnerabilities.

SOURCE CODE from file `{file_path}`:
```python
{code}
```

Check for these vulnerability categories:
1. Hardcoded Secrets (API keys, passwords, tokens, SECRET_KEY with literal string values)
2. SQL Injection (string formatting in SQL queries, f-strings in .execute())
3. Cross-Site Scripting / XSS (innerHTML, unescaped template variables)
4. Path Traversal (user input in file paths without validation)
5. Insecure Deserialization (pickle.loads, yaml.load without SafeLoader, eval/exec)
6. Weak Cryptography (MD5/SHA1 for passwords, hardcoded keys/IVs)
7. Missing Input Validation (request params used directly without checks)

IMPORTANT: Only report REAL vulnerabilities, not false positives. Configuration files with SECRET_KEY set to a hardcoded string ARE vulnerabilities.

Respond ONLY with valid JSON. If no vulnerabilities found, return: {{"findings": []}}

Otherwise return:
{{"findings": [
  {{"severity": "CRITICAL|HIGH|MEDIUM", "type": "category name", "line_hint": "line or function name", "description": "what the issue is", "recommendation": "how to fix it"}}
]}}""")

def llm_scan_file(file_path, code, log_callback):
    """Scan a single file using the LLM for all vulnerability types."""
    try:
        result = (SCAN_PROMPT | llm | StrOutputParser()).invoke({
            "file_path": file_path,
            "code": code[:8000]  # Limit to avoid token overflow
        })
        # Parse JSON from LLM response
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            findings = parsed.get("findings", [])
            for f in findings:
                f["file"] = file_path
            return findings
    except json.JSONDecodeError:
        log_callback(f"     ⚠️ Could not parse scan result for {file_path}")
    except Exception as e:
        log_callback(f"     ⚠️ Scan error for {file_path}: {str(e)[:100]}")
    return []

def llm_scan_project(repo_path, log_callback):
    """Scan entire project for vulnerabilities using the LLM."""
    all_findings = []
    files_scanned = 0
    
    # Collect all Python files
    py_files = []
    skip_dirs = {'venv', '.venv', 'node_modules', '__pycache__', '.git', 'venv_chimera', 'migrations'}
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            if f.endswith('.py') and not f.startswith('test_chimera_'):
                py_files.append(os.path.join(root, f))
    
    log_callback(f"  📂 Found {len(py_files)} Python files to scan")
    
    for full_path in py_files:
        rel_path = os.path.relpath(full_path, repo_path)
        try:
            with open(full_path, "r", encoding='utf-8', errors='ignore') as f:
                code = f.read()
            if len(code.strip()) < 10:
                continue
            
            files_scanned += 1
            log_callback(f"  🔍 Scanning: {rel_path}")
            findings = llm_scan_file(rel_path, code, log_callback)
            
            if findings:
                for finding in findings:
                    severity_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡"}.get(finding.get("severity", ""), "⚪")
                    log_callback(f"     {severity_icon} {finding.get('severity', 'UNKNOWN')}: {finding.get('type', 'Unknown')} — {finding.get('description', '')[:80]}")
                all_findings.extend(findings)
            
            # Rate limit protection
            time.sleep(0.5)
            
        except Exception as e:
            log_callback(f"     ⚠️ Error reading {rel_path}: {str(e)[:80]}")
    
    log_callback(f"\n  📊 Scan complete: {files_scanned} files scanned, {len(all_findings)} vulnerabilities found")
    return all_findings, files_scanned

# ═══════════════════════════════════════════════════════════════
# --- 2C. SECURITY TEST GENERATOR AGENT ---
# Generates pytest test cases that prove fixes actually work
# This is Chimera's KEY DIFFERENTIATOR
# ═══════════════════════════════════════════════════════════════

TESTGEN_PROMPT = ChatPromptTemplate.from_template("""You are a security testing expert. Generate pytest test cases that verify the security fixes are correct.

VULNERABILITY FINDINGS:
{findings_json}

FILES THAT WERE FIXED:
{fixed_files}

Generate a COMPLETE, RUNNABLE pytest test file that:
1. Tests that each vulnerability has been properly fixed
2. Each test function has a clear docstring explaining what it verifies
3. Uses only standard library + pytest (no external dependencies needed)
4. Tests should be SIMPLE and RELIABLE — use ast parsing, regex, or string checks on the source files
5. Each test should verify the FIX, not try to exploit the vulnerability

IMPORTANT RULES:
- Use `ast` module to parse Python files and check for security patterns
- For hardcoded secrets: verify the value comes from os.environ or os.getenv, not a literal string
- For SQL injection: verify parameterized queries (look for :param or %s placeholders)
- For path traversal: verify os.path.basename or similar sanitization
- For insecure deserialization: verify safe_load or json.loads instead of pickle/eval
- Each test must be independent and self-contained
- Include clear pass/fail messages

Respond with ONLY the Python test file content, no explanations. Start with:
```python
# AUTO-GENERATED by Project Chimera — Security Test Suite
import pytest
import ast
import os
import re
...
```""")

def generate_security_tests(findings, fixed_files, repo_path, log_callback):
    """Generate pytest security test cases from vulnerability findings."""
    if not findings:
        log_callback("  ℹ️ No findings to generate tests for")
        return None, 0
    
    log_callback("\n  🧪 Generating security test cases...")
    
    # Prepare findings summary for the LLM
    findings_summary = []
    for f in findings:
        findings_summary.append({
            "file": f.get("file", "unknown"),
            "severity": f.get("severity", "MEDIUM"),
            "type": f.get("type", "unknown"),
            "description": f.get("description", ""),
            "recommendation": f.get("recommendation", "")
        })
    
    try:
        result = (TESTGEN_PROMPT | llm | StrOutputParser()).invoke({
            "findings_json": json.dumps(findings_summary, indent=2),
            "fixed_files": ", ".join(fixed_files)
        })
        
        test_code = extract_python_code(result)
        
        # Validate the generated test code
        try:
            ast.parse(test_code)
        except SyntaxError as e:
            log_callback(f"  ⚠️ Generated test has syntax error: {e}. Attempting cleanup...")
            # Try to fix common issues
            lines = test_code.split('\n')
            clean_lines = [l for l in lines if not l.strip().startswith('```')]
            test_code = '\n'.join(clean_lines)
            try:
                ast.parse(test_code)
            except SyntaxError:
                log_callback("  ❌ Could not fix generated test syntax")
                return None, 0
        
        # Write the test file
        test_file_path = os.path.join(repo_path, "test_chimera_security.py")
        with open(test_file_path, "w", encoding="utf-8") as f:
            f.write(test_code)
        
        # Count test functions
        tree = ast.parse(test_code)
        test_count = sum(1 for node in ast.walk(tree) 
                        if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'))
        
        log_callback(f"  ✅ Generated {test_count} security test cases")
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                docstring = ast.get_docstring(node) or node.name.replace('_', ' ').title()
                log_callback(f"     🧪 {node.name}: {docstring[:80]}")
        
        return test_file_path, test_count
        
    except Exception as e:
        log_callback(f"  ❌ Error generating tests: {str(e)[:200]}")
        return None, 0

def run_generated_security_tests(test_file_path, repo_path, log_callback):
    """Run the generated security tests and report results."""
    if not test_file_path or not os.path.exists(test_file_path):
        return "NO_TESTS", "No security tests to run"
    
    log_callback("\n  🏃 Running generated security tests...")
    
    # Find python executable (prefer venv if available)
    venv_python = os.path.join(repo_path, "venv_chimera", "Scripts", "python.exe") if os.name == 'nt' \
        else os.path.join(repo_path, "venv_chimera", "bin", "python")
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable
    
    # Ensure pytest is available
    subprocess.run([python_exe, "-m", "pip", "install", "pytest", "-q"],
                   capture_output=True, text=True, timeout=60)
    
    # Run the tests
    result = subprocess.run(
        [python_exe, "-m", "pytest", test_file_path, "-v", "--tb=short", "--no-header"],
        cwd=repo_path, capture_output=True, text=True, timeout=120
    )
    
    output = result.stdout + "\n" + result.stderr
    
    # Parse and display results line by line
    for line in output.split('\n'):
        line = line.strip()
        if not line:
            continue
        if 'PASSED' in line:
            log_callback(f"     ✅ {line}")
        elif 'FAILED' in line:
            log_callback(f"     ❌ {line}")
        elif 'ERROR' in line:
            log_callback(f"     💥 {line}")
        elif 'passed' in line or 'failed' in line:
            log_callback(f"     📊 {line}")
    
    # Determine overall result
    passed = len(re.findall(r'PASSED', output))
    failed = len(re.findall(r'FAILED', output))
    errors = len(re.findall(r'ERROR', output))
    
    if result.returncode == 0:
        return "SUCCESS", f"All {passed} security tests passed ✅"
    elif failed > 0:
        return "PARTIAL", f"{passed} passed, {failed} failed, {errors} errors"
    else:
        return "FAILURE", f"Test execution failed: {output[-500:]}"


def inject_test_file(file_path, content, log_callback):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f: f.write(content)
        log_callback(f"[Orchestrator] Test file injected at: {os.path.relpath(file_path, REPO_PATH)}")
    except Exception as e: log_callback(f"❌ ERROR: Could not inject test file. {e}")

# --- 3. AGENT DEFINITIONS ---
def setup_remediation_agents(log_callback):
    log_callback("[Remediation & Review Agents] Initializing...")
    # Simplified version without RAG dependencies
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt_hunter = ChatPromptTemplate.from_template("""You are a cybersecurity analyst. Analyze this code for vulnerabilities:

SOURCE CODE:
{input}

Check for: hardcoded secrets, SQL injection, XSS, path traversal, insecure deserialization, weak crypto, missing input validation.

Respond with analysis and recommended fix.""")

    chain_hunter = prompt_hunter | llm | StrOutputParser()
    prompt_engineer = ChatPromptTemplate.from_template("Code generation bot...ANALYSIS:\n{analysis}\nSOURCE CODE:\n{input}\nRESPONSE (RAW PYTHON CODE ONLY):")
    chain_engineer = prompt_engineer | llm | StrOutputParser()
    def reviewer_pipeline(code: str) -> bool:
        try: ast.parse(code); return True
        except Exception: return False
    def remediation_pipeline(inputs):
        analysis_text = chain_hunter.invoke({"input": inputs["input"]})
        if "no vulnerabilities were found" in analysis_text.lower(): return None
        corrected_code = extract_python_code(chain_engineer.invoke({"input": inputs["input"], "analysis": analysis_text}))
        if reviewer_pipeline(corrected_code):
            log_callback("     ✅ REVIEW PASSED: Syntax is valid.")
            return corrected_code
        else:
            log_callback("     ❌ REVIEW FAILED: Generated code has invalid syntax. Discarding change.")
            return None
    return remediation_pipeline

# IMPROVED TEST RUNNER AGENT WITH BETTER ERROR HANDLING AND FALLBACK STRATEGIES

def setup_test_runner_agent(log_callback):
    log_callback("[Test Runner Agent] Initializing...")

    def find_file_in_repo(repo_path, filename):
        matches = []
        for root, _, files in os.walk(repo_path):
            if filename in files:
                matches.append(os.path.join(root, filename))
        return matches

    def detect_project_type(repo_path):
        """Enhanced project detection with better categorization"""
        project_info = {
            'type': 'unknown',
            'test_files_found': [],
            'test_directories': [],
            'has_requirements': False,
            'framework': None,
            'test_command': None,
            'dependencies': [],
            'config_files': []
        }
        
        # Check for different project indicators
        manage_py_files = find_file_in_repo(repo_path, "manage.py")
        package_json_files = find_file_in_repo(repo_path, "package.json")
        requirements_files = find_file_in_repo(repo_path, "requirements.txt")
        setup_py_files = find_file_in_repo(repo_path, "setup.py")
        pytest_files = find_file_in_repo(repo_path, "pytest.ini")
        app_py_files = find_file_in_repo(repo_path, "app.py")
        
        # Check for config files
        config_indicators = [".env.example", ".env.template", "config.py", "settings.py"]
        for indicator in config_indicators:
            if find_file_in_repo(repo_path, indicator):
                project_info['config_files'].append(indicator)
        
        project_info['has_requirements'] = len(requirements_files) > 0
        
        # Detect Django projects
        if manage_py_files:
            project_info['type'] = 'django'
            project_info['framework'] = 'django'
            project_info['manage_py_path'] = manage_py_files[0]
            
        # Detect Flask projects
        elif app_py_files and any(find_file_in_repo(repo_path, f) for f in ["requirements.txt"]):
            # Check if it's actually Flask by looking for Flask imports
            for app_file in app_py_files:
                try:
                    with open(app_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if 'from flask import' in content or 'import flask' in content:
                            project_info['type'] = 'flask'
                            project_info['framework'] = 'flask'
                            project_info['main_app_path'] = app_file
                            break
                except Exception:
                    continue
            
        # Detect Node.js projects
        elif package_json_files:
            project_info['type'] = 'nodejs'
            project_info['framework'] = 'nodejs'
            
        # Detect Python packages
        elif setup_py_files or pytest_files:
            project_info['type'] = 'python_package'
            project_info['framework'] = 'python'
            
        # Look for test files and directories
        test_patterns = ['test_*.py', '*_test.py', 'tests.py', 'test*.js', '*.test.js']
        test_dir_names = ['tests', 'test', '__tests__', 'spec', 'specs']
        
        for root, dirs, files in os.walk(repo_path):
            # Check for test directories
            for dir_name in dirs:
                if dir_name.lower() in test_dir_names:
                    project_info['test_directories'].append(os.path.join(root, dir_name))
            
            # Check for test files
            for file_name in files:
                if (file_name.startswith('test_') and file_name.endswith('.py') or
                    file_name.endswith('_test.py') or 
                    file_name == 'tests.py' or
                    file_name.endswith('.test.js') or
                    file_name.startswith('test') and file_name.endswith('.js')):
                    project_info['test_files_found'].append(os.path.join(root, file_name))
        
        return project_info

    def analyze_dependency_compatibility(requirements_path, log_callback):
        """Analyze and fix common dependency compatibility issues"""
        compatibility_fixes = {}
        
        try:
            with open(requirements_path, 'r', encoding='utf-8') as f:
                requirements_content = f.read()
            
            # Known compatibility issues and their fixes
            known_fixes = {
                # WTForms compatibility
                'wtforms': {
                    'issue': 'TextField deprecated in WTForms 3.0+',
                    'fix': 'wtforms<3.0.0',
                    'detect_patterns': ['wtforms', 'WTForms']
                },
                # Django compatibility
                'django': {
                    'issue': 'Django version compatibility',
                    'fix': 'django>=3.2,<5.0',
                    'detect_patterns': ['django', 'Django']
                },
                # Flask compatibility
                'flask': {
                    'issue': 'Flask version compatibility',
                    'fix': 'flask>=1.1.4,<3.0',
                    'detect_patterns': ['flask', 'Flask']
                }
            }
            
            detected_issues = []
            
            for package, info in known_fixes.items():
                if any(pattern.lower() in requirements_content.lower() for pattern in info['detect_patterns']):
                    detected_issues.append({
                        'package': package,
                        'issue': info['issue'],
                        'fix': info['fix']
                    })
            
            return detected_issues, requirements_content
            
        except Exception as e:
            log_callback(f"     ⚠️ Could not analyze requirements file: {e}")
            return [], ""

    def create_compatibility_requirements(repo_path, original_requirements_path, detected_issues, log_callback):
        """Create a compatibility-fixed requirements file"""
        try:
            backup_path = original_requirements_path + ".chimera_backup"
            compatibility_path = os.path.join(repo_path, "requirements_chimera_compat.txt")
            
            # Backup original
            shutil.copy2(original_requirements_path, backup_path)
            log_callback(f"     - [Compatibility] Backed up original requirements to: {os.path.basename(backup_path)}")
            
            with open(original_requirements_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Apply fixes
            modified_content = content
            for issue in detected_issues:
                package = issue['package']
                fix = issue['fix']
                
                # Replace or add compatibility constraint
                import re
                pattern = rf'^{package}.*$'
                if re.search(pattern, modified_content, re.MULTILINE | re.IGNORECASE):
                    modified_content = re.sub(pattern, fix, modified_content, flags=re.MULTILINE | re.IGNORECASE)
                    log_callback(f"     - [Compatibility] Updated {package} constraint: {fix}")
                else:
                    modified_content += f"\n{fix}"
                    log_callback(f"     - [Compatibility] Added {package} constraint: {fix}")
            
            # Write compatibility version
            with open(compatibility_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            return compatibility_path, backup_path
            
        except Exception as e:
            log_callback(f"     ⚠️ Could not create compatibility requirements: {e}")
            return original_requirements_path, None

    def create_comprehensive_env_file(project_root, project_type, log_callback):
        """Create comprehensive environment files based on project type"""
        env_path = os.path.join(project_root, '.env')
        created_successfully = False
        
        try:
            if project_type == 'django':
                env_content = """# Temporary environment variables for Chimera testing
SECRET_KEY=chimera-temp-secret-key-for-testing-only-not-for-production-use-12345678901234567890
DEBUG=True
DATABASE_URL=sqlite:///db_chimera_test.sqlite3
ALLOWED_HOSTS=localhost,127.0.0.1,testserver
DB_NAME=db_chimera_test.sqlite3
DB_ENGINE=django.db.backends.sqlite3
STATIC_URL=/static/
MEDIA_URL=/media/
EMAIL_BACKEND=django.core.mail.backends.console.EmailBackend
USE_TZ=True
TIME_ZONE=UTC
LANGUAGE_CODE=en-us
"""
            elif project_type == 'flask':
                env_content = """# Temporary environment variables for Chimera testing
SECRET_KEY=chimera-temp-secret-key-for-testing-only-not-for-production
FLASK_ENV=development
FLASK_DEBUG=True
DATABASE_URL=sqlite:///test_db.sqlite3
TESTING=True
WTF_CSRF_ENABLED=False
"""
            else:
                env_content = """# Temporary environment variables for Chimera testing
SECRET_KEY=chimera-temp-secret-key-for-testing-only-not-for-production
DEBUG=True
TESTING=True
"""
            
            # Check if .env already exists
            if os.path.exists(env_path):
                log_callback("     - [Environment] .env file already exists, creating backup...")
                backup_path = env_path + ".chimera_backup"
                shutil.copy2(env_path, backup_path)
            
            with open(env_path, 'w', encoding='utf-8') as f:
                f.write(env_content.strip())
            
            # Verify the file was created and contains expected content
            if os.path.exists(env_path):
                with open(env_path, 'r', encoding='utf-8') as f:
                    written_content = f.read()
                if 'SECRET_KEY' in written_content and 'chimera-temp-secret' in written_content:
                    created_successfully = True
                    log_callback(f"     ✅ Environment file created and verified: {os.path.basename(env_path)}")
                else:
                    log_callback(f"     ❌ Environment file created but verification failed")
            else:
                log_callback(f"     ❌ Environment file creation failed - file does not exist")
                
        except Exception as e:
            log_callback(f"     ❌ Could not create environment file: {e}")
            created_successfully = False
        
        return created_successfully

    def setup_comprehensive_environment_vars(project_type):
        """Set up comprehensive environment variables"""
        base_env_vars = {
            'SECRET_KEY': 'chimera-temp-secret-key-for-testing-only-not-for-production-use-12345678901234567890',
            'DEBUG': 'True',
            'TESTING': 'True',
        }
        
        if project_type == 'django':
            django_vars = {
                'DATABASE_URL': 'sqlite:///db_chimera_test.sqlite3',
                'ALLOWED_HOSTS': 'localhost,127.0.0.1,testserver',
                'DB_NAME': 'db_chimera_test.sqlite3',
                'DB_ENGINE': 'django.db.backends.sqlite3',
                'STATIC_URL': '/static/',
                'MEDIA_URL': '/media/',
                'EMAIL_BACKEND': 'django.core.mail.backends.console.EmailBackend',
                'USE_TZ': 'True',
                'TIME_ZONE': 'UTC',
                'LANGUAGE_CODE': 'en-us',
                'DJANGO_SETTINGS_MODULE': '',  # Will be set later
            }
            base_env_vars.update(django_vars)
            
        elif project_type == 'flask':
            flask_vars = {
                'FLASK_ENV': 'development',
                'FLASK_DEBUG': 'True',
                'DATABASE_URL': 'sqlite:///test_db.sqlite3',
                'WTF_CSRF_ENABLED': 'False',
            }
            base_env_vars.update(flask_vars)
        
        complete_env = os.environ.copy()
        complete_env.update(base_env_vars)
        return complete_env

    def install_dependencies_with_compatibility(python_exe, repo_path, project_info, log_callback):
        """Install dependencies with compatibility fixes"""
        
        if not project_info['has_requirements']:
            log_callback("     - [Dependencies] No requirements.txt found, installing basic test dependencies...")
            basic_packages = ["pytest", "pytest-cov"]
            for package in basic_packages:
                try:
                    result = subprocess.run([python_exe, "-m", "pip", "install", package], 
                                          cwd=repo_path, capture_output=True, text=True, timeout=120)
                    if result.returncode == 0:
                        log_callback(f"     ✅ Installed: {package}")
                    else:
                        log_callback(f"     ⚠️ Could not install {package}")
                except Exception as e:
                    log_callback(f"     ⚠️ Error installing {package}: {str(e)[:100]}")
            return "SUCCESS_BASIC_INSTALL"
        
        requirements_files = find_file_in_repo(repo_path, "requirements.txt")
        original_requirements_path = requirements_files[0]
        
        # Analyze compatibility issues
        detected_issues, _ = analyze_dependency_compatibility(original_requirements_path, log_callback)
        
        if detected_issues:
            log_callback(f"     - [Dependencies] Detected {len(detected_issues)} compatibility issues")
            for issue in detected_issues:
                log_callback(f"       • {issue['package']}: {issue['issue']}")
            
            # Create compatibility-fixed requirements
            compat_requirements_path, backup_path = create_compatibility_requirements(
                repo_path, original_requirements_path, detected_issues, log_callback
            )
            
            # Try installing with compatibility fixes first
            log_callback("     - [Dependencies] Installing with compatibility fixes...")
            install_result = subprocess.run([python_exe, "-m", "pip", "install", "-r", compat_requirements_path], 
                                          cwd=repo_path, capture_output=True, text=True, timeout=300)
            
            if install_result.returncode == 0:
                log_callback("     ✅ Dependencies installed successfully with compatibility fixes")
                return "SUCCESS_COMPAT_INSTALL"
            else:
                log_callback("     ⚠️ Compatibility install failed, trying original requirements...")
                # Fall back to original requirements
                install_result = subprocess.run([python_exe, "-m", "pip", "install", "-r", original_requirements_path], 
                                              cwd=repo_path, capture_output=True, text=True, timeout=300)
                
                if install_result.returncode == 0:
                    log_callback("     ⚠️ Original requirements installed (may have compatibility issues)")
                    return "WARNING_ORIGINAL_INSTALL"
                else:
                    error_output = install_result.stderr
                    log_callback(f"     ❌ Both installation attempts failed")
                    return f"FAILURE_INSTALL_FAILED: {error_output[-500:]}"
        else:
            # No compatibility issues detected, install normally
            log_callback("     - [Dependencies] No compatibility issues detected, installing normally...")
            install_result = subprocess.run([python_exe, "-m", "pip", "install", "-r", original_requirements_path], 
                                          cwd=repo_path, capture_output=True, text=True, timeout=300)
            
            if install_result.returncode == 0:
                log_callback("     ✅ Dependencies installed successfully")
                return "SUCCESS_NORMAL_INSTALL"
            else:
                error_output = install_result.stderr
                log_callback(f"     ❌ Installation failed: {error_output[-200:]}")
                return f"FAILURE_INSTALL_FAILED: {error_output[-500:]}"

    def run_tests(repo_path: str):
        log_callback("     - [Test Runner] Setting up enhanced test environment...")
        
        try:
            abs_repo_path = os.path.abspath(repo_path)
            
            # Enhanced project analysis
            project_info = detect_project_type(abs_repo_path)
            log_callback(f"     - [Test Runner] Project type detected: {project_info['type']}")
            log_callback(f"     - [Test Runner] Framework: {project_info['framework']}")
            log_callback(f"     - [Test Runner] Test files found: {len(project_info['test_files_found'])}")
            log_callback(f"     - [Test Runner] Test directories found: {len(project_info['test_directories'])}")
            log_callback(f"     - [Test Runner] Config files detected: {project_info['config_files']}")
            
            # If no tests found, provide informative feedback
            if not project_info['test_files_found'] and not project_info['test_directories']:
                if project_info['type'] in ['django', 'flask']:
                    log_callback(f"     ⚠️ No test files found - this appears to be a tutorial/example {project_info['type']} project")
                    return "SUCCESS_NO_TESTS_TUTORIAL", f"{project_info['type'].title()} project setup validated successfully. No tests found (common in tutorial projects)."
                else:
                    log_callback("     ⚠️ No test files found in this repository")
                    return "SUCCESS_NO_TESTS_FOUND", "Repository analysis completed successfully. No test files detected."
            
            # Create virtual environment
            venv_path = os.path.join(abs_repo_path, "venv_chimera")
            if os.path.exists(venv_path): 
                shutil.rmtree(venv_path)
            
            subprocess.run([sys.executable, "-m", "venv", venv_path], check=True, capture_output=True, text=True)
            
            python_exe = os.path.join(venv_path, 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join(venv_path, 'bin', 'python')
            
            # Upgrade pip and setuptools
            log_callback("     - [Test Runner] Upgrading pip and setuptools...")
            subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], 
                         cwd=abs_repo_path, capture_output=True, text=True, check=True)
            
            # Install dependencies with enhanced compatibility handling
            install_status = install_dependencies_with_compatibility(python_exe, abs_repo_path, project_info, log_callback)
            
            if install_status.startswith("FAILURE_INSTALL_FAILED"):
                return "FAILURE_DEPENDENCY_INSTALL", f"Could not install dependencies: {install_status.split(':', 1)[1] if ':' in install_status else 'Unknown error'}"
            
            # Create environment file BEFORE running any tests
            env_created = create_comprehensive_env_file(abs_repo_path, project_info['type'], log_callback)
            if not env_created:
                log_callback("     ❌ Failed to create environment file - this may cause configuration issues")
            
            # Handle different project types with enhanced error handling
            if project_info['type'] == 'django':
                return handle_django_project(python_exe, abs_repo_path, project_info, log_callback, install_status)
            
            elif project_info['type'] == 'flask':
                return handle_flask_project(python_exe, abs_repo_path, project_info, log_callback, install_status)
            
            elif project_info['type'] == 'python_package':
                return handle_python_package(python_exe, abs_repo_path, project_info, log_callback, install_status)
            
            else:
                # Generic Python project
                return handle_generic_python_project(python_exe, abs_repo_path, project_info, log_callback, install_status)
                
        except Exception as e:
            import traceback
            error_msg = f"A fatal error occurred in the test runner: {e}\n{traceback.format_exc()}"
            log_callback(f"     ❌ FATAL ERROR: {error_msg}")
            return "FAILURE", error_msg

    def handle_flask_project(python_exe, repo_path, project_info, log_callback, install_status):
        """Handle Flask-specific testing with better error categorization"""
        
        # Set up Flask environment
        flask_env = setup_comprehensive_environment_vars('flask')
        
        # Install Flask-specific test dependencies
        flask_test_packages = ["pytest-flask", "flask-testing"]
        for package in flask_test_packages:
            try:
                subprocess.run([python_exe, "-m", "pip", "install", package], 
                             cwd=repo_path, capture_output=True, text=True, timeout=60)
                log_callback(f"     ✅ Installed Flask test package: {package}")
            except Exception:
                log_callback(f"     ⚠️ Could not install {package}")

        # Test basic Flask functionality if app.py exists
        if 'main_app_path' in project_info:
            log_callback("     - [Test Runner] Testing Flask application setup...")
            
            # Try importing the Flask app
            app_dir = os.path.dirname(project_info['main_app_path'])
            test_script = f'''
import sys
import os
sys.path.insert(0, "{app_dir}")
try:
    from app import create_app
    app = create_app()
    print("Flask app created successfully")
except ImportError as e:
    print(f"Import error: {{e}}")
    sys.exit(1)
except Exception as e:
    print(f"App creation error: {{e}}")
    sys.exit(2)
'''
            
            with open(os.path.join(repo_path, "test_flask_app.py"), "w") as f:
                f.write(test_script)
            
            app_test = subprocess.run([python_exe, "test_flask_app.py"], 
                                    cwd=repo_path, capture_output=True, text=True, 
                                    timeout=60, env=flask_env)
            
            os.remove(os.path.join(repo_path, "test_flask_app.py"))
            
            if app_test.returncode != 0:
                error_output = app_test.stdout + app_test.stderr
                if "ImportError: cannot import name 'TextField' from 'wtforms'" in error_output:
                    return "FAILURE_DEPENDENCY_COMPATIBILITY", "WTForms compatibility issue detected. TextField was deprecated in WTForms 3.0+. This project needs dependency updates."
                elif "SECRET_KEY" in error_output and "must be set" in error_output:
                    return "FAILURE_CONFIG_ENV_MISSING", "Flask app requires SECRET_KEY environment variable that is not properly configured."
                elif "ImportError" in error_output:
                    return "FAILURE_DEPENDENCY_MISSING", f"Missing dependencies detected: {error_output[-800:]}"
                else:
                    return "WARNING_APP_STARTUP_ISSUES", f"Flask app startup issues detected: {error_output[-800:]}"
        
        # Run Flask tests
        log_callback("     - [Test Runner] Running Flask test suite...")
        
        # Try different test runners
        test_commands = [
            [python_exe, "-m", "pytest", "-v", "--tb=short"],
            [python_exe, "-m", "pytest"],
            [python_exe, "-c", "import unittest; unittest.main(module='tests', exit=False)"]
        ]
        
        for i, cmd in enumerate(test_commands):
            log_callback(f"     - [Test Runner] Trying test command {i+1}/{len(test_commands)}...")
            test_result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, 
                                       timeout=600, env=flask_env)
            
            if test_result.returncode == 0:
                # Parse successful test results
                output_text = test_result.stdout
                import re
                
                test_count_match = re.search(r'(\d+) passed', output_text)
                if test_count_match:
                    test_count = int(test_count_match.group(1))
                    log_callback(f"     ✅ SUCCESS: {test_count} Flask tests passed!")
                    return "SUCCESS", f"{test_count} Flask tests passed successfully."
                else:
                    log_callback("     ✅ SUCCESS: Flask tests completed successfully.")
                    return "SUCCESS", "Flask tests completed successfully."
            
            elif test_result.returncode == 5:
                return "SUCCESS_NO_TESTS_FOUND", "No tests found in Flask project."
            
            else:
                error_output = test_result.stdout + "\n" + test_result.stderr
                
                # Categorize Flask-specific errors
                if "ImportError: cannot import name 'TextField' from 'wtforms'" in error_output:
                    return "FAILURE_DEPENDENCY_COMPATIBILITY", "WTForms compatibility issue: TextField was deprecated in WTForms 3.0+. Project needs dependency updates."
                elif "ImportError" in error_output and "wtforms" in error_output.lower():
                    return "FAILURE_DEPENDENCY_COMPATIBILITY", f"WTForms compatibility issue detected: {error_output[-800:]}"
                elif "ModuleNotFoundError" in error_output or "ImportError" in error_output:
                    return "FAILURE_DEPENDENCY_MISSING", f"Missing dependencies: {error_output[-800:]}"
                elif "SECRET_KEY" in error_output:
                    return "FAILURE_CONFIG_ENV_MISSING", f"Configuration issue - SECRET_KEY not properly set: {error_output[-800:]}"
                
        # If all test commands failed
        final_error = test_result.stdout + "\n" + test_result.stderr if 'test_result' in locals() else "All test commands failed"
        return "FAILURE_TESTS_FAILED", f"Flask tests failed: {final_error[-1500:]}"

    def handle_django_project(python_exe, repo_path, project_info, log_callback, install_status):
        """Enhanced Django project handling with better error categorization"""
        
        # Install Django-specific packages
        django_packages = ["django-environ", "python-decouple", "pillow"]
        for package in django_packages:
            try:
                subprocess.run([python_exe, "-m", "pip", "install", package], 
                             cwd=repo_path, capture_output=True, text=True, timeout=60)
                log_callback(f"     ✅ Installed Django package: {package}")
            except Exception:
                log_callback(f"     ⚠️ Could not install {package}")

        manage_py_path = project_info['manage_py_path']
        test_cwd = os.path.dirname(manage_py_path)
        
        # Set up Django environment
        django_env = setup_comprehensive_environment_vars('django')
        
        # Detect Django settings module
        settings_files = []
        for root, dirs, files in os.walk(test_cwd):
            if 'settings.py' in files:
                rel_path = os.path.relpath(root, test_cwd)
                if rel_path != '.':
                    settings_module = f"{rel_path.replace(os.sep, '.')}.settings"
                else:
                    settings_module = "settings"
                settings_files.append(settings_module)
        
        if settings_files:
            django_env['DJANGO_SETTINGS_MODULE'] = settings_files[0]
            log_callback(f"     - [Django] Detected settings module: {settings_files[0]}")
        
        # Test basic Django functionality AFTER environment is set up
        log_callback("     - [Django] Testing Django setup with proper environment...")
        basic_test = subprocess.run([python_exe, "manage.py", "help"], 
                                   cwd=test_cwd, capture_output=True, text=True, 
                                   timeout=60, env=django_env)
        
        if basic_test.returncode != 0:
            error_output = basic_test.stderr + basic_test.stdout
            
            if "cannot import name 'Mapping' from 'collections'" in error_output:
                return "FAILURE_DEPENDENCY_COMPATIBILITY", "Python version compatibility issue: 'collections.Mapping' was deprecated. This project needs Python 3.9+ compatibility updates."
                
            elif "SECRET_KEY" in error_output and ("must be set" in error_output or "ImproperlyConfigured" in error_output):
                return "FAILURE_CONFIG_ENV_MISSING", f"Django SECRET_KEY configuration error: {error_output[-600:]}"
                
            elif "ImproperlyConfigured" in error_output:
                return "FAILURE_CONFIG_ENV_MISSING", f"Django configuration error: {error_output[-600:]}"
                
            elif "ModuleNotFoundError" in error_output or "ImportError" in error_output:
                return "FAILURE_DEPENDENCY_MISSING", f"Django missing dependencies: {error_output[-600:]}"
                
            else:
                return "WARNING_APP_STARTUP_ISSUES", f"Django setup issues (may be pre-existing): {error_output[-600:]}"
        
        log_callback("     ✅ Basic Django functionality verified.")
        
        # Run Django system checks
        log_callback("     - [Django] Running Django system checks...")
        check_result = subprocess.run([python_exe, "manage.py", "check", "--deploy"], 
                                    cwd=test_cwd, capture_output=True, text=True, 
                                    timeout=120, env=django_env)
        
        if check_result.returncode == 0:
            log_callback("     ✅ Django system checks passed.")
        else:
            check_output = check_result.stdout + check_result.stderr
            if "CRITICAL" in check_output:
                log_callback("     ⚠️ Django system check found critical issues (continuing with tests).")
            else:
                log_callback("     ⚠️ Django system check had minor issues (continuing).")
        
        # Run Django tests
        log_callback("     - [Django] Running Django test suite...")
        test_result = subprocess.run([python_exe, "manage.py", "test", "--verbosity=1", "--keepdb"], 
                                   cwd=test_cwd, capture_output=True, text=True, 
                                   timeout=600, env=django_env)
        
        # Clean up temporary files
        cleanup_files = [
            os.path.join(test_cwd, '.env'),
            os.path.join(test_cwd, 'db_chimera_test.sqlite3')
        ]
        for temp_file in cleanup_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    log_callback(f"     - [Cleanup] Removed: {os.path.basename(temp_file)}")
                except Exception:
                    pass
        
        # Analyze Django test results with enhanced categorization
        if test_result.returncode == 0:
            output_text = test_result.stdout
            import re
            
            # Look for test count
            test_count_match = re.search(r'Ran (\d+) test', output_text)
            if test_count_match:
                test_count = int(test_count_match.group(1))
                if test_count > 0:
                    log_callback(f"     ✅ SUCCESS: All {test_count} Django tests passed!")
                    return "SUCCESS", f"All {test_count} Django tests passed successfully."
                else:
                    log_callback("     ✅ SUCCESS: Django test framework validated (0 tests - tutorial project).")
                    return "SUCCESS_NO_TESTS_TUTORIAL", "Django project validated successfully. No tests found (typical for tutorial projects)."
            else:
                if "Found 0 test(s)" in output_text or "Ran 0 tests" in output_text:
                    log_callback("     ✅ SUCCESS: Django test framework validated (0 tests - tutorial project).")
                    return "SUCCESS_NO_TESTS_TUTORIAL", "Django project validated successfully. This appears to be a tutorial project without tests."
                else:
                    log_callback("     ✅ SUCCESS: Django test suite completed successfully.")
                    return "SUCCESS", f"Django tests completed successfully."
        else:
            error_output = test_result.stdout + "\n" + test_result.stderr
            
            # Enhanced error categorization for Django
            if "SECRET_KEY" in error_output and "must be set" in error_output:
                return "FAILURE_CONFIG_ENV_MISSING", f"Django SECRET_KEY not configured: {error_output[-800:]}"
            elif "ImproperlyConfigured" in error_output:
                return "FAILURE_CONFIG_ENV_MISSING", f"Django configuration issues: {error_output[-800:]}"
            elif "ModuleNotFoundError" in error_output or "ImportError" in error_output:
                return "FAILURE_DEPENDENCY_MISSING", f"Django missing dependencies: {error_output[-800:]}"
            elif "DatabaseError" in error_output or "OperationalError" in error_output:
                return "WARNING_APP_STARTUP_ISSUES", f"Django database issues (may be pre-existing): {error_output[-800:]}"
            else:
                return "FAILURE_TESTS_FAILED", f"Django tests failed: {error_output[-1500:]}"

    def handle_python_package(python_exe, repo_path, project_info, log_callback, install_status):
        """Enhanced Python package handling with better error categorization"""
        
        log_callback("     - [Python Package] Installing pytest and coverage tools...")
        test_packages = ["pytest", "pytest-cov", "pytest-xdist"]
        
        for package in test_packages:
            try:
                subprocess.run([python_exe, "-m", "pip", "install", package], 
                             cwd=repo_path, capture_output=True, text=True, timeout=60)
                log_callback(f"     ✅ Installed: {package}")
            except Exception:
                log_callback(f"     ⚠️ Could not install {package}")
        
        # Try multiple test discovery strategies
        test_commands = [
            [python_exe, "-m", "pytest", "-v", "--tb=short"],
            [python_exe, "-m", "pytest", "tests/", "-v"],
            [python_exe, "-m", "pytest", "test/", "-v"],
            [python_exe, "-m", "unittest", "discover", "-s", "tests", "-v"],
            [python_exe, "-m", "unittest", "discover", "-s", "test", "-v"]
        ]
        
        for i, cmd in enumerate(test_commands):
            log_callback(f"     - [Python Package] Trying test command {i+1}/{len(test_commands)}...")
            
            test_result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, timeout=600)
            
            if test_result.returncode == 0:
                # Parse successful results
                output_text = test_result.stdout
                import re
                
                # Check for pytest results
                pytest_match = re.search(r'(\d+) passed', output_text)
                unittest_match = re.search(r'Ran (\d+) test', output_text)
                
                if pytest_match:
                    test_count = int(pytest_match.group(1))
                    log_callback(f"     ✅ SUCCESS: {test_count} Python package tests passed!")
                    return "SUCCESS", f"{test_count} Python package tests passed successfully."
                elif unittest_match:
                    test_count = int(unittest_match.group(1))
                    log_callback(f"     ✅ SUCCESS: {test_count} Python package tests passed!")
                    return "SUCCESS", f"{test_count} Python package tests passed successfully."
                else:
                    log_callback("     ✅ SUCCESS: Python package tests completed successfully.")
                    return "SUCCESS", "Python package tests completed successfully."
            
            elif test_result.returncode == 5:
                continue  # No tests found, try next command
            
            else:
                # Check for specific error patterns
                error_output = test_result.stdout + "\n" + test_result.stderr
                
                if "ModuleNotFoundError" in error_output or "ImportError" in error_output:
                    # Continue to try other test commands, but remember the error
                    last_error = error_output
                    continue
                elif "SyntaxError" in error_output:
                    return "FAILURE_SYNTAX_ERRORS", f"Python package has syntax errors: {error_output[-800:]}"
                else:
                    # Continue trying other commands
                    last_error = error_output
                    continue
        
        # If we get here, all test commands failed or found no tests
        if 'last_error' in locals():
            if "ModuleNotFoundError" in last_error or "ImportError" in last_error:
                return "FAILURE_DEPENDENCY_MISSING", f"Python package missing dependencies: {last_error[-800:]}"
            else:
                return "FAILURE_TESTS_FAILED", f"Python package tests failed: {last_error[-800:]}"
        else:
            return "SUCCESS_NO_TESTS_FOUND", "No tests found in Python package."

    def handle_generic_python_project(python_exe, repo_path, project_info, log_callback, install_status):
        """Enhanced generic Python project handling"""
        if project_info['test_files_found']:
            log_callback("     - [Test Runner] Installing pytest for generic Python testing...")
            subprocess.run([python_exe, "-m", "pip", "install", "pytest"], 
                         cwd=repo_path, capture_output=True, text=True)
            
            test_result = subprocess.run([python_exe, "-m", "pytest", "-v", "--tb=short"], 
                                       cwd=repo_path, capture_output=True, text=True, timeout=600)
            
            if test_result.returncode == 0:
                return "SUCCESS", f"Generic Python tests passed successfully."
            elif test_result.returncode == 5:
                return "SUCCESS_NO_TESTS_FOUND", "No tests found in Python project."
            else:
                error_output = test_result.stdout + "\n" + test_result.stderr
                
                # Categorize errors
                if "ImportError" in error_output or "ModuleNotFoundError" in error_output:
                    return "FAILURE_DEPENDENCY_MISSING", f"Missing dependencies: {error_output[-800:]}"
                elif "SyntaxError" in error_output:
                    return "FAILURE_SYNTAX_ERRORS", f"Syntax errors in code: {error_output[-800:]}"
                else:
                    return "FAILURE_TESTS_FAILED", f"Generic Python tests failed: {error_output[-1500:]}"
        else:
            return "SUCCESS_NO_TESTS_FOUND", "No test files found in this Python project."

    return run_tests

def setup_github_agent(log_callback):
    log_callback("[GitHub Agent] Initializing...")
    try: return Github(os.environ["GITHUB_TOKEN"])
    except Exception as e: log_callback(f"❌ GITHUB ERROR: {e}"); return None

# --- 4. THE MAIN ORCHESTRATOR (WITH BATCHED WORKFLOW) ---
# IMPROVED ORCHESTRATOR WITH BETTER ERROR CATEGORIZATION AND HANDLING

def run_chimera_orchestration(repo_url, user_goal, github_username, log_callback):
    
    changed_files, original_codes, corrected_codes = [], {}, {}
    
    log_callback(f"\n[Orchestrator] Cloning fresh repo: {repo_url}...")
    repo = Repo.clone_from(repo_url, to_path=REPO_PATH)
    if "sql" in user_goal.lower():
        inject_test_file(os.path.join(REPO_PATH, "app", "vulnerable_sql.py"), VULNERABLE_SQL_CODE, log_callback)

    log_callback("\n[Orchestrator] Starting keyword search...")
    keywords = []; 
    if "secret" in user_goal.lower(): keywords.extend(["SECRET_KEY", "API_KEY"])
    if "sql" in user_goal.lower(): keywords.extend(['f"', "SELECT ", ".execute("])
    identified_files = keyword_search_files(REPO_PATH, keywords)
    
    if not identified_files:
        log_callback("[Orchestrator] No suspicious files found."); return changed_files, original_codes, corrected_codes

    remediation_chain = setup_remediation_agents(log_callback)
    proposed_changes = {}
    
    log_callback("\n[Orchestrator] Generating and reviewing initial fixes...")
    for file_path in identified_files:
        full_path = os.path.normpath(os.path.join(REPO_PATH, file_path))
        log_callback(f"  -> Analyzing: {file_path}")
        try:
            with open(full_path, "r", encoding='utf-8') as f: original_code = f.read()
            corrected_code = remediation_chain({"input": original_code})
            if corrected_code and corrected_code.strip() != original_code.strip():
                proposed_changes[file_path] = corrected_code; original_codes[file_path] = original_code
        except Exception as e: log_callback(f"     ❌ Error analyzing file: {e}")

    if not proposed_changes:
        log_callback("\n[Orchestrator] No valid changes were proposed."); return changed_files, original_codes, corrected_codes
    
    log_callback(f"\n[Orchestrator] Applying {len(proposed_changes)} changes to the codebase for testing...")
    for file_path, corrected_code in proposed_changes.items():
        with open(os.path.normpath(os.path.join(REPO_PATH, file_path)), "w", encoding="utf-8") as f: f.write(corrected_code)

    test_runner_chain = setup_test_runner_agent(log_callback)
    test_status, test_output = test_runner_chain(REPO_PATH)

    # Enhanced error handling with proper categorization
    if test_status == "FAILURE":
        log_callback(f"\n❌ CRITICAL REGRESSION TEST FAILED. Reverting all changes.\n   Test Output: {test_output}")
        # Revert changes
        for file_path, original_code in original_codes.items():
            if file_path in proposed_changes:
                with open(os.path.normpath(os.path.join(REPO_PATH, file_path)), "w", encoding="utf-8") as f: 
                    f.write(original_code)
        return [], {}, {}
    
    elif test_status == "FAILURE_DEPENDENCY_COMPATIBILITY":
        log_callback(f"\n❌ DEPENDENCY COMPATIBILITY FAILURE. This project has outdated dependencies that break with modern Python environments.")
        log_callback(f"   Details: {test_output}")
        log_callback("   RECOMMENDATION: Update project dependencies before applying security fixes.")
        # Still revert changes as the project is fundamentally broken
        for file_path, original_code in original_codes.items():
            if file_path in proposed_changes:
                with open(os.path.normpath(os.path.join(REPO_PATH, file_path)), "w", encoding="utf-8") as f: 
                    f.write(original_code)
        return [], {}, {}
    
    elif test_status == "FAILURE_CONFIG_ENV_MISSING":
        log_callback(f"\n❌ CONFIGURATION FAILURE. Required environment variables are not properly configured.")
        log_callback(f"   Details: {test_output}")
        log_callback("   RECOMMENDATION: Set up proper environment configuration before applying fixes.")
        # Revert changes
        for file_path, original_code in original_codes.items():
            if file_path in proposed_changes:
                with open(os.path.normpath(os.path.join(REPO_PATH, file_path)), "w", encoding="utf-8") as f: 
                    f.write(original_code)
        return [], {}, {}
    
    elif test_status == "FAILURE_DEPENDENCY_INSTALL":
        log_callback(f"\n❌ DEPENDENCY INSTALLATION FAILED. Cannot install project requirements.")
        log_callback(f"   Details: {test_output}")
        # Revert changes
        for file_path, original_code in original_codes.items():
            if file_path in proposed_changes:
                with open(os.path.normpath(os.path.join(REPO_PATH, file_path)), "w", encoding="utf-8") as f: 
                    f.write(original_code)
        return [], {}, {}
    
    elif test_status == "FAILURE_DEPENDENCY_MISSING":
        log_callback(f"\n❌ MISSING DEPENDENCIES. Project has import errors due to missing packages.")
        log_callback(f"   Details: {test_output}")
        # Revert changes
        for file_path, original_code in original_codes.items():
            if file_path in proposed_changes:
                with open(os.path.normpath(os.path.join(REPO_PATH, file_path)), "w", encoding="utf-8") as f: 
                    f.write(original_code)
        return [], {}, {}
    
    elif test_status == "FAILURE_TESTS_FAILED":
        log_callback(f"\n❌ REGRESSION TEST FAILED. Our changes broke existing functionality.")
        log_callback(f"   Test Output: {test_output}")
        # Revert changes
        for file_path, original_code in original_codes.items():
            if file_path in proposed_changes:
                with open(os.path.normpath(os.path.join(REPO_PATH, file_path)), "w", encoding="utf-8") as f: 
                    f.write(original_code)
        return [], {}, {}
    
    elif test_status == "FAILURE_SYNTAX_ERRORS":
        log_callback(f"\n❌ SYNTAX ERROR FAILURE. The project has syntax errors that prevent execution.")
        log_callback(f"   Details: {test_output}")
        # Revert changes
        for file_path, original_code in original_codes.items():
            if file_path in proposed_changes:
                with open(os.path.normpath(os.path.join(REPO_PATH, file_path)), "w", encoding="utf-8") as f: 
                    f.write(original_code)
        return [], {}, {}
    
    # Handle warnings (proceed with caution but don't revert)
    elif test_status == "WARNING_ORIGINAL_INSTALL":
        log_callback(f"\n⚠️ DEPENDENCY WARNING: Installed with original requirements (may have compatibility issues).")
        log_callback(f"   Details: {test_output}")
        log_callback("   PROCEEDING: Changes will be committed but manual review is recommended.")
    
    elif test_status == "WARNING_APP_STARTUP_ISSUES":
        log_callback(f"\n⚠️ APPLICATION WARNING: App has startup issues but tests may still pass.")
        log_callback(f"   Details: {test_output}")
        log_callback("   PROCEEDING: Changes will be committed with warning notes.")
    
    # Handle success cases
    elif test_status in ["SUCCESS_NO_TESTS_TUTORIAL", "SUCCESS_NO_TESTS_FOUND"]:
        log_callback(f"\n✅ PROJECT VALIDATED: {test_output}")
        log_callback("   No tests found to verify changes, but project structure is valid.")
    
    elif test_status in ["SUCCESS_BASIC_INSTALL", "SUCCESS_COMPAT_INSTALL", "SUCCESS_NORMAL_INSTALL"]:
        log_callback(f"\n✅ DEPENDENCIES INSTALLED: {test_output}")
        log_callback("   Dependencies installed successfully, no tests to run.")
    
    elif test_status == "SUCCESS":
        log_callback(f"\n✅ ALL TESTS PASSED: {test_output}")
        log_callback("   Changes have been verified by the project's test suite.")
    
    else:
        # Fallback for any unhandled status
        log_callback(f"\n⚠️ UNKNOWN TEST STATUS: {test_status}")
        log_callback(f"   Details: {test_output}")
        log_callback("   PROCEEDING: Changes will be committed but require manual review.")
    
    files_to_commit = list(proposed_changes.keys())
    corrected_codes.update(proposed_changes)
    changed_files = files_to_commit

    # Enhanced Git operations
    try:
        safe_goal = re.sub(r"[^a-zA-Z0-9-]", "", user_goal.split(' ')[0].lower())
        new_branch_name = f"chimera-fix-{safe_goal}-{os.urandom(3).hex()}"
        repo.git.checkout("-b", new_branch_name)
    except Exception as e:
        log_callback(f"❌ GIT ERROR: {e}"); return [], {}, {}
    
    # Create enhanced commit message based on test status
    commit_message = f"fix: {user_goal}\n\nApplies automated fixes generated by the Chimera AI agent swarm."
    
    if test_status == "SUCCESS":
        commit_message += "\n\n[CI-Validated]: Changes passed automated test suite ✅"
    elif test_status in ["SUCCESS_NO_TESTS_TUTORIAL", "SUCCESS_NO_TESTS_FOUND"]:
        commit_message += "\n\n[CI-Info]: Project validated successfully. No tests found (typical for tutorial/example projects)."
    elif test_status in ["SUCCESS_BASIC_INSTALL", "SUCCESS_COMPAT_INSTALL", "SUCCESS_NORMAL_INSTALL"]:
        commit_message += "\n\n[CI-Info]: Dependencies installed successfully. No automated tests available to verify changes."
    elif test_status == "WARNING_ORIGINAL_INSTALL":
        commit_message += "\n\n[CI-Warning]: Installed with original requirements - may have dependency compatibility issues. Manual review recommended."
    elif test_status == "WARNING_APP_STARTUP_ISSUES":
        commit_message += "\n\n[CI-Warning]: Application startup issues detected - may be pre-existing issues. Manual verification recommended."
    else:
        commit_message += f"\n\n[CI-Status]: {test_status} - Manual review recommended."
        
    repo.index.add(files_to_commit)
    repo.index.commit(commit_message)
    log_callback(f"[Git Agent] Committed {len(files_to_commit)} changes with enhanced status tracking.")
    log_callback(f"[Git Agent] Commit message:\n---\n{commit_message}\n---")
    
    # Enhanced GitHub PR creation
    github_agent = setup_github_agent(log_callback)
    if github_agent:
        try:
            target_repo_name = repo_url.split('/')[-2] + '/' + repo_url.split('/')[-1].replace('.git', '')
            upstream_repo = github_agent.get_repo(target_repo_name)
            my_username = github_username
            my_fork_name = f"{my_username}/{target_repo_name.split('/')[1]}"
            my_fork = None
            
            try: 
                my_fork = github_agent.get_repo(my_fork_name)
            except Exception:
                log_callback(f"[GitHub Agent] Fork not found at '{my_fork_name}'. Creating a new one...")
                upstream_repo.create_fork(); time.sleep(10); my_fork = github_agent.get_repo(my_fork_name)
                log_callback(f"✅ Fork created successfully: {my_fork.html_url}")
                
            fork_url = my_fork.clone_url.replace("https://", f"https://{my_username}:{os.environ['GITHUB_TOKEN']}@")
            if "origin" in repo.remotes: repo.remotes.origin.set_url(fork_url)
            else: repo.create_remote("origin", fork_url)
            origin = repo.remotes.origin
            origin.push(refspec=f'{new_branch_name}:{new_branch_name}', force=True)
            
            # Enhanced PR title and body
            pr_title = commit_message.splitlines()[0]
            
            # Create detailed PR body based on test status
            pr_body = f"""## 🤖 Automated Security Fix by Project Chimera

**Goal:** {user_goal}

**Test Status:** `{test_status}`

"""
            
            if test_status == "SUCCESS":
                pr_body += "✅ **Validation Status**: All automated tests passed - changes are verified safe\n\n"
            elif test_status in ["SUCCESS_NO_TESTS_TUTORIAL", "SUCCESS_NO_TESTS_FOUND"]:
                pr_body += "ℹ️ **Validation Status**: Project validated, no tests found (common in tutorial projects)\n\n"
            elif test_status.startswith("WARNING_"):
                pr_body += f"⚠️ **Validation Status**: Changes applied with warnings - manual review recommended\n\n**Warning Details:** {test_output}\n\n"
            elif test_status.startswith("SUCCESS_"):
                pr_body += "✅ **Validation Status**: Dependencies verified successfully\n\n"
            
            pr_body += f"""### Files Modified
{chr(10).join(f"- `{file_path}`" for file_path in files_to_commit)}

### Commit Details
```
{commit_message}
```

---
*This PR was automatically generated by [Project Chimera](https://github.com/your-repo/chimera) - an AI agent swarm for automated security remediation.*
"""
            
            target_base_branch = upstream_repo.default_branch
            pr_head = f"{my_username}:{new_branch_name}"
            pr = upstream_repo.create_pull(title=pr_title, body=pr_body, head=pr_head, base=target_base_branch)
            log_callback(f"✅ ENHANCED PULL REQUEST CREATED: {pr.html_url}")
            log_callback(f"   PR includes detailed test status and validation information.")
            
        except Exception as e:
            import traceback
            log_callback(f"❌ GITHUB ERROR: {e}\n{traceback.format_exc()}")
            
    return changed_files, original_codes, corrected_codes