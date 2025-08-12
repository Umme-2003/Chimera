# CHIMERA.PY - THE STABLE, POLISHED, TIER-2 PORTFOLIO VERSION

import ast
import os
import shutil
import re
import stat
import time
import sys
from dotenv import load_dotenv
from git import Repo
from github import Github

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_core.output_parsers import StrOutputParser

# --- 1. CONFIGURATION ---
load_dotenv()
REPO_PATH = "temp_repo"
KNOWLEDGE_BASE_FILE = "knowledge_base.txt"
llm = ChatGroq(model="llama3-8b-8192", temperature=0.1)
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
                except Exception: pass
    return list(matching_files)

def inject_test_file(file_path, content, log_callback):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f: f.write(content)
        log_callback(f"[Orchestrator] Test file injected at: {os.path.relpath(file_path, REPO_PATH)}")
    except Exception as e: log_callback(f"❌ ERROR: Could not inject test file. {e}")

# --- 3. AGENT DEFINITIONS ---
def setup_remediation_agents(log_callback):
    log_callback("[Remediation & Review Agents] Initializing...")
    # Setup knowledge base retriever
    loader = GenericLoader.from_filesystem(path="", glob=KNOWLEDGE_BASE_FILE, parser=LanguageParser())
    docs = loader.load(); text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50); documents = text_splitter.split_documents(docs); vector = FAISS.from_documents(documents, embeddings); retriever = vector.as_retriever()
    
    # Hunter Agent
    prompt_hunter = ChatPromptTemplate.from_template("Cybersecurity analyst...CONTEXT: {context}\nSOURCE CODE:\n{input}\nANALYSIS:")
    chain_hunter = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt_hunter))
    
    # Engineer Agent
    prompt_engineer = ChatPromptTemplate.from_template("Code generation bot...ANALYSIS:\n{analysis}\nSOURCE CODE:\n{input}\nRESPONSE (RAW PYTHON CODE ONLY):")
    chain_engineer = prompt_engineer | llm | StrOutputParser()
    
    # Reviewer Agent (Syntax Check)
    def reviewer_pipeline(code: str) -> bool:
        try:
            ast.parse(code)
            log_callback("     ✅ REVIEW PASSED: Syntax is valid.")
            return True
        except Exception as e:
            log_callback(f"     ❌ REVIEW FAILED: Invalid syntax. {e}")
            return False

    def remediation_pipeline(inputs):
        analysis_result = chain_hunter.invoke({"input": inputs["input"]})
        analysis_text = analysis_result["answer"]
        if "no vulnerabilities were found" in analysis_text.lower():
            return None
        
        corrected_code = extract_python_code(chain_engineer.invoke({"input": inputs["input"], "analysis": analysis_text}))
        
        if reviewer_pipeline(corrected_code):
            return corrected_code
        else:
            return None # Return None if the generated code is invalid
            
    return remediation_pipeline

def setup_github_agent(log_callback):
    log_callback("[GitHub Agent] Initializing...")
    try: return Github(os.environ["GITHUB_TOKEN"])
    except Exception as e: log_callback(f"❌ GITHUB ERROR: {e}"); return None

# --- 4. THE MAIN ORCHESTRATOR ---
def run_chimera_orchestration(repo_url, user_goal, github_username, log_callback):
    
    changed_files, original_codes, corrected_codes = [], {}, {}
    
    log_callback(f"\n[Orchestrator] Cloning fresh repo: {repo_url}...")
    repo = Repo.clone_from(repo_url, to_path=REPO_PATH)
    if "sql" in user_goal.lower():
        inject_test_file(os.path.join(REPO_PATH, "app", "vulnerable_sql.py"), VULNERABLE_SQL_CODE, log_callback)

    log_callback("\n[Orchestrator] Starting keyword search...")
    keywords = []; 
    if "secret" in user_goal.lower(): keywords.extend(["SECRET_KEY", "API_KEY", "'SECRET'"])
    if "sql" in user_goal.lower(): keywords.extend(['f"', "SELECT ", ".execute("])
    identified_files = keyword_search_files(REPO_PATH, keywords)
    
    if not identified_files:
        log_callback("[Orchestrator] No suspicious files found."); return changed_files, original_codes, corrected_codes

    remediation_chain = setup_remediation_agents(log_callback)
    files_to_commit = []
    
    log_callback("\n[Orchestrator] Generating, fixing, and reviewing code...")
    for file_path in identified_files:
        full_path = os.path.normpath(os.path.join(REPO_PATH, file_path))
        log_callback(f"  -> Processing: {file_path}")
        try:
            with open(full_path, "r", encoding='utf-8') as f: original_code = f.read()
            corrected_code = remediation_chain({"input": original_code})
            
            if corrected_code and corrected_code.strip() != original_code.strip():
                log_callback("     - Change proposed and validated. Staging for commit.")
                with open(full_path, "w", encoding="utf-8") as f: f.write(corrected_code)
                files_to_commit.append(file_path)
                original_codes[file_path] = original_code
                corrected_codes[file_path] = corrected_code
            else:
                log_callback("     - No valid changes proposed.")
        except Exception as e: log_callback(f"     ❌ Error processing file: {e}")

    if not files_to_commit:
        log_callback("\n[Orchestrator] No valid changes were made."); return changed_files, original_codes, corrected_codes

    # --- COMMIT & PULL REQUEST ---
    try:
        safe_goal = re.sub(r"[^a-zA-Z0-9-]", "", user_goal.split(' ')[0].lower())
        new_branch_name = f"chimera-fix-{safe_goal}-{os.urandom(3).hex()}"
        repo.git.checkout("-b", new_branch_name)
    except Exception as e:
        log_callback(f"❌ GIT ERROR: {e}"); return [], {}, {}
    
    commit_message = f"fix: {user_goal}\n\nApplies automated fixes generated and validated by the Chimera AI agent swarm."
    repo.index.add(files_to_commit); repo.index.commit(commit_message)
    log_callback(f"[Git Agent] Committed {len(files_to_commit)} changes with message:\n---\n{commit_message}\n---")
    
    github_agent = setup_github_agent(log_callback)
    if github_agent:
        try:
            target_repo_name = repo_url.split('/')[-2] + '/' + repo_url.split('/')[-1].replace('.git', '')
            upstream_repo = github_agent.get_repo(target_repo_name)
            my_username = github_username
            my_fork_name = f"{my_username}/{target_repo_name.split('/')[1]}"
            my_fork = None
            try: my_fork = github_agent.get_repo(my_fork_name)
            except Exception:
                log_callback(f"[GitHub Agent] Fork not found at '{my_fork_name}'. Creating a new one...")
                upstream_repo.create_fork(); time.sleep(10); my_fork = github_agent.get_repo(my_fork_name)
                log_callback(f"✅ Fork created successfully: {my_fork.html_url}")
            
            fork_url = my_fork.clone_url.replace("https://", f"https://{my_username}:{os.environ['GITHUB_TOKEN']}@")
            if "origin" in repo.remotes: repo.remotes.origin.set_url(fork_url)
            else: repo.create_remote("origin", fork_url)
            
            origin = repo.remotes.origin
            origin.push(refspec=f'{new_branch_name}:{new_branch_name}', force=True)
            
            pr_title = commit_message.splitlines()[0]
            pr_body = f"Auto-generated by Project Chimera.\n**Goal:** {user_goal}\n**Commit:**\n```\n{commit_message}\n```\n\n*This change was automatically reviewed for syntax validity.*"
            target_base_branch = upstream_repo.default_branch
            pr_head = f"{my_username}:{new_branch_name}"
            
            pr = upstream_repo.create_pull(title=pr_title, body=pr_body, head=pr_head, base=target_base_branch)
            log_callback(f"✅ PULL REQUEST CREATED: {pr.html_url}")
        except Exception as e:
            import traceback
            log_callback(f"❌ GITHUB ERROR: {e}\n{traceback.format_exc()}")
            
    changed_files = files_to_commit
    return changed_files, original_codes, corrected_codes