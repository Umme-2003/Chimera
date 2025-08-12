# APP.PY - THE STABLE, POLISHED, PORTFOLIO VERSION

import streamlit as st
import os
import shutil
import stat
from chimera import run_chimera_orchestration

# --- Page Configuration ---
st.set_page_config(page_title="Project Chimera", page_icon="🤖", layout="wide")

# --- UI Elements ---
st.title("🤖 Project Chimera")
st.caption("An autonomous AI agent swarm for codebase analysis and remediation.")

st.sidebar.header("Configuration")
github_username = st.sidebar.text_input("Your GitHub Username", help="The account where the fork will be created.")
repo_url = st.sidebar.text_input("Target GitHub Repository URL", value="https://github.com/mjhea0/flask-boilerplate.git")
user_goal = st.sidebar.text_input("High-Level Goal", value="Find and fix SQL injection vulnerabilities")
run_button = st.sidebar.button("🚀 Run Chimera Analysis", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.info(
    "**How it works:**\n"
    "1. **Discovery:** Finds suspicious files via keywords.\n"
    "2. **Remediation:** Hunter & Engineer agents fix vulnerabilities.\n"
    "3. **Review:** A Reviewer agent validates the AI-generated code for correct syntax.\n"
    "4. **Deployment:** Automatically creates a fork and a Pull Request with the validated changes."
)

# --- Main App Logic ---
if 'log_messages' not in st.session_state: st.session_state['log_messages'] = []
if 'results' not in st.session_state: st.session_state['results'] = None

log_container = st.expander("Agent Activity Log", expanded=True)
log_placeholder = log_container.empty()
results_container = st.container()

def robust_rmtree(path):
    def remove_readonly(func, path, _):
        if not os.access(path, os.W_OK): os.chmod(path, stat.S_IWUSR); func(path)
        else: raise
    if os.path.exists(path): shutil.rmtree(path, onerror=remove_readonly)

if run_button and repo_url and user_goal and github_username:
    st.session_state.log_messages = []
    st.session_state.results = None
    
    try:
        st.info("🧹 Preparing a clean workspace...")
        robust_rmtree("temp_repo")
        st.success("✅ Workspace is clean.")
    except Exception as e:
        st.error(f"Fatal Error: Could not clean workspace. Error: {e}"); st.stop()

    with st.spinner("🤖 Chimera swarm is active... This may take several minutes."):
        def log_callback(message):
            st.session_state.log_messages.append(message)
            # This is a bit of a hack to force UI updates inside the spinner
            log_text = "\n".join(st.session_state.log_messages)
            log_placeholder.code(log_text, language='text')

        changed_files, original_codes, corrected_codes = run_chimera_orchestration(
            repo_url=repo_url,
            user_goal=user_goal,
            github_username=github_username,
            log_callback=log_callback
        )
        st.session_state.results = (changed_files, original_codes, corrected_codes)
    
    st.success("Analysis complete!")
    st.rerun()

# Display logs and results
log_text = "\n".join(st.session_state.log_messages)
log_placeholder.code(log_text, language='text')

if st.session_state.results:
    changed_files, original_codes, corrected_codes = st.session_state.results
    if changed_files:
        results_container.success(f"**Analysis Complete!** {len(changed_files)} file(s) were fixed and a PR was created.")
        for file_path in changed_files:
            with results_container.expander(f"Changes for `{file_path}`", expanded=True):
                col1, col2 = st.columns(2); col1.text("Original"); col1.code(original_codes[file_path], language='python'); col2.text("Corrected"); col2.code(corrected_codes[file_path], language='python')
    else:
        results_container.info("Analysis complete. No files required changes.")