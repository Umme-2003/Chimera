# APP.PY - Streamlit UI for Project Chimera
# GitLab AI Hackathon 2026

import streamlit as st
import os
import shutil
import stat
import time

# Try GitLab engine first, fall back to standalone
try:
    from chimera_gitlab import run_chimera_orchestration_gitlab
    USE_GITLAB_ENGINE = True
except ImportError:
    from chimera import run_chimera_orchestration
    USE_GITLAB_ENGINE = False

# Optional modules
try:
    from chimera_security_debt import calculate_total_security_debt, format_security_debt_report, get_debt_severity_badge
    HAS_SECURITY_DEBT = True
except ImportError:
    HAS_SECURITY_DEBT = False

try:
    from chimera_confidence import rate_fix_confidence, format_confidence_report, should_auto_merge
    HAS_CONFIDENCE = True
except ImportError:
    HAS_CONFIDENCE = False

# --- Page Configuration ---
st.set_page_config(page_title="Project Chimera", page_icon="🤖", layout="wide")

# --- Custom CSS for a polished look ---
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%); }
    .phase-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    .phase-active {
        border-color: #4fc3f7;
        background: rgba(79,195,247,0.08);
        box-shadow: 0 0 15px rgba(79,195,247,0.15);
    }
    .phase-done {
        border-color: #66bb6a;
        background: rgba(102,187,106,0.08);
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- UI Elements ---
st.title("🤖 Project Chimera")
st.caption("An autonomous AI agent swarm for codebase security analysis and remediation.")

# --- Sidebar ---
st.sidebar.header("⚙️ Configuration")

if USE_GITLAB_ENGINE:
    username = st.sidebar.text_input("GitLab Username", help="Your GitLab username for MR creation")
else:
    username = st.sidebar.text_input("GitHub Username", help="Your GitHub username for PR creation")

repo_url = st.sidebar.text_input(
    "Target Repository URL",
    value="https://github.com/sibtc/django-multiple-user-types-example.git"
)
user_goal = st.sidebar.text_input(
    "Security Scan Goal",
    value="Find and fix any hardcoded secrets and SQL injection vulnerabilities"
)

dry_run = st.sidebar.checkbox("🧪 Dry Run (no MR/PR created)", value=True)

run_button = st.sidebar.button("🚀 Run Chimera Analysis", use_container_width=True, type="primary")

st.sidebar.markdown("---")
st.sidebar.info(
    "**Pipeline Phases:**\n"
    "1. 🔍 **Discovery** — Clone & scan for suspicious files\n"
    "2. 🛡️ **Scanning** — AI-powered vulnerability detection\n"
    "3. 🔧 **Remediation** — Generate & apply fixes\n"
    "4. 🧪 **Testing** — Run project's real test suite\n"
    "5. 📊 **Report** — Create Issue + Merge Request"
)

engine_name = "GitLab Engine" if USE_GITLAB_ENGINE else "Standalone Engine"
st.sidebar.caption(f"Engine: {engine_name}")

# --- Main App Logic ---
if 'log_messages' not in st.session_state:
    st.session_state['log_messages'] = []
if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'current_phase' not in st.session_state:
    st.session_state['current_phase'] = 0

# Phase progress display
PHASES = [
    ("🔍", "Discovery", "Cloning repository and identifying files"),
    ("🛡️", "Scanning", "AI-powered vulnerability detection"),
    ("🔧", "Remediation", "Generating and applying security fixes"),
    ("🧪", "Testing", "Running test suite to validate fixes"),
    ("📊", "Report", "Creating Issue and Merge Request"),
]

def render_phase_bar(current_phase):
    """Render a visual phase progress bar."""
    cols = st.columns(len(PHASES))
    for i, (icon, name, desc) in enumerate(PHASES):
        with cols[i]:
            if i < current_phase:
                st.markdown(f"<div class='phase-card phase-done'>✅ {icon} <b>{name}</b></div>", unsafe_allow_html=True)
            elif i == current_phase:
                st.markdown(f"<div class='phase-card phase-active'>⏳ {icon} <b>{name}</b><br/><small>{desc}</small></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='phase-card'>⬜ {icon} <b>{name}</b></div>", unsafe_allow_html=True)


def robust_rmtree(path):
    """Safely remove directory tree, handling read-only files on Windows."""
    def remove_readonly(func, path, _):
        if not os.access(path, os.W_OK):
            os.chmod(path, stat.S_IWUSR)
            func(path)
        else:
            raise
    if os.path.exists(path):
        shutil.rmtree(path, onerror=remove_readonly)


# --- Activity Log ---
log_container = st.expander("📋 Agent Activity Log", expanded=True)
log_placeholder = log_container.empty()
results_container = st.container()

# --- Run Pipeline ---
if run_button and repo_url and user_goal and username:
    st.session_state.log_messages = []
    st.session_state.results = None
    st.session_state.current_phase = 0

    try:
        st.info("🧹 Preparing clean workspace...")
        robust_rmtree("temp_repo")
    except Exception as e:
        st.error(f"Fatal Error: Could not clean workspace. Error: {e}")
        st.stop()

    # Phase progress bar
    phase_bar = st.empty()

    with st.spinner("🤖 Chimera agent swarm is active... This may take several minutes."):
        def log_callback(message):
            """Enhanced log callback that detects phase transitions."""
            st.session_state.log_messages.append(message)
            log_text = "\n".join(st.session_state.log_messages)
            log_placeholder.code(log_text, language='text')

            # Auto-detect phase transitions from log messages
            msg_lower = message.lower()
            if "stage 1" in msg_lower or "discovery" in msg_lower or "cloning" in msg_lower:
                st.session_state.current_phase = 0
            elif "stage 2" in msg_lower or "scanning" in msg_lower or "keyword search" in msg_lower:
                st.session_state.current_phase = 1
            elif "stage 3" in msg_lower or "remediation" in msg_lower or "analyzing" in msg_lower:
                st.session_state.current_phase = 2
            elif "stage 4" in msg_lower or "test runner" in msg_lower or "testing" in msg_lower:
                st.session_state.current_phase = 3
            elif "stage 5" in msg_lower or "gitlab" in msg_lower or "report" in msg_lower:
                st.session_state.current_phase = 4

        if USE_GITLAB_ENGINE:
            changed_files, original_codes, corrected_codes, metrics_summary = run_chimera_orchestration_gitlab(
                repo_url=repo_url,
                user_goal=user_goal,
                gitlab_username=username,
                log_callback=log_callback,
                dry_run=dry_run
            )
            st.session_state.results = (changed_files, original_codes, corrected_codes, metrics_summary)
        else:
            changed_files, original_codes, corrected_codes = run_chimera_orchestration(
                repo_url=repo_url,
                user_goal=user_goal,
                github_username=username,
                log_callback=log_callback
            )
            st.session_state.results = (changed_files, original_codes, corrected_codes, "")

    st.success("✅ Analysis complete!")
    st.rerun()

# --- Display Results ---
log_text = "\n".join(st.session_state.log_messages)
log_placeholder.code(log_text, language='text')

if st.session_state.results:
    if USE_GITLAB_ENGINE:
        changed_files, original_codes, corrected_codes, metrics_summary = st.session_state.results
    else:
        changed_files, original_codes, corrected_codes, _ = st.session_state.results
        metrics_summary = ""

    if changed_files:
        # --- Metrics Dashboard ---
        if metrics_summary:
            with results_container.expander("📊 Execution Metrics", expanded=True):
                st.markdown(metrics_summary)

        # --- Security Debt Dashboard ---
        if HAS_SECURITY_DEBT:
            vuln_list = []
            for fp in changed_files:
                vuln_list.append({
                    'file': fp,
                    'code': corrected_codes.get(fp, ''),
                    'file_content': original_codes.get(fp, '')
                })
            debt_report = calculate_total_security_debt(vuln_list) if vuln_list else None

            if debt_report:
                severity_badge = get_debt_severity_badge(debt_report['total_debt'])
                results_container.success(
                    f"**Analysis Complete!** {len(changed_files)} file(s) fixed. {severity_badge}"
                )
                with results_container.expander("💰 Security Debt Analysis", expanded=True):
                    st.markdown(format_security_debt_report(debt_report))

        # --- Confidence Scoring ---
        if HAS_CONFIDENCE:
            confidence_reports = []
            for fp in changed_files:
                test_status = "SUCCESS"
                conf = rate_fix_confidence(
                    original_code=original_codes.get(fp, ''),
                    fixed_code=corrected_codes.get(fp, ''),
                    test_status=test_status,
                    test_output="",
                    vulnerability_type="hardcoded_secret",
                    file_extension=".py"
                )
                confidence_reports.append((fp, conf))

            with results_container.expander("🎯 Fix Confidence Scoring", expanded=True):
                for fp, conf in confidence_reports:
                    st.markdown(f"#### {fp}")
                    st.markdown(format_confidence_report(conf))
                    if should_auto_merge(conf):
                        st.success("✅ Confident enough for auto-merge")
                    else:
                        st.warning("⚠️ Review recommended before merge")
                    st.markdown("---")

        # --- Code Diff View ---
        results_container.subheader("📝 Code Changes")
        for file_path in changed_files:
            with results_container.expander(f"Changes for `{file_path}`", expanded=True):
                col1, col2 = st.columns(2)
                col1.text("Original")
                col1.code(original_codes.get(file_path, ""), language='python')
                col2.text("Corrected")
                col2.code(corrected_codes.get(file_path, ""), language='python')
    else:
        results_container.info(
            "Analysis complete. No files required changes or the changes failed regression testing."
        )