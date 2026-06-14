import streamlit as st
import json
import time
from backend.heuristics import simulate_vulnerability_scan, simulate_remediation

# Configure Streamlit page layout
st.set_page_config(
    page_title="CodeElixir.AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for matching React UI ---
CUSTOM_CSS = """
<style>
/* Main Background and Glassmorphism */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to top right, rgba(251, 207, 232, 0.35), rgba(230, 244, 234, 0.45), rgba(204, 251, 241, 0.4));
    font-family: 'Inter', sans-serif;
    color: #1e293b; /* slate-800 */
}

/* Hide Streamlit Header */
[data-testid="stHeader"] {
    background-color: transparent;
}

/* Header Container */
.main-header {
    background: rgba(255, 255, 255, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(16px);
    border-radius: 1.5rem;
    padding: 1.25rem;
    box-shadow: 0 20px 25px -5px rgba(226, 232, 240, 0.3);
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
}

/* Header Text */
.header-title {
    font-size: 1.5rem;
    font-weight: 900;
    background: linear-gradient(to right, #e11d48, #4f46e5, #059669);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    padding: 0;
}
.header-subtitle {
    font-size: 0.75rem;
    color: #475569;
    margin-top: 0.25rem;
    font-weight: 500;
}

/* Cards & Containers */
.glass-panel {
    background: rgba(255, 255, 255, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(12px);
    border-radius: 1.5rem;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 20px 25px -5px rgba(226, 232, 240, 0.2);
}

/* Vulnerability Card */
.vuln-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 1rem;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    border-left: 4px solid #f43f5e; /* rose-500 */
}
.vuln-card.fixed {
    border-left-color: #10b981; /* emerald-500 */
}

.vuln-title {
    font-size: 1rem;
    font-weight: 700;
    color: #0f172a;
    display: flex;
    justify-content: space-between;
}
.vuln-desc {
    font-size: 0.875rem;
    color: #475569;
    margin-top: 0.5rem;
}

/* Code Snippet */
.code-snippet {
    background-color: #0f172a;
    color: #e2e8f0;
    padding: 1rem;
    border-radius: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    overflow-x: auto;
    margin-top: 0.5rem;
}

/* Buttons overriding */
div.stButton > button {
    background: linear-gradient(to right, #f43f5e, #4f46e5);
    color: white;
    border: none;
    border-radius: 0.75rem;
    padding: 0.5rem 1rem;
    font-weight: 700;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    transition: all 0.2s;
}
div.stButton > button:hover {
    opacity: 0.9;
    transform: scale(0.98);
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- State Management ---
if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = []
if "vulnerabilities" not in st.session_state:
    st.session_state.vulnerabilities = []
if "scan_state" not in st.session_state:
    st.session_state.scan_state = "idle"

# --- Main App ---

# Header Section
st.markdown("""
<div class="main-header">
    <div>
        <h1 class="header-title">🛡️ CodeElixir.AI <span style="font-size: 0.6rem; background: rgba(16, 185, 129, 0.1); color: #047857; border: 1px solid rgba(16, 185, 129, 0.2); padding: 0.15rem 0.5rem; border-radius: 9999px; vertical-align: middle; margin-left: 0.5rem;">JAVA AUDITOR v1.1</span></h1>
        <p class="header-subtitle">Execute deep AST-level data flow threat scanning and secure prompt remediation.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Stats row
active_count = len([v for v in st.session_state.vulnerabilities if v.get("status") not in ["Ignored", "Approved"]])
fixed_count = len([v for v in st.session_state.vulnerabilities if v.get("status") == "Approved"])

cols = st.columns(4)
with cols[0]:
    st.metric("Loaded Files", len(st.session_state.loaded_files))
with cols[1]:
    st.metric("Threats Active", active_count)
with cols[2]:
    st.metric("Severity Matrix", f"{len([v for v in st.session_state.vulnerabilities if v.get('severity') == 'High'])}H {len([v for v in st.session_state.vulnerabilities if v.get('severity') == 'Medium'])}M")
with cols[3]:
    st.metric("Auto Patches", fixed_count)

st.markdown("<hr style='border-color: rgba(255,255,255,0.5);'>", unsafe_allow_html=True)

# --- Layout: 2 Columns ---
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 📂 File & Project Selection")
    
    # Pre-canned upload or manual upload
    uploaded_files = st.file_uploader("Drag and Drop .java files here", accept_multiple_files=True, type=["java"])
    
    # Or enter custom code
    st.markdown("#### 💻 Web Playground")
    sandbox_name = st.text_input("Filename", "PaymentVerificationService.java")
    sandbox_code = st.text_area("Java Source Code", height=200, placeholder="Paste your custom vulnerable Class structures...")
    
    if st.button("Load Sandbox Into Explorer"):
        if sandbox_code.strip():
            st.session_state.loaded_files.append({
                "name": sandbox_name,
                "path": f"sandbox/{sandbox_name}",
                "content": sandbox_code
            })
            st.success(f"Loaded {sandbox_name}")

    if uploaded_files:
        for f in uploaded_files:
            # check if already exists
            if not any(lf["name"] == f.name for lf in st.session_state.loaded_files):
                st.session_state.loaded_files.append({
                    "name": f.name,
                    "path": f"uploads/{f.name}",
                    "content": f.getvalue().decode("utf-8")
                })
        st.success(f"Parsed {len(uploaded_files)} files.")

    # Action Trigger
    st.markdown("---")
    if len(st.session_state.loaded_files) > 0:
        if st.button("🚀 Analyze Java Project"):
            with st.spinner("Analyzing AST and tracing data parameters..."):
                time.sleep(1.5)  # Simulate progress
                results = simulate_vulnerability_scan(st.session_state.loaded_files)
                st.session_state.vulnerabilities = results
                st.session_state.scan_state = "completed"
    
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown(f"### 🔥 Scan Results Findings ({active_count})")
    
    if st.session_state.scan_state == "idle":
        st.info("Review files or select a preset project, and hit 'Analyze Java Project' to extract findings.")
    elif st.session_state.scan_state == "completed" and len(st.session_state.vulnerabilities) == 0:
        st.success("Clean Bill of Health! We could not identify any vulnerability parameters inside scanned Java files.")
    else:
        for i, vuln in enumerate(st.session_state.vulnerabilities):
            status = vuln.get("status", "Scanned")
            if status == "Ignored":
                continue
                
            is_fixed = status == "Approved"
            card_class = "vuln-card fixed" if is_fixed else "vuln-card"
            badge_color = "#f43f5e" if vuln.get("severity") == "High" else "#f59e0b"
            if is_fixed: badge_color = "#10b981"
            
            st.markdown(f"""
            <div class="{card_class}">
                <div class="vuln-title">
                    <span>{vuln.get('type')}</span>
                    <span style="font-size: 0.75rem; background: {badge_color}; color: white; padding: 0.1rem 0.4rem; border-radius: 0.5rem;">{vuln.get('severity')}</span>
                </div>
                <div style="font-size: 0.75rem; font-family: monospace; color: #64748b; margin-top: 0.25rem;">📄 {vuln.get('filePath')} (Line {vuln.get('lineNumber')})</div>
                <div class="vuln-desc">{vuln.get('description')}</div>
                <div class="code-snippet">{vuln.get('snippet')}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if not is_fixed:
                col_btn1, col_btn2, col_btn3 = st.columns(3)
                with col_btn1:
                    if st.button("✨ Auto-Fix", key=f"fix_{i}"):
                        # find file content
                        file_content = ""
                        for lf in st.session_state.loaded_files:
                            if lf["path"] == vuln.get("filePath") or lf["name"] == vuln.get("filePath"):
                                file_content = lf["content"]
                                break
                        
                        fix_res = simulate_remediation(vuln.get("filePath"), file_content, vuln)
                        vuln["remediatedSnippet"] = fix_res["remediatedSnippet"]
                        vuln["remediationExplanation"] = fix_res["remediationExplanation"]
                        vuln["fullRemediatedContent"] = fix_res["fullRemediatedContent"]
                        st.rerun()
                        
                with col_btn2:
                    if st.button("Ignore", key=f"ignore_{i}"):
                        vuln["status"] = "Ignored"
                        st.rerun()
            
            # Show diff if fixing
            if "remediatedSnippet" in vuln and not is_fixed:
                st.markdown("#### 💡 Suggested Remediation")
                st.info(vuln["remediationExplanation"])
                st.markdown(f'<div class="code-snippet" style="border: 1px solid #10b981;">{vuln["remediatedSnippet"]}</div>', unsafe_allow_html=True)
                
                if st.button("✅ Approve Fix", key=f"approve_{i}"):
                    vuln["status"] = "Approved"
                    # Update file content
                    for lf in st.session_state.loaded_files:
                        if lf["path"] == vuln.get("filePath") or lf["name"] == vuln.get("filePath"):
                            lf["content"] = vuln["fullRemediatedContent"]
                            break
                    st.rerun()
                
    st.markdown('</div>', unsafe_allow_html=True)
