"""
Research Agent — Streamlit UI
==============================
Run:  streamlit run src/app.py
"""

from __future__ import annotations
import os
import streamlit as st
import urllib.request

from agent import run_research_pipeline, ask_question

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.paper-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-left: 4px solid #818cf8;
    padding: 1.1rem 1.3rem;
    border-radius: 8px;
    margin-bottom: 0.8rem;
    color: #e2e8f0;
}
.paper-title  { font-weight: 600; font-size: 0.95rem; color: #c7d2fe; }
.paper-meta   { font-size: 0.78rem; color: #94a3b8; margin-top: 4px; }
.answer-box   {
    background: #0f172a;
    border: 1px solid #334155;
    border-left: 4px solid #34d399;
    border-radius: 8px;
    padding: 1rem 1.4rem;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
    font-size: 0.93rem;
    line-height: 1.7;
}
.source-chip {
    display: inline-block;
    background: #1e293b;
    color: #818cf8;
    font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid #334155;
    margin: 3px 4px 3px 0;
    text-decoration: none;
}
.source-chip:hover { border-color: #818cf8; }
.step-badge {
    display: inline-block;
    background: #312e81;
    color: #a5b4fc;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    margin-right: 6px;
    font-family: 'IBM Plex Mono', monospace;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    def _ollama_reachable() -> bool:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        try:
            urllib.request.urlopen(f"{base_url}/api/tags", timeout=0.8)
            return True
        except Exception:
            return False

    env_backend = os.getenv("LLM_BACKEND", "").strip().lower()
    default_backend = env_backend if env_backend in {"ollama", "huggingface"} else ("ollama" if _ollama_reachable() else "huggingface")

    backend = st.selectbox(
        "LLM Backend",
        ["ollama", "huggingface"],
        index=0 if default_backend == "ollama" else 1,
        help="Ollama = local model, no API key. HuggingFace = free cloud API.",
    )

    if backend == "ollama":
        # Default to the first locally available Ollama model (when possible),
        # otherwise fall back to the first option in the list.
        import subprocess

        available_models = []
        try:
            out = subprocess.check_output(["ollama", "list"], text=True, stderr=subprocess.STDOUT)
            for line in out.splitlines():
                # Expected format: "NAME  ID  SIZE  MODIFIED"
                parts = line.split()
                if parts:
                    available_models.append(parts[0])
        except Exception:
            available_models = []

        # Normalize "phi3:latest" -> "phi3" for display/selection.
        normalized = {m.split(":")[0] for m in available_models}
        options = ["llama3", "mistral", "phi3", "gemma"]
        env_model = os.getenv("OLLAMA_MODEL", "").strip().split(":")[0]
        # Prefer env_model when it is one of our known options; if we can't read
        # `ollama list` (hosted env / CLI missing), this prevents defaulting to
        # llama3 by accident.
        default_model = env_model if env_model in options else next((m for m in options if m in normalized), options[0])

        model = st.selectbox("Ollama Model", options, index=options.index(default_model))
        st.info("Make sure `ollama serve` is running locally.")
    else:
        st.warning("Needs a free HuggingFace token in your `.env` file.")
        model = "mistralai/Mistral-7B-Instruct-v0.3"

    # Defaults kept small so first-time runs complete quickly.
    num_papers = st.slider("Papers to fetch", 2, 8, 2)
    max_pages = st.slider("PDF pages to read", 4, 15, 4)

    st.markdown("---")
    st.markdown("### 🛠️ Stack")
    st.markdown("""
- **LLM**: Ollama / HuggingFace (free)
- **Embeddings**: `all-MiniLM-L6-v2` (offline)
- **Vector DB**: FAISS
- **Papers**: ArXiv API
- **UI**: Streamlit
    """)
    st.markdown("---")
    st.caption("Built with 100% free & open-source tools.")


# ── Header ────────────────────────────────────────────────────────────────────

st.title("🔬 Agentic Research Assistant")
st.caption(
    "Enter a topic → agent searches ArXiv, reads papers, builds a knowledge base, "
    "then answers your questions. **No paid APIs required.**"
)

# Pipeline status
STATUS_STEPS = [
    "🔍 Initialising LLM & embeddings",
    "📡 Searching ArXiv",
    "📄 Fetching & reading PDFs",
    "🤖 Summarising papers",
    "🧠 Building vector index",
    "✅ Ready",
]

# ── Session State ─────────────────────────────────────────────────────────────

for key, default in [
    ("pipeline_result", None),
    ("chat_history", []),
    ("backend", backend),
    ("model", model),
    ("user_q", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Search Bar ────────────────────────────────────────────────────────────────

with st.form("search_form"):
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "Research topic",
            placeholder="e.g.  attention mechanisms in transformers",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("🔍 Search", use_container_width=True)

if submitted and query:
    st.session_state.pipeline_result = None
    st.session_state.chat_history = []

    progress_bar = st.progress(0)
    status_text = st.empty()
    step_counter = [0]

    def on_status(msg: str):
        # If the user closes/refreshes the browser while the pipeline is running,
        # Streamlit's websocket can be closed; UI updates would raise.
        # We treat those as non-fatal because the backend work may still complete.
        try:
            step_counter[0] = min(step_counter[0] + 1, len(STATUS_STEPS) - 1)
            pct = int((step_counter[0] / (len(STATUS_STEPS) - 1)) * 100)
            progress_bar.progress(pct)
            status_text.markdown(f"**{msg}**")
        except Exception:
            return

    import os

    os.environ["LLM_BACKEND"] = backend
    if backend == "ollama":
        os.environ["OLLAMA_MODEL"] = model

    result = run_research_pipeline(
        query,
        max_papers=num_papers,
        status_callback=on_status,
    )

    progress_bar.progress(100)

    if "error" in result:
        try:
            status_text.error(result["error"])
        except Exception:
            pass
    else:
        st.session_state.pipeline_result = result
        try:
            status_text.success(f"✅ Loaded {len(result['papers'])} papers. Ask questions below!")
        except Exception:
            pass

elif submitted and not query:
    st.warning("Please enter a research topic.")


# ── Results ───────────────────────────────────────────────────────────────────

if st.session_state.pipeline_result:
    result = st.session_state.pipeline_result
    papers = result["papers"]

    st.markdown(f"### 📚 {len(papers)} Papers Analysed")

    tab_papers, tab_qa = st.tabs(["Papers & Summaries", "Q&A Chat"])

    # ── Tab 1: Papers ─────────────────────────────────────────────────────────
    with tab_papers:
        for i, paper in enumerate(papers):
            with st.expander(f"**{i+1}. {paper['title']}**"):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown("**🤖 AI Summary**")
                    st.write(paper["ai_summary"])
                    st.markdown("**📝 Original Abstract**")
                    st.caption(paper["abstract"][:500] + "…")
                with c2:
                    st.markdown("**📌 Info**")
                    st.write(f"👥 {', '.join(paper['authors'])}")
                    st.write(f"📅 {paper['published']}")
                    st.markdown(f"[🔗 View on ArXiv]({paper['url']})")

    # ── Tab 2: Q&A ────────────────────────────────────────────────────────────
    with tab_qa:
        st.markdown("**💡 Suggested questions**")
        suggestions = [
            "What are the main contributions across these papers?",
            "What datasets or benchmarks were used?",
            "What limitations are mentioned?",
            "How do these papers compare to each other?",
            "What future work do the authors suggest?",
        ]
        cols = st.columns(3)
        for i, s in enumerate(suggestions):
            if cols[i % 3].button(s, key=f"sug_{i}"):
                # Persist the selected question for the next rerun (Ask button reruns too).
                st.session_state["user_q"] = s

        user_q = st.text_input(
            "Your question",
            value=st.session_state["user_q"],
            placeholder="Ask anything about the loaded papers…",
            key="user_q_input",
        )

        if st.button("💬 Ask", type="primary") and user_q:
            with st.spinner("Searching papers and generating answer…"):
                qa_result = ask_question(result["llm"], result["vector_store"], user_q)
            st.session_state.chat_history.append(
                {
                    "question": user_q,
                    "answer": qa_result["answer"],
                    "sources": qa_result["sources"],
                }
            )

        if st.session_state.chat_history:
            st.markdown("---")
            for item in reversed(st.session_state.chat_history):
                st.markdown(f"**🙋 {item['question']}**")
                st.markdown(
                    f'<div class="answer-box">{item["answer"]}</div>',
                    unsafe_allow_html=True,
                )
                chips = "".join(
                    (
                        f'<a class="source-chip" href="{s["url"]}" target="_blank">'
                        f'📄 {s["title"][:45]}…</a>'
                        if len(s["title"]) > 45
                        else f'<a class="source-chip" href="{s["url"]}" target="_blank">'
                        f'📄 {s["title"]}</a>'
                    )
                    for s in item["sources"]
                )
                if chips:
                    st.markdown(f"**Sources:** {chips}", unsafe_allow_html=True)
                st.markdown("")

else:
    st.markdown(
        """
    <div style='text-align:center;padding:3rem 1rem;color:#64748b;'>
      <div style='font-size:3.5rem'>🔬</div>
      <p style='font-size:1.15rem;font-weight:600;margin-top:.5rem;'>Enter a research topic above to get started</p>
      <p style='font-size:.9rem'>The agent will autonomously search, read, and learn from ArXiv papers.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
