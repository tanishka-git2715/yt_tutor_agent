"""
app.py — YouTube Channel AI Tutor (Streamlit UI)
-------------------------------------------------
Run with:
    streamlit run app.py
"""

import os
import json
import streamlit as st


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="YT Channel Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
.source-card {
    background: rgba(128,128,128,0.08);
    border-left: 3px solid #7F77DD;
    border-radius: 4px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 13px;
}
.source-card a {
    color: #7F77DD;
    text-decoration: none;
    font-weight: 500;
}
.source-card .score {
    color: #888;
    font-size: 11px;
    margin-left: 6px;
}
.stat-box {
    background: rgba(128,128,128,0.06);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
.stat-number { font-size: 24px; font-weight: 600; }
.stat-label { font-size: 12px; color: #888; }
.setup-step {
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
        "chat_history": [],       # [{role, content}]
        "sources_history": [],    # sources for each assistant turn
        "engine": None,
        "engine_ready": False,
        "api_key_set": False,
        "show_sources": True,
        "creator_name": "Alex Hormozi",
        "creator_style": "",
        "provider": "anthropic",  # anthropic or groq
        "groq_api_key": os.environ.get("GROQ_API_KEY", ""),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ---------------------------------------------------------------------------
# Lazy engine loader
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_engine(creator_name: str, creator_style: str):
    """Cache the engine so it loads once per session."""
    from rag_engine import YTTutorEngine
    engine = YTTutorEngine(creator_name=creator_name, creator_style=creator_style)
    return engine


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🎓 YT Tutor Setup")
    st.markdown("---")

    # Provider Selection
    st.subheader("1. AI Provider")
    provider = st.radio(
        "Select Provider",
        options=["anthropic", "groq"],
        index=0 if st.session_state.provider == "anthropic" else 1,
        horizontal=True
    )
    st.session_state.provider = provider

    # API Keys
    if provider == "anthropic":
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            placeholder="sk-ant-...",
        )
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
    else:
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=st.session_state.groq_api_key or os.environ.get("GROQ_API_KEY", ""),
            placeholder="gsk_...",
        )
        if api_key:
            st.session_state.groq_api_key = api_key
            os.environ["GROQ_API_KEY"] = api_key

    st.markdown("---")

    # Creator profile
    st.subheader("2. Creator profile")
    creator_name = st.text_input(
        "Creator name",
        value=st.session_state.creator_name or "",
        placeholder="e.g. Andrej Karpathy",
    )
    creator_style = st.text_area(
        "Teaching style (optional)",
        value=st.session_state.creator_style or "",
        placeholder="e.g. Builds intuition from first principles, uses analogies, talks through code step by step.",
        height=90,
    )

    if creator_name != st.session_state.creator_name:
        st.session_state.creator_name = creator_name
        st.session_state.engine = None
    if creator_style != st.session_state.creator_style:
        st.session_state.creator_style = creator_style
        st.session_state.engine = None

    st.markdown("---")

    # Load engine
    st.subheader("3. Knowledge base")
    meta_exists = os.path.exists("data/vectorstore_meta.json")
    index_exists = os.path.exists("data/search_index/tfidf_index.joblib")

    if not meta_exists or not index_exists:
        st.warning("No knowledge base found yet.")
        if st.button("🏗 Build Initial Index", use_container_width=True):
             from scripts.build_vectorstore import build_vectorstore
             with st.status("Building index...", expanded=True) as status:
                st.write("Processing transcripts...")
                build_vectorstore(progress_cb=lambda p, m: st.write(m))
                status.update(label="Index built!", state="complete", expanded=False)
                st.rerun()
    else:
        with open("data/vectorstore_meta.json", encoding="utf-8") as f:
            meta = json.load(f)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""<div class='stat-box'>
                <div class='stat-number'>{meta['total_videos']}</div>
                <div class='stat-label'>videos</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class='stat-box'>
                <div class='stat-number'>{meta['total_chunks']:,}</div>
                <div class='stat-label'>chunks</div></div>""", unsafe_allow_html=True)

        if st.button("🔌 Load / Reload Tutor", use_container_width=True, type="primary"):
            if not api_key:
                st.error("Enter your API key first.")
            elif not creator_name:
                st.error("Enter the creator's name.")
            else:
                with st.spinner("Loading..."):
                    try:
                        st.session_state.engine = load_engine(
                            creator_name, creator_style
                        )
                        st.session_state.engine_ready = True
                        st.success("Tutor ready!")
                    except Exception as e:
                        st.error(f"Failed to load: {e}")

        if st.button("🔄 Re-build Search Index", use_container_width=True):
            from scripts.build_vectorstore import build_vectorstore
            with st.status("Re-indexing...", expanded=True) as status:
                st.write("Loading transcripts...")
                build_vectorstore(progress_cb=lambda p, m: st.write(m))
                status.update(label="Index updated!", state="complete", expanded=False)
                st.session_state.engine = None # Force reload
                st.rerun()

    st.markdown("---")

    # Settings
    st.subheader("Settings")
    st.session_state.show_sources = st.toggle(
        "Show source videos", value=st.session_state.show_sources
    )

    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.sources_history = []
        st.rerun()

    # Video list
    if meta_exists:
        with open("data/vectorstore_meta.json", encoding="utf-8") as f:
            meta2 = json.load(f)
        with st.expander(f"📹 {meta2['total_videos']} indexed videos"):
            for v in meta2.get("videos", []):
                st.markdown(f"- [{v['title'][:50]}]({v['url']})")


# ---------------------------------------------------------------------------
# Main chat UI
# ---------------------------------------------------------------------------

engine_ready = st.session_state.engine_ready and st.session_state.engine is not None

if not engine_ready:
    # Welcome / setup screen
    st.title("🎓 YouTube Channel AI Tutor")
    st.markdown("Turn any YouTube channel into your personal AI tutor.")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='setup-step'>
        <b>Step 1 — Extract</b><br>
        Run <code>1_extract_transcripts.py</code> to download all transcripts from your chosen channel.
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='setup-step'>
        <b>Step 2 — Index</b><br>
        Run <code>2_build_vectorstore.py</code> to chunk and embed the transcripts into a local vector database.
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='setup-step'>
        <b>Step 3 — Chat</b><br>
        Fill in the sidebar (API key + creator name), click <b>Load Tutor</b>, and start asking questions.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
### Example questions you can ask
- *"Explain backpropagation the way you always do"*
- *"What's your advice on learning deep learning from scratch?"*
- *"Walk me through how transformers work"*
- *"What tools do you recommend for training neural nets?"*
    """)

else:
    creator = st.session_state.creator_name
    st.title(f"🎓 {creator} — AI Tutor")
    st.caption(f"Answers grounded in {st.session_state.engine.meta['total_videos']} videos · {st.session_state.engine.meta['total_chunks']:,} knowledge chunks")

    # Display chat history
    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🎓"):
            st.markdown(msg["content"])

            # Show sources for assistant messages
            if (
                msg["role"] == "assistant"
                and st.session_state.show_sources
                and i // 2 < len(st.session_state.sources_history)
            ):
                sources = st.session_state.sources_history[i // 2]
                if sources:
                    with st.expander(f"📚 {len(sources)} source excerpts used", expanded=False):
                        seen_titles = set()
                        for s in sources:
                            if s.title not in seen_titles:
                                seen_titles.add(s.title)
                                st.markdown(
                                    f"<div class='source-card'>"
                                    f"<a href='{s.url}' target='_blank'>{s.title}</a>"
                                    f"<span class='score'>relevance: {s.score:.0%}</span>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

    # Chat input
    if prompt := st.chat_input(f"Ask {creator} anything..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        # Stream the response
        with st.chat_message("assistant", avatar="🎓"):
            message_placeholder = st.empty()
            full_response = ""
            sources = []

            try:
                engine = st.session_state.engine
                # Build history without the current message (last item)
                history = st.session_state.chat_history[:-1]

                sources, generator = engine.ask_stream(
                    prompt, 
                    chat_history=history,
                    provider=st.session_state.provider,
                    api_key=api_key
                )

                for chunk in generator:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

            except Exception as e:
                full_response = f"⚠️ Error: {e}"
                message_placeholder.error(full_response)

        # Save assistant message and sources
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        st.session_state.sources_history.append(sources)

        # Show sources inline right after response
        if st.session_state.show_sources and sources:
            with st.expander(f"📚 {len(sources)} source excerpts used", expanded=False):
                seen_titles = set()
                for s in sources:
                    if s.title not in seen_titles:
                        seen_titles.add(s.title)
                        st.markdown(
                            f"<div class='source-card'>"
                            f"<a href='{s.url}' target='_blank'>{s.title}</a>"
                            f"<span class='score'>relevance: {s.score:.0%}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

        st.rerun()
