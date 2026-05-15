"""
Agentic RAG — Streamlit UI (LangGraph version)
All imports inside functions to prevent startup crashes.
"""
import os
import streamlit as st

st.set_page_config(page_title="Agentic RAG", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.main{background:#0e1117}
.stChatMessage{border-radius:12px;margin-bottom:8px}
.tool-badge{display:inline-block;background:#1e3a5f;color:#64b5f6;
  border:1px solid #1565c0;border-radius:6px;padding:2px 10px;
  font-size:12px;margin:2px 4px;font-family:monospace}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────
for key, val in [("agent",None),("messages",[]),("last_pdf",""),("ingested",False)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Agentic RAG")
    st.caption("LangGraph · Vision · Tool-Calling")
    st.divider()

    # Backend
    st.subheader("🔧 LLM Backend")
    backend = st.radio("Choose backend:", ["mistral","ollama"])
    os.environ["LLM_BACKEND"] = backend
    try:
        import config as _cfg
        _cfg.LLM_BACKEND = backend
    except Exception:
        pass

    if backend == "mistral":
        try:
            from dotenv import load_dotenv
            load_dotenv("api.env")
        except Exception:
            pass
        saved_key = os.environ.get("MISTRAL_API_KEY","")
        st.markdown("**Mistral API Key**")
        st.caption("Free key at [console.mistral.ai](https://console.mistral.ai)")
        api_key = st.text_input("Key:", value=saved_key, type="password",
                                placeholder="Paste your Mistral key here",
                                label_visibility="collapsed")
        if api_key:
            os.environ["MISTRAL_API_KEY"] = api_key
            try:
                import config as _cfg
                _cfg.MISTRAL_API_KEY = api_key
            except Exception:
                pass
            st.success("✅ API key loaded")
        else:
            st.warning("⚠️ Enter your Mistral API key")
    else:
        try:
            import requests as _r
            _r.get("http://localhost:11434", timeout=2)
            st.success("✅ Ollama is running")
        except Exception:
            st.error("❌ Ollama not detected — run: ollama serve")

    st.divider()

    # DB selector
    st.subheader("🗄️ Database Selection")
    db_labels = {"db1":"DB1 — General","db2":"DB2 — Technical",
                 "db3":"DB3 — Research","db4":"DB4 — Custom"}
    selected_dbs = st.multiselect("Active databases:",
                                  ["db1","db2","db3","db4"], default=["db1"],
                                  format_func=lambda x: db_labels[x])

    # Target DB for ingestion
    target_db = st.selectbox("Ingest PDF into:",
                             ["db1","db2","db3","db4"],
                             format_func=lambda x: db_labels[x])

    st.divider()

    # PDF Upload
    st.subheader("📄 Document Ingestion")
    uploaded = st.file_uploader("Upload any PDF", type=["pdf"])

    if uploaded:
        import tempfile
        tmp = os.path.join(tempfile.gettempdir(), uploaded.name)
        with open(tmp,"wb") as f:
            f.write(uploaded.read())
        st.success(f"📁 {uploaded.name}")
        st.session_state["last_pdf"] = tmp

        if st.button("🚀 Ingest PDF", type="primary", use_container_width=True):
            prog = st.progress(0,"Starting...")
            try:
                from ingestion import pdf_to_images, process_batches, save_chunks, save_failed, store_chromadb
                prog.progress(10,"📸 Converting to images...")
                images = pdf_to_images(tmp)
                prog.progress(30,f"🔍 Analyzing {len(images)} pages...")
                chunks, failed = process_batches(images)
                prog.progress(75,"💾 Saving metadata...")
                save_chunks(chunks)
                save_failed(failed)
                prog.progress(90,"🗄️ Storing in ChromaDB...")
                store_chromadb(chunks, collection_name=f"rag_{target_db}")
                prog.progress(100,"✅ Done!")
                st.session_state["ingested"] = True
                good = len(chunks) - len(failed)
                st.success(f"✅ {good}/{len(chunks)} pages stored in {target_db}!")
                if failed:
                    st.warning(f"⚠️ {len(failed)} pages failed extraction")
                    for p in failed:
                        st.caption(f"Page {p['page_no']}: {p.get('reason','?')[:50]}")
            except Exception as e:
                st.error(f"❌ Ingestion failed: {e}")
                st.exception(e)

    st.divider()

    # Agent
    st.subheader("🤖 Agent")
    if st.button("Initialize LangGraph Agent", use_container_width=True):
        try:
            from langgraph_agent import LangGraphRAG
            st.session_state["agent"] = LangGraphRAG()
            if selected_dbs:
                st.session_state["agent"].set_databases(selected_dbs)
            stats = st.session_state["agent"].get_collection_stats()
            st.success(f"✅ Agent ready! {stats['count']} chunks in [{stats['collection']}]")
        except Exception as e:
            st.error(f"❌ {e}")
            st.exception(e)

    if st.session_state["agent"] and selected_dbs:
        if hasattr(st.session_state["agent"],"set_databases"):
            st.session_state["agent"].set_databases(selected_dbs)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["messages"] = []
        if st.session_state["agent"]:
            st.session_state["agent"].reset()
        st.rerun()

    # Stats
    if st.session_state["agent"]:
        try:
            stats = st.session_state["agent"].get_collection_stats()
            st.markdown(f"""
<div style='background:#1a1a2e;border:1px solid #21262d;border-radius:8px;padding:12px;margin-top:8px;font-size:13px'>
📊 <b>ChromaDB Status</b><br>
Collections: <code>{stats['collection'] or 'none'}</code><br>
Total chunks: <b>{stats['count']}</b><br>
Status: {'🟢 Ready' if stats['status']=='ready' else '🔴 Empty'}
</div>""", unsafe_allow_html=True)
        except Exception:
            pass

# ── Main area ─────────────────────────────────────────────────────
st.title("🤖 Agentic RAG — LangGraph Chat")

tab_chat, tab_chunks, tab_how = st.tabs(["💬 Chat","📋 Chunks","📖 How It Works"])

with tab_chat:
    if not st.session_state["agent"]:
        st.warning("⚠️ Initialize the agent from the sidebar first.")
    else:
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("tools_used"):
                    badges = "".join(
                        f'<span class="tool-badge">🔍 {t["tool"]}: {t["args"].get("query","")[:30]}</span>'
                        for t in msg["tools_used"]
                    )
                    st.markdown(f'<div style="margin-top:6px">{badges}</div>',
                                unsafe_allow_html=True)

        user_input = st.chat_input("Ask anything about your document...")
        if user_input:
            st.session_state["messages"].append({"role":"user","content":user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("🤔 Agent thinking..."):
                    try:
                        answer, tools = st.session_state["agent"].chat(user_input)
                    except Exception as e:
                        answer = f"❌ Error: {e}"
                        tools  = []
                st.markdown(answer)
                if tools:
                    badges = "".join(
                        f'<span class="tool-badge">🔍 {t["tool"]}</span>'
                        for t in tools
                    )
                    st.markdown(f'<div style="margin-top:8px;font-size:12px">Tools: {badges}</div>',
                                unsafe_allow_html=True)
            st.session_state["messages"].append(
                {"role":"assistant","content":answer,"tools_used":tools})

with tab_chunks:
    st.subheader("📋 Ingested Chunks")
    import json as _json
    from pathlib import Path as _P
    if _P("chunks.json").exists():
        with open("chunks.json","r",encoding="utf-8") as _f:
            _chunks = _json.load(_f)
        st.info(f"Total: **{len(_chunks)}** chunks")
        _hf = st.text_input("🔎 Filter by heading")
        _pf = st.number_input("Filter by page (0=all)", min_value=0)
        _fc = _chunks
        if _hf:
            _fc = [c for c in _fc if _hf.lower() in c["heading"].lower()]
        if _pf:
            _fc = [c for c in _fc if c["page_no"] == _pf]
        for c in _fc:
            status = "❌" if c.get("extraction_failed") else "✅"
            with st.expander(f"{status} Chunk {c['chunk_no']} | Page {c['page_no']} — {c['heading']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Batch:** {c['batch_no']} | **Type:** `{c['page_type']}`")
                    st.write(f"**Keywords:** {', '.join(c.get('keywords',[]))}")
                with col2:
                    st.write(f"**Summary:** {c['summary']}")
                st.markdown(c["content"][:800] + "..." if len(c["content"])>800 else c["content"])
    else:
        st.info("No chunks yet. Upload and ingest a PDF first.")

with tab_how:
    st.markdown("""
## 🏗️ LangGraph Architecture

```
START → [router_node] → [db1_node] → [db2_node] → [db3_node] → [db4_node] → [history_node] → [generate_node] → END
```

### Nodes:
| Node | Role |
|------|------|
| **router_node** | Decides: off-topic? which DBs? need history? |
| **db1–db4 nodes** | Each searches one ChromaDB collection |
| **history_node** | Returns last 6 messages if needed |
| **generate_node** | Combines all results → final answer with citations |

### AgentState flows through all nodes:
```json
{
  "user_query": "What is RSA?",
  "selected_dbs": ["db1"],
  "is_off_topic": false,
  "db1_results": "Page 5: RSA is...",
  "final_answer": "RSA is... (Page 5)"
}
```

### Why LangGraph vs Custom Agent?
- **Custom agent**: manual loop, single DB, no proper state
- **LangGraph**: StateGraph, 4 DBs, dynamic routing, industry standard
""")
