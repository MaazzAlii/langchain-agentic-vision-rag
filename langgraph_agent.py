"""
LangGraph Agentic RAG
=====================
Graph: START → router → db1 → db2 → db3 → db4 → history → generate → END

Each node reads from AgentState and writes back to it.
Router decides: which DBs to search, is it off-topic, need history?
"""
import json
import os
from typing import TypedDict, List, Optional

# ── State ────────────────────────────────────────────────────────
class AgentState(TypedDict):
    user_query:       str
    selected_dbs:     List[str]
    is_off_topic:     bool
    needs_history:    bool
    db1_results:      str
    db2_results:      str
    db3_results:      str
    db4_results:      str
    history_context:  str
    final_answer:     str
    chat_history:     List[dict]
    error:            Optional[str]
    _expanded_query:  str          # Query expanded for better semantic search

# ── Database Manager ─────────────────────────────────────────────
class DBManager:
    COLLECTIONS = {
        "db1": "rag_db1",
        "db2": "rag_db2",
        "db3": "rag_db3",
        "db4": "rag_db4",
    }
    LABELS = {
        "db1": "General Knowledge",
        "db2": "Technical Docs",
        "db3": "Research Papers",
        "db4": "Custom Upload",
    }

    def __init__(self):
        import chromadb
        from config import CHROMA_PATH
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.cols = {}
        for key, name in self.COLLECTIONS.items():
            try:
                col = self.client.get_collection(name)
                if col.count() > 0:
                    self.cols[key] = col
                    print(f"✅ {key} ({name}): {col.count()} chunks")
            except Exception:
                pass
        if not self.cols:
            print("⚠️  No databases loaded. Ingest a PDF first.")

    def available(self):
        return list(self.cols.keys())

    def max_page(self, key):
        col = self.cols.get(key)
        if not col:
            return 0
        try:
            metas = col.get(include=["metadatas"])["metadatas"]
            return max(m["page_no"] for m in metas) if metas else 0
        except Exception:
            return 0

    def search(self, key, query_emb, n=3):
        col = self.cols.get(key)
        if not col:
            return ""
        try:
            mp  = self.max_page(key)
            res = col.query(query_embeddings=[query_emb],
                            n_results=min(n, col.count()),
                            include=["documents","metadatas","distances"])
            parts = []
            for doc, meta, dist in zip(res["documents"][0],
                                       res["metadatas"][0],
                                       res["distances"][0]):
                rel  = round(1 - dist, 2)
                pg   = meta["page_no"]
                parts.append(
                    f"SOURCE: {key.upper()} | Page {pg} of {mp} | {meta.get('heading','')}\n"
                    f"Relevance: {rel}\n{doc[:700]}\n"
                    f"(Cite as: Page {pg}. Document has {mp} pages total.)"
                )
            return "\n\n".join(parts)
        except Exception as e:
            return f"[Search error in {key}: {str(e)[:60]}]"

    def ingest_to(self, chunks, key="db1"):
        import chromadb
        from config import CHROMA_PATH
        name = self.COLLECTIONS.get(key, "rag_db1")
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            client.delete_collection(name)
        except Exception:
            pass
        col = client.create_collection(name)
        self.cols[key] = col
        return col

# ── LLM helpers ──────────────────────────────────────────────────
def _embed(text):
    from config import LLM_BACKEND, MISTRAL_EMBED_MODEL, OLLAMA_EMBED_MODEL
    if LLM_BACKEND == "mistral":
        from mistral_client import get_embedding
        return get_embedding(MISTRAL_EMBED_MODEL, text)
    else:
        import ollama
        return ollama.embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)["embedding"]

def _llm(messages):
    from config import LLM_BACKEND, MISTRAL_CHAT_MODEL, OLLAMA_CHAT_MODEL
    if LLM_BACKEND == "mistral":
        from mistral_client import chat_complete
        r = chat_complete(MISTRAL_CHAT_MODEL, messages)
        return r["choices"][0]["message"].get("content","")
    else:
        import ollama
        return ollama.chat(model=OLLAMA_CHAT_MODEL, messages=messages)["message"]["content"]

# ── Nodes ────────────────────────────────────────────────────────
_db_manager = None

def _get_db():
    global _db_manager
    if _db_manager is None:
        _db_manager = DBManager()
    return _db_manager

def _expand_query(query: str) -> str:
    """
    Expand/rephrase query for better semantic search.
    Maps common question patterns to document-friendly search terms.
    """
    q = query.lower().strip()
    expansions = {
        "technology stack":    "ReactJS Flutter FastAPI Firebase programming languages frameworks",
        "tech stack":          "ReactJS Flutter FastAPI Firebase programming languages frameworks",
        "tools used":          "ReactJS Flutter FastAPI Firebase programming languages frameworks",
        "technologies":        "ReactJS Flutter FastAPI Firebase programming languages frameworks",
        "what was used":       "technology framework programming language implementation",
        "how was it built":    "technology framework programming language implementation",
        "programming language":"ReactJS Flutter FastAPI Firebase programming languages frameworks",
        "supervisor":          "supervisor guided supervised project mentor teacher",
        "team":                "team members group students developed built",
        "architecture":        "system design architecture modular monolith structure components",
        "database":            "database SQLite Firebase storage data schema",
        "recommendation":      "recommendation engine counter scoring personalized suggestions",
        "authentication":      "JWT authentication login registration security token",
        "payment":             "payment Stripe XPay transaction processing",
        "mobile app":          "Flutter mobile application Android iOS cross-platform",
        "web app":             "ReactJS web frontend user interface browser",
    }
    for key, expansion in expansions.items():
        if key in q:
            print(f"   🔄 Query expanded: '{query}' → '{expansion}'")
            return expansion
    return query


def router_node(state: AgentState) -> AgentState:
    query     = state["user_query"]
    available = _get_db().available()
    print(f"   🗺️  Router — available DBs: {available}")

    off_kw = ["prime minister","president","ronaldo","cricket","movie","song",
              "recipe","love","celebrity","become","meet","weather","sports"]
    is_off = any(k in query.lower() for k in off_kw)
    needs_hist = any(w in query.lower() for w in ["earlier","before","you said","previous","last time"])

    # Expand query for better retrieval
    expanded = _expand_query(query)

    if not is_off and available:
        sel = available[:2] if len(available) >= 2 else available[:1]
    else:
        sel = available[:1] if available else ["db1"]

    print(f"   🗺️  Decision: off_topic={is_off}, dbs={sel}, history={needs_hist}")
    # Store expanded query in state for DB nodes to use
    return {**state, "is_off_topic":is_off, "selected_dbs":sel,
            "needs_history":needs_hist, "user_query": query,
            "_expanded_query": expanded,
            "db1_results":"","db2_results":"","db3_results":"","db4_results":"",
            "history_context":""}

def _db_node(state, key):
    if key not in state["selected_dbs"] or state["is_off_topic"]:
        return state
    print(f"   🔍 Searching {key} ({DBManager.LABELS.get(key,'')})...")
    try:
        # Use expanded query for embedding if available
        search_query = state.get("_expanded_query", "") or state["user_query"]
        emb = _embed(search_query)
        # Get top 5 results for better coverage
        res = _get_db().search(key, emb, n=5)
        return {**state, f"{key}_results": res}
    except Exception as e:
        return {**state, f"{key}_results": f"[Error: {e}]"}

def db1_node(state): return _db_node(state, "db1")
def db2_node(state): return _db_node(state, "db2")
def db3_node(state): return _db_node(state, "db3")
def db4_node(state): return _db_node(state, "db4")

def history_node(state: AgentState) -> AgentState:
    if not state["needs_history"]:
        return state
    hist = state.get("chat_history",[])
    if not hist:
        return {**state, "history_context":"No previous conversation."}
    lines = []
    for m in hist[-6:]:
        role = "USER" if m["role"]=="user" else "ASSISTANT"
        lines.append(f"[{role}]: {m['content'][:300]}")
    return {**state, "history_context":"\n\n".join(lines)}

def generate_node(state: AgentState) -> AgentState:
    if state["is_off_topic"]:
        ans = "I can only answer questions about the uploaded document(s). Your question is outside the document scope."
        new_hist = state.get("chat_history",[]) + [
            {"role":"user","content":state["user_query"]},
            {"role":"assistant","content":ans}
        ]
        return {**state, "final_answer":ans, "chat_history":new_hist[-20:]}

    ctx_parts = []
    for key in ["db1","db2","db3","db4"]:
        r = state.get(f"{key}_results","")
        # Only skip if empty or is an error message (starts with [Error or [Search error)
        if r and not r.startswith("[Error") and not r.startswith("[Search error"):
            ctx_parts.append(f"=== {key.upper()} ===\n{r}")
    if not ctx_parts:
        ans = "This information was not found in any of the loaded documents."
        new_hist = state.get("chat_history",[]) + [
            {"role":"user","content":state["user_query"]},
            {"role":"assistant","content":ans}
        ]
        return {**state, "final_answer":ans, "chat_history":new_hist[-20:]}

    ctx  = "\n\n".join(ctx_parts)
    hist = state.get("history_context","")
    sys  = """You are a helpful document Q&A assistant. You have been given text chunks extracted from a PDF document.

YOUR JOB: Answer the user's question using the document context provided below.

RULES:
- Read the context carefully and answer based on what is written there
- Cite page numbers like (Page 5) when you use information from a specific page
- If the context contains relevant information, USE IT to answer — do not say "not found"
- Only say "This information is not in the document" if the context truly has NO relevant information
- Be helpful, clear and concise
- You may summarize and explain information from the context in your own words"""

    usr  = (f"DOCUMENT CONTEXT (extracted from PDF):\n\n{ctx}\n\n"
            + (f"PREVIOUS CONVERSATION:\n{hist}\n\n" if hist else "")
            + f"USER QUESTION: {state['user_query']}\n\n"
            + "Please answer the question based on the document context above:")
    print("   ✍️  Generating answer...")
    ans = _llm([{"role":"system","content":sys},{"role":"user","content":usr}])
    new_hist = state.get("chat_history",[]) + [
        {"role":"user","content":state["user_query"]},
        {"role":"assistant","content":ans}
    ]
    return {**state, "final_answer":ans, "chat_history":new_hist[-20:]}

# ── Build graph ──────────────────────────────────────────────────
def build_graph():
    try:
        from langgraph.graph import StateGraph, END
        g = StateGraph(AgentState)
        for name, fn in [("router",router_node),("db1",db1_node),("db2",db2_node),
                          ("db3",db3_node),("db4",db4_node),("history",history_node),
                          ("generate",generate_node)]:
            g.add_node(name, fn)
        g.set_entry_point("router")
        g.add_edge("router","db1")
        g.add_edge("db1","db2")
        g.add_edge("db2","db3")
        g.add_edge("db3","db4")
        g.add_edge("db4","history")
        g.add_edge("history","generate")
        g.add_edge("generate",END)
        compiled = g.compile()
        print("✅ LangGraph compiled")
        return compiled
    except ImportError:
        print("⚠️  LangGraph not installed — using fallback")
        return None

# ── Public class ─────────────────────────────────────────────────
class LangGraphRAG:
    def __init__(self):
        self.graph        = build_graph()
        self.chat_history = []
        self.selected_dbs = _get_db().available()[:1] or ["db1"]
        print(f"🤖 LangGraph RAG ready | DBs: {_get_db().available()}")

    def set_databases(self, dbs):
        avail = _get_db().available()
        self.selected_dbs = [d for d in dbs if d in avail] or avail[:1]
        print(f"🗄️  Active DBs: {self.selected_dbs}")

    def chat(self, message):
        print(f"\n{'─'*45}\n📨 {message[:60]}")
        state: AgentState = {
            "user_query":message, "selected_dbs":self.selected_dbs,
            "is_off_topic":False, "needs_history":False,
            "db1_results":"","db2_results":"","db3_results":"","db4_results":"",
            "history_context":"","final_answer":"",
            "chat_history":self.chat_history.copy(),"error":None,
            "_expanded_query": "",
        }
        if self.graph:
            final = self.graph.invoke(state)
        else:
            for fn in [router_node,db1_node,db2_node,db3_node,db4_node,history_node,generate_node]:
                state = fn(state)
            final = state
        answer = final.get("final_answer","Could not generate answer.")
        self.chat_history = final.get("chat_history", self.chat_history)
        tools_used = []
        for db in final.get("selected_dbs",[]):
            if final.get(f"{db}_results","").strip():
                tools_used.append({"tool":f"search_{db}","args":{"query":message[:40]}})
        if final.get("needs_history"):
            tools_used.append({"tool":"check_chat_history","args":{"question":message[:40]}})
        return answer, tools_used

    def reset(self):
        self.chat_history = []
        print("🔄 History cleared.")

    def get_collection_stats(self):
        # Refresh DB manager to pick up newly ingested collections
        global _db_manager
        _db_manager = DBManager()
        avail = _get_db().available()
        total = 0
        for k in avail:
            try:
                total += _get_db().cols[k].count()
            except Exception:
                pass  # Collection was replaced — ignore stale reference
        return {
            "status":"ready" if avail else "empty",
            "count":total,
            "collection":", ".join(avail),
            "path":"./chroma_db",
            "available_dbs":avail,
            "db_labels":DBManager.LABELS,
        }