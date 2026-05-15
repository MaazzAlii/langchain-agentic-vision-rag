"""CLI for Agentic RAG"""
import sys, argparse
from pathlib import Path

def ingest(pdf, db="db1"):
    if not Path(pdf).exists():
        print(f"❌ Not found: {pdf}"); sys.exit(1)
    from ingestion import ingest_pdf
    ingest_pdf(pdf, collection_name=f"rag_{db}")

def chat():
    from langgraph_agent import LangGraphRAG
    agent = LangGraphRAG()
    stats = agent.get_collection_stats()
    if stats["status"] == "empty":
        print("⚠️ No data. Run: python main.py ingest <pdf>"); return
    print(f"\n{'='*50}\n🤖 LangGraph RAG Chat | {stats['count']} chunks\n{'='*50}")
    print("Commands: reset | quit\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋"); break
        if not q: continue
        if q.lower() in ["quit","exit","q"]: print("👋"); break
        if q.lower() == "reset": agent.reset(); continue
        ans, tools = agent.chat(q)
        if tools:
            for t in tools:
                print(f"   🔧 {t['tool']}({t['args'].get('query','')})")
        print(f"Assistant: {ans}\n{'-'*50}")

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    i = sub.add_parser("ingest"); i.add_argument("pdf")
    i.add_argument("--db", default="db1", choices=["db1","db2","db3","db4"])
    i.add_argument("--chat", action="store_true")
    sub.add_parser("chat")
    args = p.parse_args()
    if args.cmd == "ingest":
        ingest(args.pdf, args.db)
        if args.chat: chat()
    elif args.cmd == "chat":
        chat()
    else:
        p.print_help()
        print("\nQuick start:\n  1. python main.py ingest doc.pdf\n  2. streamlit run app.py")

if __name__ == "__main__":
    main()
