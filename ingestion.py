"""
Ingestion Pipeline — PDF → Images → Vision LLM → ChromaDB
With retry mechanism and failed pages logging.
"""
import os, io, json, base64, time
from pathlib import Path

# ── Config (imported lazily to avoid circular issues) ────────────
def _cfg():
    from config import (
        LLM_BACKEND, BATCH_SIZE, PDF_DPI, OUTPUT_JSON,
        CHROMA_PATH, CHROMA_COLLECTION,
        OLLAMA_VISION_MODEL, OLLAMA_EMBED_MODEL,
        MISTRAL_API_KEY, MISTRAL_VISION_MODEL, MISTRAL_EMBED_MODEL,
        FAILED_PAGES_LOG, MAX_RETRIES, RETRY_DELAY
    )
    return locals()

VISION_PROMPT = """You are an expert document analyzer. Carefully examine this page image and extract ALL visible content.

This page may contain: text, tables, diagrams, handwritten notes, mathematical equations, use case diagrams, ER diagrams, flowcharts, images, figures, or any combination.

Return ONLY a valid JSON object. No markdown fences, no extra text before or after. Use this exact structure:
{
  "heading": "<main heading on this page, or Section number, or No Heading if none>",
  "sub_headings": ["<any sub-headings found>"],
  "content_markdown": "<ALL text content you can see on this page. For tables: use markdown table format. For diagrams/figures: describe what you see in detail. For math: write equations in plain text. For handwriting: transcribe it. NEVER leave this empty - always write something>",
  "summary": "<1-2 sentences describing what this page is about>",
  "keywords": ["<3-5 important keywords from this page>"],
  "page_type": "<choose one: introduction|content|conclusion|table|figure|diagram|requirements|chapter|references|other>"
}

CRITICAL RULES:
- content_markdown MUST NOT be empty - describe anything you see
- For blank/nearly blank pages: write what little you see (page number, chapter title, etc.)
- For diagram pages: describe the diagram components and relationships in text
- For table pages: reproduce the table in markdown format
- Always return valid JSON"""

# ── Helpers ───────────────────────────────────────────────────────
def _to_b64(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def _parse_json(raw):
    """Robustly parse LLM JSON — handles fences, partial JSON, empty fields."""
    if not raw:
        return None
    raw = raw.strip()

    # Remove markdown fences like ```json ... ```
    if "```" in raw:
        lines = raw.split("\n")
        inner_lines = []
        in_fence = False
        for line in lines:
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            inner_lines.append(line)
        raw = "\n".join(inner_lines).strip()

    # Extract JSON object if surrounded by extra text
    if not raw.startswith("{"):
        start = raw.find("{")
        if start != -1:
            raw = raw[start:]

    # Find matching closing brace
    if raw.startswith("{"):
        depth, end_idx = 0, -1
        for i, ch in enumerate(raw):
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i; break
        if end_idx != -1:
            raw = raw[:end_idx+1]

    try:
        d = json.loads(raw)
        # If content_markdown is empty, build it from other fields
        if not d.get("content_markdown", "").strip():
            parts = []
            if d.get("heading"):
                parts.append("# " + d["heading"])
            for sh in d.get("sub_headings", []):
                parts.append("## " + sh)
            if d.get("summary"):
                parts.append(d["summary"])
            if parts:
                d["content_markdown"] = "\n\n".join(parts)
            else:
                return None
        return d
    except json.JSONDecodeError:
        # Last resort regex extraction
        import re
        cm = re.search(r'"content_markdown"\s*:\s*"(.*?)(?<!\\)"', raw, re.DOTALL)
        hd = re.search(r'"heading"\s*:\s*"(.*?)(?<!\\)"', raw)
        sm = re.search(r'"summary"\s*:\s*"(.*?)(?<!\\)"', raw)
        if cm or hd or sm:
            return {
                "heading": hd.group(1) if hd else "Unknown",
                "sub_headings": [],
                "content_markdown": cm.group(1) if cm else (sm.group(1) if sm else ""),
                "summary": sm.group(1) if sm else "",
                "keywords": [],
                "page_type": "content"
            }
        return None

# ── Vision LLM ───────────────────────────────────────────────────
def _vision_mistral(image):
    from mistral_client import vision_chat
    from config import MISTRAL_VISION_MODEL
    return _parse_json(vision_chat(MISTRAL_VISION_MODEL, _to_b64(image), VISION_PROMPT))

def _vision_ollama(image):
    import ollama
    from config import OLLAMA_VISION_MODEL
    resp = ollama.chat(model=OLLAMA_VISION_MODEL,
                       messages=[{"role":"user","content":VISION_PROMPT,"images":[_to_b64(image)]}])
    return _parse_json(resp["message"]["content"])

def _vision(image):
    from config import LLM_BACKEND
    return _vision_mistral(image) if LLM_BACKEND == "mistral" else _vision_ollama(image)

# ── Embedding ────────────────────────────────────────────────────
def _embed(text):
    from config import LLM_BACKEND, MISTRAL_EMBED_MODEL, OLLAMA_EMBED_MODEL
    if LLM_BACKEND == "mistral":
        from mistral_client import get_embedding
        return get_embedding(MISTRAL_EMBED_MODEL, text)
    else:
        import ollama
        return ollama.embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)["embedding"]

# ── Retry ────────────────────────────────────────────────────────
def _extract_with_retry(image, page_no):
    from config import MAX_RETRIES, RETRY_DELAY, LLM_BACKEND
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = _vision(image)
            if result is not None:
                if attempt > 1:
                    print(f"   ✅ Page {page_no} succeeded on attempt {attempt}")
                return result, True
            last_err = "Empty/invalid JSON from LLM"
        except Exception as e:
            last_err = str(e)
        print(f"   ⚠️  Page {page_no} attempt {attempt}/{MAX_RETRIES} — {str(last_err)[:60]}")
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
            if LLM_BACKEND == "mistral":
                time.sleep(1.2)
    print(f"   ❌ Page {page_no} FAILED after {MAX_RETRIES} attempts")
    return {
        "heading": f"Page {page_no} — Extraction Failed",
        "sub_headings": [], "content_markdown": "",
        "summary": f"Page {page_no} failed.", "keywords": [],
        "page_type": "other", "failure_reason": last_err or "Unknown"
    }, False

# ── Main pipeline ────────────────────────────────────────────────
def pdf_to_images(pdf_path):
    from pdf2image import convert_from_path
    from config import PDF_DPI
    print(f"📄 Loading: {pdf_path}")
    imgs = convert_from_path(pdf_path, dpi=PDF_DPI)
    print(f"   → {len(imgs)} pages")
    return imgs

def process_batches(images):
    from config import BATCH_SIZE, LLM_BACKEND
    chunks, failed = [], []
    total_batches = (len(images) + BATCH_SIZE - 1) // BATCH_SIZE
    for b_idx in range(total_batches):
        start = b_idx * BATCH_SIZE
        end   = min(start + BATCH_SIZE, len(images))
        print(f"\n📦 Batch {b_idx+1}/{total_batches} (pages {start+1}–{end})")
        for i, img in enumerate(images[start:end]):
            page_no  = start + i + 1
            chunk_no = len(chunks) + 1
            print(f"   🔍 Analyzing page {page_no}...")
            meta, ok = _extract_with_retry(img, page_no)
            if not ok:
                failed.append({"page_no": page_no, "reason": meta.get("failure_reason","?")})
            chunks.append({
                "chunk_id": f"chunk_{chunk_no:04d}",
                "chunk_no": chunk_no, "page_no": page_no, "batch_no": b_idx+1,
                "heading":  meta.get("heading", f"Page {page_no}"),
                "sub_headings": meta.get("sub_headings", []),
                "content":  meta.get("content_markdown", ""),
                "summary":  meta.get("summary", ""),
                "keywords": meta.get("keywords", []),
                "page_type":meta.get("page_type", "content"),
                "extraction_failed": not ok,
            })
            status = "⚠️ FAILED" if not ok else "✅"
            print(f"   {status} Chunk {chunk_no} | '{chunks[-1]['heading'][:45]}'")
            if LLM_BACKEND == "mistral":
                time.sleep(1.2)
    return chunks, failed

def save_chunks(chunks, path=None):
    from config import OUTPUT_JSON
    p = path or OUTPUT_JSON
    with open(p, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved {len(chunks)} chunks → {p}")

def save_failed(failed, path=None):
    from config import FAILED_PAGES_LOG
    p = path or FAILED_PAGES_LOG
    log = {"failed_pages": failed, "total_failures": len(failed)}
    with open(p, "w") as f:
        json.dump(log, f, indent=2)
    if failed:
        print(f"📋 {len(failed)} failed pages → {p}")

def store_chromadb(chunks, collection_name=None, db_path=None):
    import chromadb
    from config import CHROMA_PATH, CHROMA_COLLECTION
    path = db_path or CHROMA_PATH
    col_name = collection_name or CHROMA_COLLECTION
    print(f"\n🗄️  Storing in ChromaDB collection '{col_name}'...")
    client = chromadb.PersistentClient(path=path)
    try:
        client.delete_collection(col_name)
    except Exception:
        pass
    col = client.create_collection(col_name)
    ids, embeddings, documents, metadatas = [], [], [], []
    skipped = 0
    for c in chunks:
        text = c.get("content","").strip()
        if c.get("extraction_failed") or not text:
            skipped += 1
            continue
        print(f"   🔢 Embedding chunk {c['chunk_no']} (page {c['page_no']})...")
        try:
            emb = _embed(text)
            ids.append(c["chunk_id"])
            embeddings.append(emb)
            documents.append(text)
            metadatas.append({
                "chunk_no": c["chunk_no"], "page_no": c["page_no"],
                "batch_no": c["batch_no"], "heading": c["heading"],
                "summary":  c["summary"],
                "keywords": ", ".join(c.get("keywords",[])),
                "page_type":c["page_type"],
            })
        except Exception as e:
            print(f"   ❌ Embed failed chunk {c['chunk_no']}: {e}")
            skipped += 1
    if ids:
        col.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    print(f"   ✅ Stored {len(ids)} chunks | Skipped {skipped}")
    return len(ids)

def ingest_pdf(pdf_path, collection_name=None):
    print("="*55)
    print("  AGENTIC RAG — INGESTION PIPELINE")
    print("="*55)
    images = pdf_to_images(pdf_path)
    chunks, failed = process_batches(images)
    save_chunks(chunks)
    save_failed(failed)
    stored = store_chromadb(chunks, collection_name)
    print("\n"+"="*55)
    print(f"  ✅ DONE: {stored}/{len(images)} pages stored")
    if failed:
        print(f"  ⚠️  Failed: {[p['page_no'] for p in failed]}")
    print("="*55)
    return chunks, failed
