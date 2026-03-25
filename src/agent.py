"""
Research Agent — Core Logic
===========================
100% free & open-source stack:
  • LLM   : Ollama (llama3 / mistral running locally)  OR
            HuggingFace Inference API (free tier)
  • Embed : sentence-transformers (runs fully offline, no API key)
  • Store : FAISS (in-memory vector search)
  • Fetch : arxiv + PyMuPDF
"""

from __future__ import annotations

import os
import json
import subprocess
import tempfile
import urllib.request
import urllib.error
import logging
from functools import lru_cache

import arxiv
import fitz  # PyMuPDF
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()
logger = logging.getLogger(__name__)

# ── LLM backend selector ──────────────────────────────────────────────────────


def get_llm():
    """
    Returns the best available free LLM:
      1. Ollama  (local — zero cost, needs `ollama serve` running)
      2. HuggingFace Inference API (free tier, needs HF_API_TOKEN in .env)
    Set LLM_BACKEND=ollama or LLM_BACKEND=huggingface in .env
    """
    backend = os.getenv("LLM_BACKEND", "ollama").lower()

    def _ollama_tags(base_url: str) -> dict:
        url = f"{base_url.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)

    def _model_present(tags_payload: dict, wanted_model: str) -> bool:
        models = tags_payload.get("models", []) or []
        wanted_root = wanted_model.split(":")[0]
        for m in models:
            name = m.get("name", "")
            if name == wanted_model:
                return True
            if name.split(":")[0] == wanted_root:
                return True
        return False

    def _try_pull_ollama_model(model: str) -> None:
        # Pull only works where `ollama` CLI exists and the Ollama server is reachable.
        # On hosted platforms, this will fail and we will fallback to HF.
        subprocess.run(["ollama", "pull", model], check=False, capture_output=True, text=True)

    if backend == "ollama":
        from langchain_community.llms import Ollama

        model = os.getenv("OLLAMA_MODEL", "llama3")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            tags = _ollama_tags(base_url)
            if not _model_present(tags, model):
                logger.warning("Ollama model '%s' not found locally. Trying to pull...", model)
                _try_pull_ollama_model(model)
                tags = _ollama_tags(base_url)
                if not _model_present(tags, model):
                    raise RuntimeError(
                        f"Ollama model '{model}' not found even after `ollama pull {model}`. "
                        "Install it locally or switch OLLAMA_MODEL."
                    )

            logger.info("Using Ollama backend with model: %s", model)
            return Ollama(
                model=model,
                base_url=base_url,
                temperature=0.2,
            )
        except Exception as exc:
            hf_token = os.getenv("HF_API_TOKEN", "").strip()
            logger.warning(
                "Ollama unavailable or model missing (%s).",
                exc,
            )
            if hf_token:
                logger.info("HF_API_TOKEN is set; falling back to HuggingFace backend.")
                backend = "huggingface"
            else:
                # Hosted env might not have HF_TOKEN either; local users should pull model or switch dropdown.
                raise

    if backend == "huggingface":
        from langchain_community.llms import HuggingFaceEndpoint

        token = os.getenv("HF_API_TOKEN")
        if not token:
            raise EnvironmentError(
                "HF_API_TOKEN is required for HuggingFace backend. "
                "On hosted environments (no Ollama), set HF_API_TOKEN in secrets/env."
            )
        repo = os.getenv("HF_MODEL_REPO", "mistralai/Mistral-7B-Instruct-v0.3")
        logger.info("Using HuggingFace backend: %s", repo)
        return HuggingFaceEndpoint(
            repo_id=repo,
            huggingfacehub_api_token=token,
            temperature=0.2,
            max_new_tokens=512,
        )

    raise ValueError(f"Unknown LLM_BACKEND: '{backend}'. Use 'ollama' or 'huggingface'.")


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Returns sentence-transformers embeddings — runs completely offline.
    Model is ~90 MB and downloads once on first run.
    """
    model_name = os.getenv(
        "EMBED_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",  # fast & lightweight
    )
    logger.info("Loading embeddings model: %s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── ArXiv Search ──────────────────────────────────────────────────────────────


def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """Search ArXiv and return structured paper metadata."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = []
    for result in client.results(search):
        papers.append(
            {
                "title": result.title,
                "authors": [a.name for a in result.authors[:3]],
                "abstract": result.summary,
                "pdf_url": result.pdf_url,
                "arxiv_id": result.entry_id.split("/")[-1],
                "published": str(result.published.date()),
                "url": f"https://arxiv.org/abs/{result.entry_id.split('/')[-1]}",
            }
        )
    return papers


# ── PDF Extraction ────────────────────────────────────────────────────────────


def extract_pdf_text(pdf_url: str, max_pages: int = 8) -> str:
    """Download a PDF and extract plain text from the first N pages."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        urllib.request.urlretrieve(pdf_url, tmp.name)
        doc = fitz.open(tmp.name)
        text_parts = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            text_parts.append(page.get_text())
        doc.close()
    return "\n".join(text_parts).strip()


# ── Summarization ─────────────────────────────────────────────────────────────

SUMMARY_PROMPT = PromptTemplate.from_template(
    """You are a helpful research assistant. Read this excerpt from the paper titled "{title}" and write a clear, concise summary in 3-4 sentences. Explain: what problem it solves, the approach used, and the key result.

Excerpt:
{text}

Summary (plain English, no jargon):"""
)


def summarize_paper(llm, title: str, text: str) -> str:
    """Ask the LLM to summarize a paper in plain English."""
    # Cap excerpt size to keep summarization fast during interactive runs.
    prompt = SUMMARY_PROMPT.format(title=title, text=text[:2000])
    response = llm.invoke(prompt)
    # Ollama returns a string; HF may return an object
    return response if isinstance(response, str) else response.content


# ── Vector Store ──────────────────────────────────────────────────────────────


def build_vector_store(papers: list[dict], embeddings) -> FAISS:
    """Chunk paper texts and build a FAISS vector index."""
    splitter = RecursiveCharacterTextSplitter(
        # Fewer/smaller chunks => faster indexing + less LLM overhead later.
        chunk_size=900,
        chunk_overlap=120,
    )
    docs = []
    for paper in papers:
        chunks = splitter.split_text(paper["full_text"])
        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "title": paper["title"],
                        "arxiv_id": paper["arxiv_id"],
                        "authors": ", ".join(paper["authors"]),
                        "url": paper["url"],
                    },
                )
            )
    return FAISS.from_documents(docs, embeddings)


# ── Q&A Chain ─────────────────────────────────────────────────────────────────

QA_PROMPT = PromptTemplate.from_template(
    """You are a research assistant. Use only the provided context from research papers to answer the question.
If you cannot find the answer in the context, say "I couldn't find that in the loaded papers."

Context:
{context}

Question: {question}

Answer:"""
)


def ask_question(llm, vector_store: FAISS, question: str) -> dict:
    """Retrieve relevant chunks and answer the question with source attribution."""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT},
    )
    result = qa_chain.invoke({"query": question})

    seen, sources = set(), []
    for doc in result["source_documents"]:
        aid = doc.metadata.get("arxiv_id", "")
        if aid not in seen:
            seen.add(aid)
            sources.append(
                {
                    "title": doc.metadata.get("title", "Unknown"),
                    "arxiv_id": aid,
                    "authors": doc.metadata.get("authors", ""),
                    "url": doc.metadata.get("url", ""),
                }
            )

    return {"answer": result["result"], "sources": sources}


# ── Full Pipeline ─────────────────────────────────────────────────────────────


def run_research_pipeline(
    query: str,
    max_papers: int = 4,
    status_callback=None,
) -> dict:
    """
    Agentic pipeline:
      1. Search ArXiv
      2. Fetch & extract PDFs
      3. Summarize with LLM
      4. Embed into FAISS
    Returns enriched papers + vector store + llm handle.
    """

    def _status(msg: str):
        logger.info(msg)
        if status_callback:
            status_callback(msg)

    _status("🔍 Initialising LLM and embeddings...")
    llm = get_llm()
    embeddings = get_embeddings()

    _status(f"📡 Searching ArXiv for: {query}")
    papers = search_arxiv(query, max_results=max_papers)
    if not papers:
        return {"error": "No papers found. Try a different query."}

    enriched = []
    for i, paper in enumerate(papers, 1):
        _status(f"📄 Processing paper {i}/{len(papers)}: {paper['title'][:60]}...")
        try:
            full_text = extract_pdf_text(paper["pdf_url"])
        except Exception as exc:
            logger.warning("PDF fetch failed for %s: %s", paper["arxiv_id"], exc)
            full_text = paper["abstract"]

        try:
            summary = summarize_paper(llm, paper["title"], full_text)
        except Exception as exc:
            logger.warning("Summarization failed: %s", exc)
            summary = paper["abstract"][:400]

        enriched.append({**paper, "full_text": full_text, "ai_summary": summary})

    _status("🧠 Building vector index...")
    vector_store = build_vector_store(enriched, embeddings)

    _status("✅ Pipeline complete!")
    return {"papers": enriched, "vector_store": vector_store, "llm": llm}
