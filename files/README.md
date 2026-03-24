# 🔬 Agentic Research Paper Assistant

> An agentic AI that autonomously searches ArXiv, reads research papers, builds a knowledge base, and answers your questions — **100% free and open-source, no paid API keys required.**

[![CI](https://github.com/YOUR_USERNAME/research-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/research-agent/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker)](Dockerfile)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## ✨ What It Does

1. You enter a research topic (e.g. *"attention mechanisms in transformers"*)
2. The agent **autonomously searches ArXiv** for the most relevant papers
3. It **downloads and reads** the actual PDFs using PyMuPDF
4. It **summarises** each paper in plain English using a local LLM
5. It **embeds** all paper content into a FAISS vector store
6. You can **ask follow-up questions** — answered with source citations

---

## 🧠 Agentic AI Concepts Demonstrated

| Concept | Implementation |
|---|---|
| **Tool use** | Agent calls ArXiv search → PDF fetcher → summariser as separate tools |
| **Multi-step reasoning** | Search → Extract → Summarise → Embed → Retrieve → Answer |
| **RAG** | Answers grounded in real paper content via FAISS retrieval |
| **Vector Memory** | Papers chunked and stored as embeddings for semantic search |
| **Autonomous decision making** | Agent selects what to fetch and how to respond without hand-holding |

---

## 🛠️ Tech Stack (100% Free & Open-Source)

| Layer | Tool | Why |
|---|---|---|
| **LLM (local)** | [Ollama](https://ollama.com) + llama3/mistral | Runs on your machine, zero cost |
| **LLM (cloud)** | [HuggingFace Inference API](https://huggingface.co/inference-api) | Free tier, no credit card |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Runs fully offline |
| **Vector DB** | FAISS | In-memory, no server needed |
| **Papers** | ArXiv API | Completely free |
| **PDF parsing** | PyMuPDF | Fast, open-source |
| **Framework** | LangChain | Agent orchestration |
| **UI** | Streamlit | Zero-config web interface |
| **Containerisation** | Docker + Docker Compose | Reproducible deploys |
| **CI/CD** | GitHub Actions | Auto-lint, test, security scan, build |
| **Linting** | Ruff + Black | Fast, modern Python tooling |
| **Security** | Bandit (SAST) + Safety (CVE scan) | Vulnerability scanning |
| **Testing** | Pytest + pytest-cov | Unit tests with coverage |

---

## 🚀 Quick Start

### Option A — Docker Compose (Recommended, one command)

```bash
git clone https://github.com/YOUR_USERNAME/research-agent.git
cd research-agent

# Start the app + Ollama LLM server
docker compose up --build

# In a new terminal, pull the LLM model (one-time, ~4 GB)
docker compose exec ollama ollama pull llama3

# Open http://localhost:8501
```

### Option B — Run Locally

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com/download) installed

```bash
git clone https://github.com/YOUR_USERNAME/research-agent.git
cd research-agent

# Install dependencies
make install               # creates venv + installs everything
source venv/bin/activate   # (Windows: venv\Scripts\activate)

# Configure environment
cp .env.example .env       # set LLM_BACKEND and other options

# Pull a local model (in a separate terminal)
ollama pull llama3         # or: mistral, phi3, gemma

# Start Ollama server (keep this running)
ollama serve

# Launch the app
make run
# → http://localhost:8501
```

### Option C — HuggingFace (no GPU, no Ollama)

```bash
cp .env.example .env
# Edit .env:
#   LLM_BACKEND=huggingface
#   HF_API_TOKEN=hf_your_free_token_here   # from huggingface.co/settings/tokens

make run
```

---

## 📁 Project Structure

```
research-agent/
│
├── src/
│   ├── agent.py           ← Core agent (search, extract, summarise, Q&A)
│   └── app.py             ← Streamlit UI
│
├── tests/
│   └── test_agent.py      ← Unit tests (pytest)
│
├── .github/
│   └── workflows/
│       └── ci.yml         ← GitHub Actions (lint → security → test → docker)
│
├── Dockerfile             ← Multi-stage Docker build
├── docker-compose.yml     ← App + Ollama side-by-side
├── requirements.txt       ← Python dependencies
├── pyproject.toml         ← Ruff, Black, Pytest, Bandit config
├── Makefile               ← Developer shortcuts
├── .env.example           ← Environment variable template
├── .gitignore
└── README.md
```

---

## 🔧 Developer Commands

```bash
make help          # list all commands
make install       # set up venv + install deps
make run           # start the Streamlit app
make test          # run unit tests with coverage
make lint          # ruff linter
make format        # black auto-formatter
make security      # bandit + safety security scan
make docker-up     # docker compose up --build
make docker-down   # stop containers
make clean         # remove caches
```

---

## 🔒 Security

This project uses a layered security approach:

| Tool | What it checks |
|---|---|
| **Bandit** | Static code analysis — finds hardcoded secrets, unsafe functions, injection risks |
| **Safety** | Scans `requirements.txt` against the CVE database for known vulnerabilities |
| **Non-root Docker user** | Container runs as `appuser` (UID 1000), not root |
| **Multi-stage Docker build** | Build tools stripped from the final image |
| **`.gitignore`** | Prevents `.env` and secret files from being committed |
| **GitHub Actions** | Security scans run automatically on every push |

Run security checks locally:
```bash
make security
```

---

## 🧪 Tests

```bash
make test           # unit tests only (fast, no internet needed)
make test-all       # includes integration tests (hits ArXiv API)
```

Coverage report is generated in `htmlcov/index.html`.

---

## 🌐 Deploying to Streamlit Cloud (Free)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set **Main file path** to `src/app.py`
4. Add environment variables (from `.env.example`) in the Secrets section
5. Deploy — your app gets a public URL in ~2 minutes

---

## 🗺️ Roadmap / Extension Ideas

- [ ] Add Semantic Scholar or PubMed as additional paper sources
- [ ] Persist FAISS index to disk for sessions that survive restarts
- [ ] Paper comparison feature ("which paper is best for X?")
- [ ] Export a PDF research summary report
- [ ] Add streaming LLM responses to the UI
- [ ] Multi-document chat with session history
- [ ] Support for local PDF uploads (not just ArXiv)

---

## 📝 Resume Description

> *"Built an agentic AI research assistant using LangChain, Ollama (local LLMs), and FAISS that autonomously searches ArXiv, extracts and summarises research papers using RAG, and answers domain-specific questions with source citations. Containerised with Docker, includes CI/CD via GitHub Actions with automated security scanning (Bandit, Safety), linting (Ruff, Black), and unit tests."*

---

## 📄 License

MIT — free to use, modify, and distribute.
