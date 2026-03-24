"""
Tests for Research Agent
Run: pytest tests/ -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from unittest.mock import MagicMock

# ── search_arxiv ──────────────────────────────────────────────────────────────


class TestSearchArxiv:
    def test_returns_list(self):
        from agent import search_arxiv

        results = search_arxiv("transformer neural network", max_results=2)
        assert isinstance(results, list)

    def test_paper_has_required_keys(self):
        from agent import search_arxiv

        results = search_arxiv("BERT language model", max_results=1)
        if results:
            paper = results[0]
            for key in [
                "title",
                "authors",
                "abstract",
                "pdf_url",
                "arxiv_id",
                "published",
            ]:
                assert key in paper, f"Missing key: {key}"

    def test_empty_query_returns_list(self):
        from agent import search_arxiv

        results = search_arxiv("xyzzy_nonexistent_topic_12345", max_results=1)
        assert isinstance(results, list)


# ── extract_pdf_text ──────────────────────────────────────────────────────────


class TestExtractPdfText:
    def test_returns_string_on_valid_pdf(self):
        from agent import search_arxiv, extract_pdf_text

        papers = search_arxiv("attention is all you need", max_results=1)
        if not papers:
            pytest.skip("ArXiv unavailable")
        text = extract_pdf_text(papers[0]["pdf_url"], max_pages=2)
        assert isinstance(text, str)
        assert len(text) > 100

    def test_raises_on_invalid_url(self):
        from agent import extract_pdf_text

        with pytest.raises(Exception):
            extract_pdf_text("https://arxiv.org/pdf/this-does-not-exist.pdf")


# ── summarize_paper ───────────────────────────────────────────────────────────


class TestSummarizePaper:
    def test_returns_string(self):
        from agent import summarize_paper

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "This paper presents a method for X."
        result = summarize_paper(
            mock_llm, "Test Paper", "Some text about neural networks."
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_uses_paper_title_in_prompt(self):
        from agent import summarize_paper

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Summary."
        summarize_paper(mock_llm, "My Unique Title XYZ", "text")
        call_args = mock_llm.invoke.call_args[0][0]
        assert "My Unique Title XYZ" in call_args


# ── build_vector_store ────────────────────────────────────────────────────────


class TestBuildVectorStore:
    def test_returns_faiss_store(self):
        from agent import build_vector_store
        from langchain_community.embeddings import HuggingFaceEmbeddings

        papers = [
            {
                "title": "Test Paper",
                "authors": ["Author A"],
                "arxiv_id": "1234.5678",
                "url": "https://arxiv.org/abs/1234.5678",
                "full_text": "This is a test paper about machine learning and neural networks. "
                * 10,
            }
        ]
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        from langchain_community.vectorstores import FAISS

        store = build_vector_store(papers, embeddings)
        assert isinstance(store, FAISS)

    def test_empty_papers_raises(self):
        from agent import build_vector_store

        with pytest.raises(Exception):
            build_vector_store([], MagicMock())


# ── ask_question ──────────────────────────────────────────────────────────────


class TestAskQuestion:
    def test_returns_answer_and_sources(self):
        from agent import ask_question
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain.schema import Document

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        docs = [
            Document(
                page_content="Attention mechanisms allow models to focus on relevant tokens.",
                metadata={
                    "title": "Attention Paper",
                    "arxiv_id": "1706.03762",
                    "authors": "Vaswani et al.",
                    "url": "https://arxiv.org/abs/1706.03762",
                },
            )
        ]
        store = FAISS.from_documents(docs, embeddings)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            "Attention helps models focus on important tokens."
        )

        result = ask_question(mock_llm, store, "What do attention mechanisms do?")
        assert "answer" in result
        assert "sources" in result
        assert isinstance(result["sources"], list)


# ── get_llm (env switching) ───────────────────────────────────────────────────


class TestGetLLM:
    def test_invalid_backend_raises(self):
        import os
        from agent import get_llm

        os.environ["LLM_BACKEND"] = "gpt4o_paid"
        with pytest.raises(ValueError):
            get_llm()
        os.environ["LLM_BACKEND"] = "ollama"

    def test_huggingface_without_token_raises(self):
        import os
        from agent import get_llm

        os.environ["LLM_BACKEND"] = "huggingface"
        os.environ.pop("HF_API_TOKEN", None)
        with pytest.raises(EnvironmentError):
            get_llm()
        os.environ["LLM_BACKEND"] = "ollama"
