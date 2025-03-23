# main.py
import pdfplumber
import docx
import pandas as pd
import requests
import json
from haystack import Document, Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter

# Initialize ChromaDocumentStore with persistence
document_store = ChromaDocumentStore(persist_path="./chroma_data")

# Initialize embedders, retriever, and writer
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
retriever = ChromaEmbeddingRetriever(document_store=document_store)
doc_writer = DocumentWriter(document_store=document_store)

# Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

def process_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return Document(content=text, meta={"source": "pdf"})

def process_docx(file_path):
    doc = docx.Document(file_path)
    text = " ".join([para.text for para in doc.paragraphs if para.text])
    return Document(content=text, meta={"source": "docx"})

def process_csv(file_path):
    df = pd.read_csv(file_path)
    text = " ".join(df.astype(str).agg(" ".join, axis=1))
    return Document(content=text, meta={"source": "csv"})

def process_file(file_path):
    if file_path.endswith(".pdf"):
        return process_pdf(file_path)
    elif file_path.endswith(".docx"):
        return process_docx(file_path)
    elif file_path.endswith(".csv"):
        return process_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

# Create indexing pipeline
indexing_pipeline = Pipeline()
indexing_pipeline.add_component("doc_embedder", doc_embedder)
indexing_pipeline.add_component("doc_writer", doc_writer)
indexing_pipeline.connect("doc_embedder.documents", "doc_writer.documents")

# Create query pipeline
query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", text_embedder)
query_pipeline.add_component("retriever", retriever)
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

def generate_summary(context, sources, prompt="Briefly summarize this text:"):
    """Generate summary with source hint."""
    source_hint = f"From {', '.join(sources)}: " if sources else ""
    full_prompt = f"{source_hint}{prompt} {context}"
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "mistral:7b-instruct-q4_0",
            "prompt": full_prompt,
            "max_tokens": 50,
            "temperature": 0.5,
            "top_p": 0.9,
        },
        stream=True
    )
    full_response = ""
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line.decode('utf-8'))
            full_response += chunk.get("response", "")
            if chunk.get("done", False):
                break
    return full_response.strip()

def main():
    # # Process and index multiple files (uncomment first run)
    # files = ["data/sample.pdf", "data/sample.docx", "data/sample.csv"]
    # documents = [process_file(file) for file in files]
    # indexing_pipeline.run({"doc_embedder": {"documents": documents}})

    # Test refined queries/prompts
    queries = [
        ("What’s in the files?", "Summarize the content:", []),
        ("Tell me about my resume", "Summarize my resume:", ["docx"]),
        ("What are the student grades?", "Summarize student performance:", ["csv"]),
        ("What’s the humanitarian data about?", "Summarize humanitarian data:", ["pdf"])
    ]

    for query, prompt, source_filter in queries:
        results = query_pipeline.run({"text_embedder": {"text": query}, "retriever": {"top_k": 3}})
        docs = results["retriever"]["documents"]
        # Optionally filter by source (uncomment to enforce strict matching)
        # # For generic query, ensure one doc from each source
        # if not source_filter:
        #     all_docs = query_pipeline.run({"text_embedder": {"text": query}, "retriever": {"top_k": 10}})["retriever"]["documents"]
        #     docs = []
        #     for source in ["pdf", "docx", "csv"]:
        #         source_doc = next((d for d in all_docs if d.meta["source"] == source), None)
        #         if source_doc: docs.append(source_doc)
        # # For specific queries, filter by source
        # else:
        #     docs = [doc for doc in docs if doc.meta["source"] in source_filter]
        if source_filter:
            docs = [doc for doc in docs if doc.meta["source"] in source_filter]
        context = " ".join([doc.content for doc in docs])
        context = context[:500] + "..." if len(context) > 500 else context
        sources = list(set(doc.meta["source"] for doc in docs))
        response = generate_summary(context, sources, prompt)

        print("\n\nQuery:", query)
        print("Prompt:", f"{', '.join(sources)}: {prompt}" if sources else prompt)
        print("\nContext (truncated):", context)
        print("\nResponse:", response)

if __name__ == "__main__":
    main()