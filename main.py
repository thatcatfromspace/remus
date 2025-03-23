# main.py
import pdfplumber
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
    """Extract text from a PDF."""
    with pdfplumber.open(file_path) as pdf:
        text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return Document(content=text, meta={"source": "pdf"})

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

def generate_summary(context):
    """Generate summary using Mistral via Ollama, handling streaming response."""
    prompt = f"Briefly summarize this humanitarian data text: {context}"
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "mistral:7b-instruct-q4_0",
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0.5,
            "top_p": 0.9,
        },
        stream=True  # Enable streaming
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
    # # Process and index the PDF (uncomment first run)
    # pdf_file = "data/sample.pdf"
    # doc = process_pdf(pdf_file)
    # indexing_pipeline.run({"doc_embedder": {"documents": [doc]}})

    # Test retrieval and generation
    query = "What’s in the PDF?"
    results = query_pipeline.run({"text_embedder": {"text": query}, "retriever": {"top_k": 3}})
    context = " ".join([doc.content for doc in results["retriever"]["documents"]])
    context = context[:500] + "..." if len(context) > 500 else context
    
    response = generate_summary(context)

    print("\n\nQuery:", query)
    print("\n\nContext (truncated):", context)
    print("\n\nResponse:", response)

if __name__ == "__main__":
    main()