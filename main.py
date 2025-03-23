# main.py
import pdfplumber
from haystack import Document
from haystack.document_stores import InMemoryDocumentStore
#from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.nodes import EmbeddingRetriever
from transformers import pipeline
# Process PDF first
# Initialize InMemoryDocumentStore and Retriever
document_store = InMemoryDocumentStore()
retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize Generator
generator = pipeline("text-generation", model="distilgpt2")

def process_pdf(file_path):
    """Extract text from a PDF and return as a Document."""
    with pdfplumber.open(file_path) as pdf:
        text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return Document(content=text, meta={"source": "pdf"})

def main():
    # Process a sample PDF
    pdf_file = "data/sample.pdf"
    doc = process_pdf(pdf_file)
    
    # Embed and store
    embeddings = retriever.embed_documents([doc])
    doc.embedding = embeddings[0]
    document_store.write_documents([doc])
    
    # Test retrieval and generation
    query = "What’s in the PDF?"
    results = retriever.retrieve(query=query, top_k=3)
    context = " ".join([doc.content for doc in results])
    response = generator(
        context,
        max_new_tokens=50,  # Generate up to 50 new tokens
        truncation=True,    # Truncate input if needed
        num_return_sequences=1
    )[0]["generated_text"]
    
    print("Query:", query)
    print("Response:", response)

if __name__ == "__main__":
    main()