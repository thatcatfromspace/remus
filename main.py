# # main.py
# import pdfplumber
# from haystack import Document
# from haystack.document_stores import InMemoryDocumentStore
# #from haystack_integrations.document_stores.chroma import ChromaDocumentStore
# from haystack.nodes import EmbeddingRetriever
# from transformers import pipeline
# # Process PDF first
# # Initialize InMemoryDocumentStore and Retriever
# document_store = InMemoryDocumentStore()
# retriever = EmbeddingRetriever(
#     document_store=document_store,
#     embedding_model="sentence-transformers/all-MiniLM-L6-v2"
# )

# # Initialize Generator
# generator = pipeline("text-generation", model="distilgpt2")

# def process_pdf(file_path):
#     """Extract text from a PDF and return as a Document."""
#     with pdfplumber.open(file_path) as pdf:
#         text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
#     return Document(content=text, meta={"source": "pdf"})

# def main():
#     # Process a sample PDF
#     pdf_file = "data/sample.pdf"
#     doc = process_pdf(pdf_file)
    
#     # Embed and store
#     embeddings = retriever.embed_documents([doc])
#     doc.embedding = embeddings[0]
#     document_store.write_documents([doc])
    
#     # Test retrieval and generation
#     query = "What’s in the PDF?"
#     results = retriever.retrieve(query=query, top_k=3)
#     context = " ".join([doc.content for doc in results])
#     response = generator(
#         context,
#         max_new_tokens=50,  # Generate up to 50 new tokens
#         truncation=True,    # Truncate input if needed
#         num_return_sequences=1
#     )[0]["generated_text"]
    
#     print("Query:", query)
#     print("Response:", response)

# if __name__ == "__main__":
#     main()

# main.py
import pdfplumber
from haystack import Document, Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter
from transformers import pipeline

# Initialize ChromaDocumentStore with persistence
document_store = ChromaDocumentStore(persist_path="./chroma_data")

# Initialize embedders, retriever, and writer
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
retriever = ChromaEmbeddingRetriever(document_store=document_store)
doc_writer = DocumentWriter(document_store=document_store)

# Initialize generator with a prompt-friendly setup
generator = pipeline("text-generation", model="distilgpt2")

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

def main():
    # Process and index the PDF
    pdf_file = "data/sample.pdf"
    doc = process_pdf(pdf_file)
    indexing_pipeline.run({"doc_embedder": {"documents": [doc]}})

    # Test retrieval and generation
    query = "What’s in the PDF?"
    results = query_pipeline.run({"text_embedder": {"text": query}, "retriever": {"top_k": 3}})
    context = " ".join([doc.content for doc in results["retriever"]["documents"]])
    context = context[:500] + "..." if len(context) > 500 else context
    
    # Add a guiding prompt
    prompt = f"Based on this context: '{context}', provide a concise summary of the PDF content."
    response = generator(
        prompt,
        max_new_tokens=150,  # More room for generation
        truncation=True,
        num_return_sequences=1,
        do_sample=True,  # Add creativity
        temperature=0.7   # Balance coherence and variety
    )[0]["generated_text"]

    print("Query:", query)
    print("Context (truncated):", context)
    print("Response:", response)

if __name__ == "__main__":
    main()