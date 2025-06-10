# pipelines.py
import logging

from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RagPipeline:
    def __init__(self):
        self.document_store = ChromaDocumentStore(persist_path="./chroma_data")
        self.doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        self.text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        self.retriever = ChromaEmbeddingRetriever(document_store=self.document_store)
        self.doc_writer = DocumentWriter(document_store=self.document_store)

        self.indexing_pipeline = Pipeline()
        self.indexing_pipeline.add_component("doc_embedder", self.doc_embedder)
        self.indexing_pipeline.add_component("doc_writer", self.doc_writer)
        self.indexing_pipeline.connect("doc_embedder.documents", "doc_writer.documents")

        self.query_pipeline = Pipeline()
        self.query_pipeline.add_component("text_embedder", self.text_embedder)
        self.query_pipeline.add_component("retriever", self.retriever)
        self.query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    
    def index_documents(self, documents):
        if not documents:
            logger.warning("No documents provided to index")
            return
        logger.info(f"Indexing {len(documents)} documents: {[doc.meta['source'] for doc in documents]}")
        result = self.indexing_pipeline.run({"doc_embedder": {"documents": documents}})
        logger.info(f"Indexed {len(documents)} documents in ChromaDB")

    def query(self, query_text, top_k=3):
        logger.info(f"Querying with text: {query_text}, top_k={top_k}")
        # Log all documents in ChromaDB before querying
        all_docs = self.document_store.filter_documents()
        logger.info(f"All documents in ChromaDB: {[doc.meta['source'] for doc in all_docs]}")
        result = self.query_pipeline.run({"text_embedder": {"text": query_text}, "retriever": {"top_k": top_k}})
        logger.info(f"Retrieved {len(result['retriever']['documents'])} documents")
        return result