import time
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import redis
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from file_processor import FileProcessor
from pipelines import RagPipeline
from summarizer import Summarizer
from web_search import WebSearch
import os
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_file_wrapper(file):
    return FileProcessor().process_file(file)

class Chatbot:
    def __init__(self, documents):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        self.processor = FileProcessor()
        self.pipeline = RagPipeline()
        self.summarizer = Summarizer()
        self.web_searcher = WebSearch()
        self.files = [
            "data/sample.pdf",
            "data/resume.docx",
            "data/grades.csv",
            "data/test.jpg",
            "data/videos/dog_flipped.mp4"
        ]
        self.uploaded_files: Dict[str, object] = {}  # Track uploaded files

        # Index the pre-processed documents
        start_time = time.time()
        self.pipeline.index_documents(documents)
        print(f"Indexing time: {time.time() - start_time:.2f}s")

# Pydantic model for /query endpoint JSON input
class QueryRequest(BaseModel):
    query_text: str
    use_web_search: bool
    file_to_query: str

def query(chatbot, query_text: str, use_web_search: bool, file_to_query: str = None) -> str:
    logger.info(f"Received query: query_text='{query_text}', use_web_search={use_web_search}, file_to_query={file_to_query}")
    
    if not query_text.strip():
        logger.error("Query text is empty")
        raise HTTPException(status_code=400, detail="Query text cannot be empty")

    file_key = file_to_query if file_to_query and file_to_query != "all_files" else "all_files"
    cache_key = f"default:{file_key}:{query_text}:{use_web_search}"
    
    try:
        cached_response = chatbot.redis_client.get(cache_key)
        if cached_response:
            logger.info(f"Cache hit: {cache_key}")
            return cached_response
    except redis.RedisError as e:
        logger.error(f"Redis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Redis error: {str(e)}")
    
    start_time = time.time()
    try:
        logger.info("Querying pipeline...")
        results = chatbot.pipeline.query(query_text)
        logger.info("Pipeline query successful")
        docs = results.get("retriever", {}).get("documents", [])
        # Log the retrieved documents
        logger.info(f"Retrieved {len(docs)} documents: {[doc.meta['source'] for doc in docs]}")
        if not docs:
            logger.warning("No documents retrieved from pipeline")
    except Exception as e:
        logger.error(f"Pipeline query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline query failed: {str(e)}")
    
    if not file_to_query or file_to_query == "all_files":
        try:
            logger.info("Querying all documents...")
            all_docs = chatbot.pipeline.query(query_text, top_k=10)["retriever"]["documents"]
            docs_by_source = {doc.meta["source"]: doc for doc in all_docs}
            docs = list(docs_by_source.values())
            if not docs:
                logger.warning("No documents available to summarize")
                raise HTTPException(status_code=404, detail="No documents available to summarize")
            context_parts = [f"{doc.meta['source']}: {doc.content[:100]}" for doc in docs]
            context = " | ".join(context_parts)
            context = context[:1000] + "..." if len(context) > 1000 else context
            prompt = "Summarize the content from all file types:"
        except Exception as e:
            logger.error(f"Error processing all documents: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing all documents: {str(e)}")
    else:
        # Filter by full file name (e.g., "synchronization_22pd26.pdf")
        docs = [doc for doc in docs if doc.meta["source"] == file_to_query]
        if not docs:
            logger.warning(f"No content found for file: {file_to_query}")
            raise HTTPException(status_code=404, detail=f"No content found for file: {file_to_query}")
        context = " ".join([doc.content for doc in docs])
        context = context[:500] + "..." if len(context) > 500 else context
        prompt = f"Summarize {file_to_query} content:"
    
    sources = list(set(doc.meta["source"] for doc in docs))
    if use_web_search:
        try:
            logger.info("Performing web search...")
            with ThreadPoolExecutor(max_workers=2) as executor:
                web_future = executor.submit(chatbot.web_searcher.search, query_text)
                web_results = web_future.result()
            if web_results:
                context += " | Web Results: " + " | ".join(web_results[:2])
                sources.append("web")
            logger.info("Web search successful")
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Web search failed: {str(e)}")
    
    try:
        logger.info("Generating summary...")
        response = chatbot.summarizer.generate_summary(context, sources, prompt)
        logger.info("Summary generated successfully")
    except Exception as e:
        logger.error(f"Summary generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")
    
    try:
        chatbot.redis_client.setex(cache_key, 3600, response)
        logger.info(f"Cached response for key: {cache_key}")
    except redis.RedisError as e:
        logger.error(f"Redis caching error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Redis caching error: {str(e)}")
    
    print(f"Query time: {time.time() - start_time:.2f}s")
    return response

# FastAPI App
app = FastAPI(
    title="RAG Chatbot API",
    description="API for querying files (PDF, DOCX, CSV, image, video) with optional web search. Powered by Redis caching.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chatbot instance (will be initialized in main)
chatbot = None

@app.get("/files", response_model=Dict[str, List[str]])
async def get_files():
    """Get a list of available files to query."""
    return {"files": ["all_files"] + chatbot.files + list(chatbot.uploaded_files.keys())}

@app.post("/query", response_model=Dict[str, str])
async def post_query(request: QueryRequest):
    """Query the chatbot with a text prompt, optionally including web search and targeting a specific file."""
    try:
        response = query(chatbot, request.query_text, request.use_web_search, request.file_to_query)
        return {"response": response}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in /query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

class UploadRequest(BaseModel):
    file_name: str
    file_content: str  # Base64-encoded file content

@app.post("/upload", response_model=Dict[str, str])
async def upload_file(file: UploadFile = File(...)):
    """Upload a file to be processed and indexed for querying."""
    if not file:
        logger.error("No file provided")
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_name = file.filename
    if not file_name:
        logger.error("File name is empty")
        raise HTTPException(status_code=400, detail="File name cannot be empty")

    file_path = f"data/{file_name}"
    os.makedirs("data", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    try:
        logger.info(f"Processing file: {file_name}")
        doc = chatbot.processor.process_file(file_path)
        logger.info(f"Document processed: {doc.meta['source']}, content length: {len(doc.content)}")
        chatbot.pipeline.index_documents([doc])
        chatbot.uploaded_files[file_name] = doc
        logger.info(f"Successfully uploaded, processed, and indexed: {file_name}")
        return {"message": f"Uploaded and processed: {file_name}"}
    except Exception as e:
        logger.error(f"Failed to process file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    
if __name__ == "__main__":
    # Process files using multiprocessing
    files = [
            "data/sample.pdf",
            "data/sample.docx",
            "data/sample.csv",
            "data/images/dog.jpeg",  
            "data/videos/dog_flipped.mp4"  
        ]
    start_time = time.time()
    print(f"Setup time: {time.time() - start_time:.2f}s")

    start_time = time.time()
    with Pool(processes=4) as pool:  # 4 performance cores on M1
        documents = pool.map(process_file_wrapper, files)
    print(f"Parallel file processing time: {time.time() - start_time:.2f}s")

    # Initialize the chatbot with pre-processed documents
    chatbot = Chatbot(documents)

    # Start the FastAPI app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)