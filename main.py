# main.py
import time
from multiprocessing import Pool
import redis
from file_processor import FileProcessor
from pipelines import RagPipeline
from summarizer import Summarizer
from web_search import WebSearch
from concurrent.futures import ThreadPoolExecutor

def process_file_wrapper(file):
    return FileProcessor().process_file(file)

class Chatbot:
    def __init__(self):
        start_time = time.time()
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        self.processor = FileProcessor()
        self.pipeline = RagPipeline()
        self.summarizer = Summarizer()
        self.web_searcher = WebSearch()
        self.files = [
            "data/sample.pdf",
            "data/sample.docx",
            "data/sample.csv",
            "data/images/dog.jpeg",  
            "data/videos/dog_flipped.mp4"  
        ]
        print(f"Setup time: {time.time() - start_time:.2f}s")

        start_time = time.time()
        # documents = [self.processor.process_file(f) for f in self.files]
        with Pool(processes=4) as pool:  # 4 performance cores on M1
            documents = pool.map(process_file_wrapper, self.files)
        print(f"Parallel file processing time: {time.time() - start_time:.2f}s")

        start_time = time.time()
        self.pipeline.index_documents(documents)
        print(f"Indexing time: {time.time() - start_time:.2f}s")

    def query(self, query_text, source_filter=[], use_web_search=False, user="default"):
        # Cache key: (user, file, prompt)
        file_key = source_filter[0] if source_filter else "all_files"
        cache_key = f"{user}:{file_key}:{query_text}:{use_web_search}"
        
        # Check Redis cache
        cached_response = self.redis_client.get(cache_key)
        if cached_response:
            print(f"Cache hit: {cache_key}")
            return cached_response

        start_time = time.time()
        results = self.pipeline.query(query_text)
        docs = results["retriever"]["documents"]
        
        if not source_filter:
            all_docs = self.pipeline.query(query_text, top_k=10)["retriever"]["documents"]
            docs_by_source = {doc.meta["source"]: doc for doc in all_docs}
            docs = list(docs_by_source.values())
            # Balanced context: 1000 chars per source
            context_parts = [f"{doc.meta['source']}: {doc.content[:100]}" for doc in docs]
            context = " | ".join(context_parts)
            context = context[:1000] + "..." if len(context) > 1000 else context
            prompt = "Summarize the content from all file types:"
        else:
            docs = [doc for doc in docs if doc.meta["source"] in source_filter]
            context = " ".join([doc.content for doc in docs])
            context = context[:500] + "..." if len(context) > 500 else context
            prompt = f"Summarize {source_filter[0]} content:"
        
        sources = list(set(doc.meta["source"] for doc in docs))
        # Web search with ThreadPoolExecutor
        if use_web_search:
            with ThreadPoolExecutor(max_workers=2) as executor:
                web_future = executor.submit(self.web_searcher.search, query_text)
                web_results = web_future.result()
            if web_results:
                context += " | Web Results: " + " | ".join(web_results[:2])
                sources.append("web")
        
        response = self.summarizer.generate_summary(context, sources, prompt)
        self.redis_client.setex(cache_key, 3600, response)       # Cache result (expire in 1 hour)
        print(f"Query time: {time.time() - start_time:.2f}s")
        return response

    def test_queries(self):
        queries = [
            ("What’s in the files?", [], False),
            ("Tell me about my resume", ["docx"], False),
            ("What are the student grades?", ["csv"], False),
            ("What’s the humanitarian data about?", ["pdf"], False),
            # ("What’s the humanitarian data about?", ["pdf"], True),
            ("What’s in the image?", ["image"], False),
            ("What’s in the video?", ["video"], False),
        ]
        for query, source_filter, use_web in queries:
            response = self.query(query, source_filter, use_web)
            print(f"\n\nQuery: {query} (Web: {use_web})\nResponse: {response}")

if __name__ == "__main__":
    chatbot = Chatbot()
    chatbot.test_queries()