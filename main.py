# main.py
from file_processor import FileProcessor
from pipelines import RagPipeline
from summarizer import Summarizer

def main():
    pipeline = RagPipeline()
    summarizer = Summarizer()
    processor = FileProcessor()

    # # Process and index files (uncomment first run)
    # files = [
    #     "data/sample.pdf",
    #     "data/sample.docx",
    #     "data/sample.csv",
    #     "data/images/dog.jpeg",  
    #     "data/videos/dog_flipped.mp4"  
    # ]
    # documents = [processor.process_file(file) for file in files]
    # pipeline.index_documents(documents)

    # Test queries
    queries = [
        ("What’s in the files?", "Summarize the content:", []),
        ("Tell me about my resume", "Summarize my resume:", ["docx"]),
        ("What are the student grades?", "Summarize student performance:", ["csv"]),
        ("What’s the humanitarian data about?", "Summarize humanitarian data:", ["pdf"]),
        ("What’s in the image?", "Summarize image content:", ["image"]),
        ("What’s in the video?", "Summarize video content:", ["video"])
    ]

    for query, prompt, source_filter in queries:
        results = pipeline.query(query)
        docs = results["retriever"]["documents"]
        
        if not source_filter:
            all_docs = pipeline.query(query, top_k=10)["retriever"]["documents"]
            docs_by_source = {}
            for doc in all_docs:
                source = doc.meta["source"]
                if source not in docs_by_source:
                    docs_by_source[source] = doc
            docs = list(docs_by_source.values())
            # Balanced context: 100 chars per source, clear separation
            context_parts = [f"{doc.meta['source']}: {doc.content[:100]}" for doc in docs]
            context = " | ".join(context_parts)  # Use separator for clarity
            context = context[:1000] + "..." if len(context) > 1000 else context
        else:
            docs = [doc for doc in docs if doc.meta["source"] in source_filter]
            context = " ".join([doc.content for doc in docs])
            context = context[:500] + "..." if len(context) > 500 else context
        # if source_filter:
        #     docs = [doc for doc in docs if doc.meta["source"] in source_filter]

        # context = " ".join([doc.content for doc in docs])
        # context = context[:500] + "..." if len(context) > 500 else context
        sources = list(set(doc.meta["source"] for doc in docs))
        response = summarizer.generate_summary(context, sources, prompt)

        print("\n\nQuery:", query)
        print("Prompt:", f"{', '.join(sources)}: {prompt}" if sources else prompt)
        print("\nContext (truncated):", context)
        print("\nResponse:", response)

if __name__ == "__main__":
    main()