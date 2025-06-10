Remus
===============

This project was developed in collaboration with [Noor Fathima](https://github.com/NoorFathima14). The project was worked on for a hackathon initially from a single computer, hence the fork.

A multimodal chatbot designed to simplify file querying, processing, and intelligent response generation. Remus supports diverse file types (PDF, CSV, DOCX, PNG, JPEG, MP4), processes text, images, and videos, and enhances responses with web search validation and caching for performance. 

Overview
--------

Remus is a web-based application that allows users to upload, process, and query files in various formats, including PDFs, images, and videos. It leverages multimodal processing to handle text, images, and videos, and uses natural language querying to provide intelligent, Markdown-formatted responses with syntax highlighting. The chatbot validates responses using web search (Google Custom Search API) and caches them with Redis for faster retrieval of repeated queries. It’s designed for academic research, content analysis, lecture summarization, and more, targeting students, researchers, educators, and professionals.

Key Features
------------

*   **Diverse File Support**: Processes PDFs, images, videos (PDF, CSV, DOCX, PNG, JPEG, MP4).
    
*   **Efficient Querying**: Replaces manual searching with fast, automated extraction.
    
*   **Multimodal Processing**: Handles multiple media types with intelligent querying.
    
*   **Web Search Validation**: Enhances responses with Google Custom Search API.
    
*   **Response Caching**: Caches query responses in Redis for instant retrieval of previously used queries.
    
*   **Markdown Rendering**: Displays responses with syntax-highlighted code blocks using React-Markdown and React-Syntax-Highlighter.
    
*   **File Management**: Upload, list, and delete files with ease.
    
*   **User-Friendly Interface**: Straightforward and intuitive design built with React.
    
*   **Optimization**: Parallelizes file processing and web search, saving 5.61s (from 25.52s to 19.91s) for efficient resource utilization and faster responses.
    
*   **Containerization**: Uses Docker for seamless deployment and scalability across services.
    

Technologies Used
-----------------

*   **Frontend**: React, Vite, React-Markdown, React-Syntax-Highlighter; Nginx for serving.
    
*   **Backend**: FastAPI, Tesseract OCR; Uvicorn 
    
*   **Database & Caching**: ChromaDB for vector storage; Redis container for caching
    
*   **Containerization**: Docker/Dockerfile
    
*   **Web Search**: Google Custom Search API for response validation and enhancement.
    
*   **LLM**: Mistral-7B via Ollama container for local inference, enhancing multimodal processing.
    

Installation
------------

### Prerequisites

*   Python 3.8+
    
*   Node.js 16+
    
*   Docker and Docker Compose
    
*   Git
    

### Steps
    
1.  **Set Up Environment Variables**:
    
    *   `cp .env.example .env`
        
    *   Update .env with your API keys and configuration (e.g., Google Custom Search API key, Redis settings).
        
2.  `cd backend pip install -r requirements.txt`
    
    
3.  **Run with Docker**:
    
    *   Ensure Docker is running.

    ```sh
    docker-compose up -d
    ```
        
    *   This will start the backend (FastAPI), frontend (React), Redis, and Ollama containers.
        
    
Contact
-------

*   **Authors**: Noor Fathima | Dinesh Veluswmay
    
*   **GitHub**: [NoorFathima14](https://github.com/NoorFathima14) | [Dinesh Veluswamy](https://github.com/thatcatfromspace)

For questions, suggestions, or collaboration, feel free to open an issue or reach out directly!
