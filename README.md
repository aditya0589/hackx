# RAG Pipeline with Gemini, FAISS, and PDF/DOCX Support

## Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your Google API key:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage
- Run `main.py` to use the CLI for document ingestion and querying.
- The codebase is modular and ready for FastAPI integration (see code comments for extension points).

## Extending to FastAPI
- Each module is designed for easy import into a FastAPI app.
- To build an API, create endpoints in a new `api.py` or similar, and reuse the pipeline logic from `rag/`. 
