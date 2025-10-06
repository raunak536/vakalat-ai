# Vakalat AI Chatbot API

A FastAPI-based chatbot for searching legal documents using vector similarity.

## Features

- **Text Query Search**: Search for relevant legal cases using natural language queries
- **Document Upload Search**: Upload documents (PDF, DOCX, TXT) to find similar cases
- **Vector Similarity**: Uses ChromaDB with Gemini embeddings for semantic search
- **Rich Metadata**: Returns case title, date, relevance score, and match reasoning

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file in the backend directory:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run the Server

```bash
python run_server.py
```

The API will be available at `http://localhost:8000`

### 4. Test the API

```bash
python test_api.py
```

## API Endpoints

### Health Check
- **GET** `/health` - Check API health and service status

### Text Query Search
- **POST** `/search/query`
- **Body**: `{"query": "your search query", "top_k": 5}`
- **Response**: List of relevant cases with metadata

### Document Upload Search
- **POST** `/search/document`
- **Form Data**: 
  - `file`: Document file (PDF, DOCX, TXT)
  - `top_k`: Number of results (default: 5)
- **Response**: List of similar cases

## Response Format

Each search result includes:

```json
{
  "case_title": "Case Name",
  "case_date": "2023-01-01",
  "relevance_score": 0.85,
  "reason_for_match": "Semantic similarity based on query...",
  "court": "Supreme Court of India",
  "document_id": "12345",
  "document_preview": "First 200 characters of the document..."
}
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Key FastAPI Concepts Used

1. **Route Decorators**: `@app.post()`, `@app.get()` define HTTP endpoints
2. **File Uploads**: `UploadFile` handles multipart file uploads
3. **Request Bodies**: Automatic JSON parsing for POST requests
4. **Dependency Injection**: Services initialized once at startup
5. **CORS Middleware**: Enables cross-origin requests for frontend integration
6. **Error Handling**: HTTPException for proper error responses
7. **Startup Events**: `@app.on_event("startup")` for service initialization

## Development

- The server runs with auto-reload enabled for development
- Logs are configured to show INFO level messages
- CORS is enabled for all origins (configure for production)
