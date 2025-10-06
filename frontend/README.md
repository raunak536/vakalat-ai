# Legal Research AI - Frontend

A modern Dash-based web interface for legal document search and analysis.

## Features

- **Text Query Search**: Search for relevant legal cases using natural language queries
- **Document Upload Search**: Upload documents (PDF, DOCX, TXT) to find similar cases
- **Modern UI**: Clean, responsive design with Bootstrap components
- **Real-time Results**: Instant search results with relevance scores and document previews
- **Suggested Queries**: Pre-defined query suggestions for quick access

## Quick Start

### 1. Install Dependencies

```bash
cd frontend
pip install -r requirements.txt
```

### 2. Start the Backend Server

Make sure the backend server is running on `http://localhost:8000`:

```bash
cd ../backend
python run_server.py
```

### 3. Run the Frontend

```bash
python app.py
```

The frontend will be available at `http://localhost:8050`

## Usage

### Text Query Search
1. Enter your legal question in the search box
2. Click the search button or press Enter
3. View relevant legal cases with relevance scores

### Document Upload Search
1. Click "Attach Files" to upload a document
2. Click "Search Document" to find similar cases
3. View results based on document content similarity

### Suggested Queries
- Click on the suggested query buttons for quick searches
- Examples include "Wrongful tenant eviction cases in Michigan" and "What is the adverse domination doctrine"

## API Integration

The frontend integrates with two backend services:

1. **Query Search**: `POST /search/query` - Text-based search
2. **Document Search**: `POST /search/document` - Document upload and search

## Styling

The interface uses:
- **Bootstrap 5** for responsive design
- **Font Awesome** for icons
- **Custom CSS** for modern styling
- **Dash Bootstrap Components** for UI components

## Development

To run in development mode with auto-reload:

```bash
python app.py
```

The app will automatically reload when you make changes to the code.
