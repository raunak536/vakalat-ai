#!/usr/bin/env python3
"""
Simple script to run the FastAPI chatbot server.

This script provides an easy way to start the server with proper configuration.
"""

import uvicorn
import os
from pathlib import Path

if __name__ == "__main__":
    # Get the directory where this script is located
    backend_dir = Path(__file__).parent
    # Change to the backend directory to ensure relative paths work
    os.chdir(backend_dir)
    
    # Run the FastAPI application
    uvicorn.run(
        "app:app",  # The FastAPI app instance
        host="0.0.0.0",  # Listen on all network interfaces
        port=8000,  # Port number
        reload=True,  # Auto-reload on file changes (development mode)
        log_level="info"  # Logging level
    )
