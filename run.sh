#!/bin/bash
cd backend
# Start Docker services
docker-compose -f docker-compose.backend.yml up -d

# Run the FastAPI application
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
