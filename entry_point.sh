#!/bin/bash

echo "Starting FastAPI backend..."
uvicorn backend.main:app --host 0.0.0.0 --port 6000 --reload &

echo "Starting Gradio frontend..."
python frontend/app.py
