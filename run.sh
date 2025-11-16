#!/bin/bash
source venv/bin/activate
uvicorn app.server:app --host 0.0.0.0 --port 8001 &