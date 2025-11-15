from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.text2sql import build_text2sql
from app.sql_streamer import stream_select_query

app = FastAPI()

engine = build_text2sql()  # твой генератор SQL

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Модель запроса
class TextRequest(BaseModel):
    text: str

@app.post("/process-text")
def process_text_stream(req: TextRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' is required")
    
    print("Received text:", text)

    sql = engine.generate(text)
    print("Generated SQL:", sql)
    if not sql:
        raise HTTPException(status_code=400, detail="Failed to generate SQL from the text")

    if not sql.strip().upper().startswith("SELECT"):
        raise HTTPException(status_code=400, detail="Generated SQL is not a SELECT query.")

    return StreamingResponse(stream_select_query(sql), media_type="application/json")

@app.post("/stream-sql")
def stream_sql(req: TextRequest):
    """Стриминг SELECT запроса напрямую (для отладки)."""
    q = req.text.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Field 'text' is required")
    try:
        return StreamingResponse(stream_select_query(q), media_type="application/json")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
