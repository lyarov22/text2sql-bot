from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from app.text2sql import build_text2sql
from app.sql_streamer import stream_select_query
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для теста, позже можно ограничить
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = build_text2sql()  # твой генератор SQL

@app.get("/process-text")
def process_text_stream(text: str = Query(..., description="Текст для обработки")):
    """
    Генерация SQL из текста и отдача результата батчами.
    """
    try:
        # Генерация SQL
        sql = engine.generate(text)
        print(f"Generated SQL: {sql}")

        # Проверка, что это SELECT
        if not sql.strip().upper().startswith("SELECT"):
            raise HTTPException(status_code=400, detail="Generated SQL is not a SELECT query.")

        # Возвращаем стриминг
        return StreamingResponse(stream_select_query(sql), media_type="application/json")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stream-sql")
def stream_sql(q: str = Query(..., description="SQL SELECT запрос")):
    """Стриминг SELECT запроса напрямую (для отладки)."""
    try:
        return StreamingResponse(stream_select_query(q), media_type="application/json")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
