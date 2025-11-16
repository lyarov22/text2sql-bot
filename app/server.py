from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.text2sql import build_text2sql
from app.text2sql_local import build_text2sql_local
from app.sql_to_db import execute_sql_query
from app.models import UserQuery, FinalResponse
from app.security_validator import SecurityException

app = FastAPI()

# Инициализируем оба движка
api_engine = build_text2sql()
llm_engine = build_text2sql_local()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.post("/process-text")
async def process_text_stream(req: UserQuery):
    """Обработка запроса с использованием production контракта и поддержкой контекста"""
    query = req.natural_language_query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Field 'natural_language_query' is required")
    
    # Выбираем движок в зависимости от параметра model
    if req.model == "api":
        engine = api_engine
        print(f"Using API engine (Gemini) for user {req.user_id}")
    else:  # "llm" или по умолчанию
        engine = llm_engine
        print(f"Using LLM engine (Ollama) for user {req.user_id}")
    
    print(f"Received query from user {req.user_id}: {query}")

    try:
        final_response: FinalResponse = await engine.process_user_request(req)
        
        if final_response.metadata.get("requires_clarification", False):
            return JSONResponse(content={
                "content": final_response.content,
                "output_format": final_response.output_format,
                "data": None,
                "row_count": 0,
                "execution_time_ms": 0,
                "metadata": final_response.metadata
            })
        
        sql_query = final_response.metadata.get("sql_query", final_response.content)
        
        if not sql_query:
            raise HTTPException(status_code=400, detail="Failed to generate SQL from the query")
        
        print("Generated SQL:", sql_query)
        
        execution_result = await execute_sql_query(sql_query, query)
        
        processed_data = execution_result.data
        text_content = final_response.content
        if final_response.output_format == "text":
            text_response = await engine.format_text_response(
                query,
                execution_result.data,
                req.user_id
            )
            text_content = text_response
            processed_data = [{"text": text_response}]
        elif final_response.output_format in ["table", "graph", "diagram"]:
            processed_data = await engine.translate_column_names(
                execution_result.data,
                query,
                req.user_id
            )
            for row in processed_data:
                for key, value in row.items():
                    if isinstance(value, float):
                        row[key] = round(value, 2)
        
        response_data = {
            "content": text_content if final_response.output_format == "text" else final_response.content,
            "output_format": final_response.output_format,
            "data": processed_data,
            "row_count": len(processed_data) if final_response.output_format == "text" else execution_result.row_count,
            "execution_time_ms": execution_result.execution_time_ms,
            "metadata": {
                **final_response.metadata,
                "execution_time_ms": execution_result.execution_time_ms,
                "row_count": len(processed_data) if final_response.output_format == "text" else execution_result.row_count
            }
        }
        
        return JSONResponse(content=response_data)
        
    except SecurityException as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

class ClearHistoryRequest(BaseModel):
    user_id: str


@app.post("/clear-history")
async def clear_history(req: ClearHistoryRequest):
    """Очистка истории диалога для пользователя"""
    try:
        # Очищаем историю в обоих движках
        api_engine._clear_history(req.user_id)
        #llm_engine._clear_history(req.user_id)
        return JSONResponse(content={"message": f"History cleared for user {req.user_id}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing history: {str(e)}")
