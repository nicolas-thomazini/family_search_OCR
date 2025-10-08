import os
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
from starlette.responses import Response

from app.config import settings
from app.database import engine, Base, get_db
from app.api import documents, corrections, ai
from app.services.ocr_service import ocr_service
from sqlalchemy.orm import Session
from fastapi import Depends
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CORBMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Expose-Headers"] = "*"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

os.makedirs("./uploads/preprocessed", exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando aplicação Family Search OCR")
    
    Base.metadata.create_all(bind=engine)
    
    # clear cache
    try:
        ocr_service.clear_cache(keep_recent=False)  # Limpar tudo na inicialização
        logger.info("Cache limpo na inicialização")
    except Exception as e:
        logger.warning(f"Erro ao limpar cache na inicialização: {str(e)}")
    
    yield
    
    logger.info("Encerrando aplicação Family Search OCR")

app = FastAPI(
    title="Family Search OCR - AI System",
    description="Sistema de IA para análise de documentos antigos em italiano",
    version="1.0.0",
    lifespan=lifespan,
    # Configuração para upload de arquivos grandes
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(CORBMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:80", "http://127.0.0.1:80"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

app.include_router(documents.router)
app.include_router(corrections.router)
app.include_router(ai.router)

# Middleware para servir imagens com CORS
class StaticFilesWithCORS(StaticFiles):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    async def __call__(self, scope, receive, send):
        async def send_with_cors(message):
            if message["type"] == "http.response.start":
                message["headers"].extend([
                    (b"access-control-allow-origin", b"*"),
                    (b"access-control-allow-methods", b"GET, OPTIONS"),
                    (b"access-control-allow-headers", b"*"),
                    (b"x-content-type-options", b"nosniff"),
                ])
            await send(message)
        
        await super().__call__(scope, receive, send_with_cors)

app.mount(
    "/preprocessed-images",
    StaticFilesWithCORS(directory="./uploads/preprocessed"),
    name="preprocessed-images"
)

@app.get("/")
async def root():
    """
    Endpoint raiz da API
    """
    return {
        "message": "Family Search OCR - AI System",
        "version": "1.0.0",
        "description": "Sistema de IA para análise de documentos antigos em italiano",
        "endpoints": {
            "documents": "/documents",
            "corrections": "/corrections", 
            "ai": "/ai",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """
    Verificação de saúde da aplicação
    """
    return {"status": "healthy", "message": "Sistema funcionando normalmente"}

@app.get("/statistics")
async def get_statistics(db: Session = Depends(get_db)):
    """
    Obtém estatísticas gerais do sistema
    """
    try:
        from app.database import Document, OCRResult, Correction
        
        # total documents
        total_documents = db.query(Document).count()
        
        # processed documents
        processed_documents = db.query(Document).filter(Document.status == "completed").count()
        
        # documents today
        today = datetime.utcnow().date()
        documents_today = db.query(Document).filter(
            Document.upload_date >= today
        ).count()
        
        # average confidence
        ocr_results = db.query(OCRResult).all()
        if ocr_results:
            average_confidence = sum(result.confidence_score for result in ocr_results) / len(ocr_results)
        else:
            average_confidence = 0.0
        
        # total corrections
        total_corrections = db.query(Correction).count()
        
        ai_accuracy = 0.85  
        
        return {
            "total_documents": total_documents,
            "processed_documents": processed_documents,
            "documents_today": documents_today,
            "average_confidence": average_confidence,
            "total_corrections": total_corrections,
            "ai_accuracy": ai_accuracy
        }
        
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter estatísticas: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 