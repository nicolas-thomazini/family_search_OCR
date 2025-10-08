from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List
import os
import shutil
from datetime import datetime
import logging
import asyncio

from app.database import get_db, Document, OCRResult
from app.models.schemas import Document as DocumentSchema, OCRResponse
from app.services.ocr_service import ocr_service
from app.services.ai_service import ai_service
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/upload", response_model=DocumentSchema)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Faz upload de um documento e processa automaticamente com OCR
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Apenas arquivos de imagem são aceitos")
    
    if file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="Arquivo muito grande")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    document = Document(
        filename=file.filename,
        original_path=file_path,
        status="uploaded"
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    
    logger.info(f"Documento enviado: {document.id} - {filename}")
    
    # Processar automaticamente com OCR
    try:
        logger.info(f"Iniciando processamento automático do documento {document.id}")
        
        document.status = "processing"
        db.commit()
        
        ocr_result = ocr_service.extract_text(document.original_path, force_reprocess=True)
        
        existing_results = db.query(OCRResult).filter(OCRResult.document_id == document.id).all()
        for result in existing_results:
            db.delete(result)
        
        db_result = OCRResult(
            document_id=document.id,
            text_extracted=ocr_result['text'],
            confidence_score=ocr_result['confidence']
        )
        db.add(db_result)
        
        document.status = "completed"
        document.processed_path = ocr_result.get('preprocessed_path')
        db.commit()
        
        logger.info(f"Processamento automático concluído para documento {document.id} com confiança {ocr_result['confidence']:.2f}")
        
    except Exception as e:
        logger.error(f"Erro no processamento automático do documento {document.id}: {str(e)}")
        document.status = "error"
        db.commit()
        # Não falha o upload, apenas registra o erro
    
    return document

@router.post("/{document_id}/process", response_model=OCRResponse)
async def process_document(
    document_id: int,
    force_reprocess: bool = True,
    db: Session = Depends(get_db)
):
    """
    Processa um documento com OCR
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Documento não encontrado")
    
    try:
        logger.info(f"Iniciando processamento do documento {document_id}")
        
        if not os.path.exists(document.original_path):
            raise HTTPException(status_code=404, detail="Arquivo original não encontrado")
        
        document.status = "processing"
        db.commit()
        
        ocr_result = ocr_service.extract_text(document.original_path, force_reprocess=force_reprocess)
        
        existing_results = db.query(OCRResult).filter(OCRResult.document_id == document.id).all()
        for result in existing_results:
            db.delete(result)
        
        db_result = OCRResult(
            document_id=document.id,
            text_extracted=ocr_result['text'],
            confidence_score=ocr_result['confidence']
        )
        db.add(db_result)
        
        document.status = "completed"
        document.processed_path = ocr_result.get('preprocessed_path')
        db.commit()
        
        await asyncio.sleep(0.1)
        
        logger.info(f"Processamento concluído para documento {document_id} com confiança {ocr_result['confidence']:.2f}")
        
        try:
            ocr_service.clear_cache(keep_recent=False)
            logger.info("Cache limpo após processamento bem-sucedido")
        except Exception as e:
            logger.warning(f"Erro ao limpar cache após processamento: {str(e)}")
        
        return OCRResponse(
            text=ocr_result['text'],
            confidence=ocr_result['confidence'],
            document_id=document.id
        )
        
    except Exception as e:
        logger.error(f"Erro no processamento do documento {document_id}: {str(e)}")
        document.status = "error"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")

@router.post("/{document_id}/reprocess", response_model=OCRResponse)
async def reprocess_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Força o reprocessamento de um documento
    """
    return await process_document(document_id, force_reprocess=True, db=db)

@router.get("/", response_model=None)
async def list_documents(
    page: int = 1,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Lista todos os documentos (paginado)
    """
    total = db.query(Document).count()
    documents = db.query(Document).offset((page - 1) * limit).limit(limit).all()
    total_pages = (total + limit - 1) // limit if limit else 1
    items = []

    for doc in documents:
        ocr_result = None
        if doc.ocr_results:
            ocr_result = sorted(doc.ocr_results, key=lambda x: x.processing_date or datetime.min, reverse=True)[0]
        items.append({
            'id': str(doc.id),
            'filename': doc.filename,
            'status': doc.status.lower() if doc.status else None,
            'original_path': doc.original_path,
            'processed_path': getattr(doc, 'processed_path', None),
            'upload_date': doc.upload_date.isoformat() if doc.upload_date else None,
            'confidence_score': ocr_result.confidence_score if ocr_result else None,
            'original_text': ocr_result.text_extracted if ocr_result else None,
        })
    return {
        'items': items,
        'total': total,
        'page': page,
        'per_page': limit,
        'total_pages': total_pages
    }

@router.get("/{document_id}", response_model=DocumentSchema)
async def get_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Obtém um documento específico
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Documento não encontrado")
    return document

@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Remove um documento
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Documento não encontrado")
    
    if os.path.exists(document.original_path):
        os.remove(document.original_path)
        logger.info(f"Arquivo original removido: {document.original_path}")
    
    if document.processed_path and os.path.exists(document.processed_path):
        os.remove(document.processed_path)
        logger.info(f"Arquivo processado removido: {document.processed_path}")
    
    db.delete(document)
    db.commit()
    
    logger.info(f"Documento {document_id} removido com sucesso")
    return {"message": "Documento removido com sucesso"}

@router.post("/clear-cache")
async def clear_processing_cache():
    """
    Limpa o cache de processamento
    """
    try:
        ocr_service.clear_cache()
        logger.info("Cache de processamento limpo com sucesso")
        return {"message": "Cache limpo com sucesso"}
    except Exception as e:
        logger.error(f"Erro ao limpar cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao limpar cache: {str(e)}") 