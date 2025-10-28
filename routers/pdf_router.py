from fastapi import APIRouter, UploadFile, File, HTTPException
from schemas.pdf_qa import PDFUploadResponse, PDFQARequest, PDFQAResponse
from services.pdf_service import save_pdf_and_index, answer_question_from_pdf

router = APIRouter()

@router.post(
    "/upload",
    response_model=PDFUploadResponse,
    summary="Upload PDF",
    description="Upload file PDF. Server trả về pdf_id để dùng ở bước hỏi đáp."
)
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf"]:
        raise HTTPException(status_code=400, detail="File phải là PDF.")
    file_bytes = await file.read()
    result = await save_pdf_and_index(file_bytes, file.filename)
    return result


@router.post(
    "/ask",
    response_model=PDFQAResponse,
    summary="Hỏi đáp PDF",
    description="Gửi câu hỏi kèm pdf_id để hỏi nội dung trong PDF đã upload."
)
async def ask_pdf(req: PDFQARequest):
    return await answer_question_from_pdf(req)
