import os
from uuid import uuid4

from core.config import settings
from schemas.pdf_qa import (
    PDFUploadResponse,
    PDFQARequest,
    PDFQAResponse,
)
from core.llm_client import llm_client

# In-memory index rất đơn giản (demo)
# pdf_store[pdf_id] = {"path": "/abs/path/file.pdf", "text": "full text extracted ..."}
pdf_store = {}

def extract_text_from_pdf(file_path: str) -> str:
    """
    TODO: dùng PyPDF2 / pypdf để bóc text thực.
    Ở đây mock cứng.
    """
    return f"[MOCK TEXT from {os.path.basename(file_path)}] Nội dung PDF giả lập."

async def save_pdf_and_index(file_bytes: bytes, filename: str) -> PDFUploadResponse:
    save_dir = settings.UPLOAD_DIR_PDF
    os.makedirs(save_dir, exist_ok=True)

    pdf_id = str(uuid4())
    save_path = os.path.join(save_dir, f"{pdf_id}_{filename}")

    with open(save_path, "wb") as f:
        f.write(file_bytes)

    # Trích xuất text (mock)
    text_content = extract_text_from_pdf(save_path)

    pdf_store[pdf_id] = {
        "path": save_path,
        "text": text_content,
    }

    return PDFUploadResponse(pdf_id=pdf_id)

async def answer_question_from_pdf(req: PDFQARequest) -> PDFQAResponse:
    if req.pdf_id not in pdf_store:
        return PDFQAResponse(
            answer="Không tìm thấy PDF. Vui lòng upload lại.",
            summary_scope="N/A"
        )

    pdf_text = pdf_store[req.pdf_id]["text"]

    messages = []
    messages.append(f"[Tài liệu PDF trích đoạn] {pdf_text[:2000]}")  # tránh quá dài
    if req.conversation_context:
        for turn in req.conversation_context:
            messages.append(f"[Context trước đó] {turn}")
    messages.append(f"[Câu hỏi] {req.question}")

    llm_answer = await llm_client.generate_answer(
        system_prompt="Bạn là trợ lý tóm tắt tài liệu PDF. Trả lời chỉ dựa trên nội dung PDF.",
        user_messages=messages
    )

    # MOCK: không có thông tin trang thật, nên trả 'unknown pages'
    return PDFQAResponse(
        answer=llm_answer,
        summary_scope="trích nội dung tổng quát (demo)"
    )
