from pydantic import BaseModel, Field
from typing import Optional, List

class PDFUploadResponse(BaseModel):
    pdf_id: str = Field(..., description="ID đại diện cho file PDF đã lưu")

class PDFQARequest(BaseModel):
    pdf_id: str = Field(..., description="ID file PDF đã upload trước đó")
    question: str = Field(..., description="Câu hỏi của người dùng về nội dung tài liệu PDF")
    conversation_context: Optional[List[str]] = Field(
        default=None,
        description="(Tuỳ chọn) lịch sử hội thoại trước đó để giữ context theo tài liệu"
    )

class PDFQAResponse(BaseModel):
    answer: str
    summary_scope: str = Field(
        "...",
        description="Mô tả phạm vi trích dẫn (ví dụ: trang 2-3)"
    )
