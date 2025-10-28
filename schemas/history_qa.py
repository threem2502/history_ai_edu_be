from pydantic import BaseModel, Field
from typing import List, Optional

class HistoryQARequest(BaseModel):
    question: str = Field(..., description="Câu hỏi của người dùng về lịch sử")
    period_hint: Optional[str] = Field(
        None,
        description="Gợi ý phạm vi thời gian/vị trí (ví dụ: 'Chiến tranh thế giới thứ 2, châu Âu')."
    )
    conversation_context: Optional[List[str]] = Field(
        default=None,
        description="(Tuỳ chọn) lịch sử hội thoại trước đó để giữ ngữ cảnh"
    )

class HistoryQAResponse(BaseModel):
    answer: str
    source_note: str = Field(
        "...",
        description="Ghi chú nguồn hoặc cách trả lời (có thể để bạn chèn citation sau này)"
    )
