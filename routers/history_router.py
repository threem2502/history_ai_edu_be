from fastapi import APIRouter
from schemas.history_qa import HistoryQARequest, HistoryQAResponse
from services.history_service import answer_history_question

router = APIRouter()

@router.post(
    "",
    response_model=HistoryQAResponse,
    summary="Hỏi đáp lịch sử",
    description="Đưa vào câu hỏi lịch sử và nhận câu trả lời ngắn gọn, có mốc thời gian."
)
async def qa_history(req: HistoryQARequest):
    return await answer_history_question(req)
