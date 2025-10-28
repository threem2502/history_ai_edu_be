from core.llm_client import llm_client
from schemas.history_qa import HistoryQARequest, HistoryQAResponse

SYSTEM_PROMPT_HISTORY = (
    "Bạn là trợ lý lịch sử. Trả lời chính xác, mốc thời gian rõ ràng, "
    "giải thích ngắn gọn, dễ hiểu cho học sinh phổ thông."
)

async def answer_history_question(payload: HistoryQARequest) -> HistoryQAResponse:
    # Chuẩn hoá context gửi LLM
    user_messages = []
    if payload.period_hint:
        user_messages.append(f"[Bối cảnh thời kỳ] {payload.period_hint}")
    if payload.conversation_context:
        for turn in payload.conversation_context:
            user_messages.append(f"[Context trước đó] {turn}")
    user_messages.append(f"[Câu hỏi] {payload.question}")

    llm_raw_answer = await llm_client.generate_answer(
        system_prompt=SYSTEM_PROMPT_HISTORY,
        user_messages=user_messages
    )

    return HistoryQAResponse(
        answer=llm_raw_answer,
        source_note="Trả lời dựa trên mô hình LLM và kiến thức lịch sử tổng quát."
    )
