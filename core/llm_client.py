from typing import List
from core.config import settings

class LLMClient:
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        # TODO: init SDK thật ở đây (vd google-generativeai, openai, ollama client...)

    async def generate_answer(self, system_prompt: str, user_messages: List[str]) -> str:
        """
        system_prompt: hướng dẫn vai trò
        user_messages: list các câu hỏi/ngữ cảnh
        return: câu trả lời của LLM
        """
        # MOCK: ghép nội dung để demo
        joined_context = "\n".join(user_messages)
        return f"[MOCK:{self.model_name}] {system_prompt}\n---\n{joined_context}"

llm_client = LLMClient(
    api_key=settings.LLM_API_KEY,
    model_name=settings.LLM_MODEL_NAME,
)
