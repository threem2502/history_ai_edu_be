import os
from pydantic import BaseModel

class Settings(BaseModel):
    # Khóa AI, model name... bạn có thể set qua env Render dashboard
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "dummy-key")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")

    # Thư mục lưu file tạm
    UPLOAD_DIR_PDF: str = os.getenv("UPLOAD_DIR_PDF", "storage/uploads/pdfs")
    UPLOAD_DIR_IMAGE: str = os.getenv("UPLOAD_DIR_IMAGE", "storage/uploads/images")

settings = Settings()
