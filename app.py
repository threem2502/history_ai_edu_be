from fastapi import FastAPI
import uvicorn

from routers.history_router import router as history_router
from routers.vision_router import router as vision_router
from routers.pdf_router import router as pdf_router

app = FastAPI(
    title="AI EDU Backend",
    description="API Hỏi đáp lịch sử / Vision / PDF QA",
    version="1.0.0",
)

# Đăng ký router
app.include_router(history_router, prefix="/qa/history", tags=["History Q&A"])
app.include_router(vision_router, prefix="/vision", tags=["Vision"])
app.include_router(pdf_router, prefix="/qa/pdf", tags=["PDF Q&A"])


if __name__ == "__main__":
    # Render chạy `python app.py` -> bạn không cần thêm host/port thủ công.
    # Nếu cần expose 0.0.0.0 (ví dụ Render), ta dùng:
    uvicorn.run(app, host="0.0.0.0", port=8000)
