from fastapi import FastAPI
import uvicorn


app = FastAPI(
    title="AI EDU Backend",
    description="API Hỏi đáp lịch sử / Vision / PDF QA",
    version="1.0.0",
)



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
