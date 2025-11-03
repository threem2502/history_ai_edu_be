# app.py
import os
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from google import genai

# =================== Cấu hình chung ===================
os.environ['GEMINI_API_KEY'] = 'AIzaSyCyiYySkfCmwn3US6C99Csu91ZYAzA3NKo'
MODEL_NAME = "gemini-2.5-flash"
USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"  # Bật mock khi free plan bị chặn outbound

# CORS cho frontend của bạn
ALLOWED_ORIGINS = [
    "https://threem2502.github.io",
    "https://threem2502.github.io/history_ai_edu/*",
]

# =================== Flask app ===================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS, "supports_credentials": True}})

@app.get("/")
def root():
    return jsonify({
        "name": "AI EDU Backend (Flask)",
        "desc": "API Hỏi đáp lịch sử / Nhận diện hình ảnh / Đọc hiểu PDF lịch sử",
        "version": "1.0.0",
        "ok": True
    })

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

# =================== Gemini (SDK mới) – lazy init ===================
_client = None  # khởi tạo trễ để tránh lỗi init sớm trong WSGI

def get_client():
    """Khởi tạo client mỗi khi cần; log ngắn vào error log của PythonAnywhere."""
    global _client
    if _client is not None:
        return _client
    try:
        _client = genai.Client() 
        print(">>> GENAI: client initialized. KEY_PRESENT=", bool(os.getenv("GEMINI_API_KEY")))
        return _client
    except Exception as e:
        import traceback
        print(">>> GENAI INIT FAILED:", repr(e))
        traceback.print_exc()
        _client = None
        return None

def ensure_client():
    """Trả về (resp, status) nếu lỗi; ngược lại trả None."""
    if not os.getenv("GEMINI_API_KEY"):
        return jsonify({"ok": False, "error": "Thiếu GEMINI_API_KEY trong env."}), 503
    cli = get_client()
    if cli is None:
        return jsonify({"ok": False, "error": "Không khởi tạo được Gemini client."}), 503
    return None

# =================== Tiện ích chung ===================
def get_is_stream():
    """Đọc tham số isStream từ query, JSON hoặc form."""
    qs = request.args.get("isStream")
    if qs is not None:
        return qs.lower() == "true"
    if request.is_json:
        v = (request.get_json(silent=True) or {}).get("isStream")
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() == "true"
    v = request.form.get("isStream")
    if v is not None:
        return v.lower() == "true"
    return True  # mặc định stream

def sse_event(text: str) -> str:
    return f"data: {text}\n\n"

def maybe_mock(answer_text: str, stream: bool):
    """Nếu USE_MOCK=true, trả về mock response để test UI khi bị chặn outbound."""
    if not USE_MOCK:
        return None
    if stream:
        def generate():
            yield sse_event(answer_text)
            yield "event: done\ndata: [END]\n\n"
        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    else:
        return jsonify({"ok": True, "answer": f"[MOCK] {answer_text}"})

# =================== 1) Hỏi đáp lịch sử ===================
@app.post("/history-chat")
def history_chat():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or payload.get("q") or "").strip()
    is_stream = get_is_stream()

    if not question:
        return jsonify({"ok": False, "error": "Thiếu trường 'question'"}), 400

    # Ràng buộc an toàn + tiếng Việt
    system_prompt = (
        "Bạn là một trợ lý AI chuyên về lịch sử Việt Nam và thế giới. "
        "TRẢ LỜI BẰNG TIẾNG VIỆT, rõ ràng, ngắn gọn, chính xác, phù hợp người học. "
        "Tuyệt đối không xuyên tạc lịch sử, không xúc phạm, không cổ vũ bạo lực, "
        "không bình luận/chống phá chính trị; luôn khách quan, tôn trọng và đúng mực."
    )
    contents = [system_prompt, question]

    # Đảm bảo client
    err = ensure_client()
    if err:
        mock = maybe_mock("Backend chạy OK. Đây là câu trả lời mẫu cho câu hỏi lịch sử.", is_stream)
        return mock or err

    cli = get_client()

    if is_stream:
        def generate():
            try:
                for chunk in cli.models.generate_content(model=MODEL_NAME, contents=contents, stream=True):
                    if getattr(chunk, "text", ""):
                        yield sse_event(chunk.text)
                yield "event: done\ndata: [END]\n\n"
            except Exception as e:
                yield sse_event(f"[LỖI] {str(e)}")
                yield "event: done\ndata: [END]\n\n"
        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    else:
        try:
            resp = cli.models.generate_content(model=MODEL_NAME, contents=contents)
            return jsonify({"ok": True, "answer": getattr(resp, "text", None)})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

# =================== 2) Nhận diện hình ảnh lịch sử ===================
@app.post("/vision-detect")
def vision_detect():
    file = request.files.get("image") or request.files.get("file")
    if not file:
        return jsonify({"ok": False, "error": "Thiếu file hình ảnh (image)"}), 400

    prompt = (
        "Phân tích hình ảnh BẰNG TIẾNG VIỆT. Nếu là danh nhân/nhân vật/đồ vật lịch sử, "
        "nêu tên, thời kỳ, vai trò/bối cảnh. Nếu không chắc, nêu các khả năng hợp lý và mức độ chắc chắn. "
        "Không xuyên tạc, không bịa đặt, không bình luận/chống phá chính trị; chỉ mô tả khách quan."
    )
    is_stream = get_is_stream()

    img_bytes = file.read()
    mime = file.mimetype or "image/jpeg"
    contents = [prompt, {"mime_type": mime, "data": img_bytes}]

    err = ensure_client()
    if err:
        mock = maybe_mock("Có thể là ảnh về một nhân vật/di vật lịch sử. (Dữ liệu giả lập)", is_stream)
        return mock or err

    cli = get_client()

    if is_stream:
        def generate():
            try:
                for chunk in cli.models.generate_content(model=MODEL_NAME, contents=contents, stream=True):
                    if getattr(chunk, "text", ""):
                        yield sse_event(chunk.text)
                yield "event: done\ndata: [END]\n\n"
            except Exception as e:
                yield sse_event(f"[LỖI] {str(e)}")
                yield "event: done\ndata: [END]\n\n"
        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    else:
        try:
            resp = cli.models.generate_content(model=MODEL_NAME, contents=contents)
            return jsonify({"ok": True, "answer": getattr(resp, "text", None)})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

# =================== 3) Hỏi đáp theo PDF lịch sử ===================
@app.post("/pdf-qa")
def pdf_qa():
    pdf_file = request.files.get("file") or request.files.get("pdf")
    if not pdf_file:
        return jsonify({"ok": False, "error": "Thiếu file PDF"}), 400

    question = (
        request.form.get("question")
        or request.args.get("question")
        or (request.get_json(silent=True) or {}).get("question")
    )
    if not question:
        return jsonify({"ok": False, "error": "Thiếu câu hỏi"}), 400

    is_stream = get_is_stream()

    pdf_bytes = pdf_file.read()
    contents = [
        ("Bạn là trợ lý đọc hiểu tài liệu lịch sử, TRẢ LỜI BẰNG TIẾNG VIỆT, "
         "ngắn gọn, chính xác; không xuyên tạc, không thêm ý kiến cá nhân, không bình luận/chống phá chính trị. "
         "Nếu có thể, trích dẫn trang (ví dụ: 'theo trang 5')."),
        {"mime_type": "application/pdf", "data": pdf_bytes},
        question
    ]

    err = ensure_client()
    if err:
        mock = maybe_mock("Tóm tắt PDF và trả lời câu hỏi. (Dữ liệu giả lập)", is_stream)
        return mock or err

    cli = get_client()

    if is_stream:
        def generate():
            try:
                for chunk in cli.models.generate_content(model=MODEL_NAME, contents=contents, stream=True):
                    if getattr(chunk, "text", ""):
                        yield sse_event(chunk.text)
                yield "event: done\ndata: [END]\n\n"
            except Exception as e:
                yield sse_event(f"[LỖI] {str(e)}")
                yield "event: done\ndata: [END]\n\n"
        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    else:
        try:
            resp = cli.models.generate_content(model=MODEL_NAME, contents=contents)
            return jsonify({"ok": True, "answer": getattr(resp, "text", None)})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

# =================== Health nâng cao (chẩn đoán cấu hình) ===================
@app.get("/health-plus")
def health_plus():
    import sys, pkgutil
    cli = get_client()
    return jsonify({
        "ok": True,
        "gemini_key_present": bool(os.getenv("GEMINI_API_KEY")),
        "client_ok": cli is not None,
        "has_google_genai": pkgutil.find_loader("google.genai") is not None,
        "python_executable": sys.executable,
        "use_mock": USE_MOCK
    })

# =================== Main (chạy local) ===================
# if __name__ == "__main__":
#     # Khi chạy local: export GEMINI_API_KEY=... ; python app.py
#     app.run(host="0.0.0.0", port=8000, debug=True)
