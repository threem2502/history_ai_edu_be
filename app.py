import os
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from google import genai


# =================== Cấu hình Gemini (SDK mới) ===================
try:
    # Client tự đọc GEMINI_API_KEY từ biến môi trường
    client = genai.Client()
    # Kiểm tra nhanh có key không
    _key_present = bool(os.getenv("GEMINI_API_KEY"))
except Exception:
    client = None
    _key_present = False

MODEL_NAME = "gemini-2.5-flash"

# =================== Khởi tạo Flask ===================
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://threem2502.github.io",
            "https://threem2502.github.io/history_ai_edu/*",
        ],
        "supports_credentials": True
    }
})

@app.route("/")
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


@app.get("/health-plus")
def health_plus():
    import sys, os, pkgutil
    return jsonify({
        "ok": True,
        "gemini_key_present": bool(os.getenv("GEMINI_API_KEY")),
        "python_executable": sys.executable,
        "cwd": os.getcwd(),
        "has_google_genai": pkgutil.find_loader("google.genai") is not None
    })


# =================== Tiện ích ===================
def ensure_client():
    if client is None or not _key_present:
        return jsonify({
            "ok": False,
            "error": "Gemini model chưa được cấu hình. Vui lòng thiết lập GEMINI_API_KEY."
        }), 503
    return None

def get_is_stream():
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
    return True

def sse_event(text: str):
    return f"data: {text}\n\n"

# =================== 1) history-chat ===================
@app.post("/history-chat")
def history_chat():
    err = ensure_client()
    if err:
        return err

    payload = request.get_json(silent=True) or {}
    question = payload.get("question") or payload.get("q") or ""
    is_stream = get_is_stream()

    if not question.strip():
        return jsonify({"ok": False, "error": "Thiếu trường 'question'"}), 400

    system_prompt = (
        "Bạn là một trợ lý AI chuyên về lịch sử Việt Nam và thế giới. "
        "Trả lời BẰNG TIẾNG VIỆT, ngắn gọn, chính xác, dễ hiểu. "
        "Tuyệt đối không xuyên tạc lịch sử, không xúc phạm, không bình luận chính trị; "
        "luôn khách quan và tôn trọng chuẩn mực đạo đức."
    )

    contents = [system_prompt, question]

    if is_stream:
        def generate():
            try:
                for chunk in client.models.generate_content(
                    model=MODEL_NAME,
                    contents=contents,
                    stream=True
                ):
                    if hasattr(chunk, "text") and chunk.text:
                        yield sse_event(chunk.text)
                yield "event: done\ndata: [END]\n\n"
            except Exception as e:
                yield sse_event(f"[LỖI] {str(e)}")
                yield "event: done\ndata: [END]\n\n"

        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    else:
        try:
            resp = client.models.generate_content(model=MODEL_NAME, contents=contents)
            return jsonify({"ok": True, "answer": getattr(resp, "text", None)})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

# =================== 2) vision-detect ===================
@app.post("/vision-detect")
def vision_detect():
    err = ensure_client()
    if err:
        return err

    file = request.files.get("image") or request.files.get("file")
    if not file:
        return jsonify({"ok": False, "error": "Thiếu file hình ảnh (image)"}), 400

    prompt = (
        "Phân tích hình ảnh BẰNG TIẾNG VIỆT. "
        "Nếu là danh nhân/nhân vật/đồ vật lịch sử, nêu tên, thời kỳ, vai trò/bối cảnh. "
        "Nếu không chắc, nêu các khả năng hợp lý và ghi rõ mức độ chắc chắn. "
        "Không xuyên tạc, không bịa đặt, không bình luận chính trị."
    )
    is_stream = get_is_stream()

    img_bytes = file.read()
    mime = file.mimetype or "image/jpeg"

    contents = [
        prompt,
        {"mime_type": mime, "data": img_bytes}
    ]

    if is_stream:
        def generate():
            try:
                for chunk in client.models.generate_content(
                    model=MODEL_NAME,
                    contents=contents,
                    stream=True
                ):
                    if hasattr(chunk, "text") and chunk.text:
                        yield sse_event(chunk.text)
                yield "event: done\ndata: [END]\n\n"
            except Exception as e:
                yield sse_event(f"[LỖI] {str(e)}")
                yield "event: done\ndata: [END]\n\n"

        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    else:
        try:
            resp = client.models.generate_content(model=MODEL_NAME, contents=contents)
            return jsonify({"ok": True, "answer": getattr(resp, "text", None)})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

# =================== 3) pdf-qa ===================
@app.post("/pdf-qa")
def pdf_qa():
    err = ensure_client()
    if err:
        return err

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
        ("Bạn là trợ lý đọc hiểu tài liệu lịch sử, trả lời BẰNG TIẾNG VIỆT, "
         "ngắn gọn, chính xác; không xuyên tạc, không thêm ý kiến cá nhân, không bàn chính trị. "
         "Nếu có thể, hãy trích dẫn trang (ví dụ: 'theo trang 5')."),
        {"mime_type": "application/pdf", "data": pdf_bytes},
        question
    ]

    if is_stream:
        def generate():
            try:
                for chunk in client.models.generate_content(
                    model=MODEL_NAME,
                    contents=contents,
                    stream=True
                ):
                    if hasattr(chunk, "text") and chunk.text:
                        yield sse_event(chunk.text)
                yield "event: done\ndata: [END]\n\n"
            except Exception as e:
                yield sse_event(f"[LỖI] {str(e)}")
                yield "event: done\ndata: [END]\n\n"

        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    else:
        try:
            resp = client.models.generate_content(model=MODEL_NAME, contents=contents)
            return jsonify({"ok": True, "answer": getattr(resp, "text", None)})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

# =============== (Không cần app.run khi WSGI) ===============
# if __name__ == "__main__":
#     app.run(debug=True)
