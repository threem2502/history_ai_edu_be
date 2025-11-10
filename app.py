import os
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from google import genai
from google.genai import types

MODEL_NAME = "gemini-2.5-flash"
# =================== Khởi tạo Flask ===================
app = Flask(__name__)

ALLOWED_ORIGINS = [
    "https://threem2502.github.io",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]

CORS(app, resources={
    r"/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False, 
    }
})

try:
    client = genai.Client(api_key='AIzaSyCyiYySkfCmwn3US6C99Csu91ZYAzA3NKo')
except Exception as e:
    raise RuntimeError(f"Lỗi khởi tạo Gemini Client: {str(e)}")

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
    return True


def sse_event(text: str):
    """Tạo sự kiện SSE."""
    return f"data: {text}\n\n"


# =================== 1) history-chat ===================
@app.post("/history-chat")
def history_chat():

    payload = request.get_json(silent=True) or {}
    question = payload.get("question") or payload.get("q") or ""
    is_stream = get_is_stream()

    if not question.strip():
        return jsonify({"ok": False, "error": "Thiếu trường 'question'"}), 400

    system_prompt = (
        "Bạn là một trợ lý AI chuyên về lịch sử Việt Nam và thế giới. "
        "Nhiệm vụ của bạn là trả lời các câu hỏi lịch sử bằng **tiếng Việt**, "
        "dễ hiểu, ngắn gọn, chính xác, có chia tách ý rõ ràng, phù hợp với học sinh và người học. "
        "Tuyệt đối **không xuyên tạc lịch sử, không bình luận chính trị, không xúc phạm cá nhân hay tổ chức**, "
        "và luôn thể hiện thái độ khách quan, tôn trọng, đúng chuẩn mực đạo đức."
    )
    prompt = system_prompt + "\n\nCâu hỏi: " + question
    parts = [prompt]

    if is_stream:
        def generate():
            try:
                for chunk in client.models.generate_content_stream(
                    model=MODEL_NAME,
                    contents=parts,
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
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=parts,
            )
            return jsonify({
                "ok": True,
                "answer": getattr(resp, "text", None),
            })
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500


# =================== 2) vision-detect ===================
@app.post("/vision-detect")
def vision_detect():

    file = request.files.get("image") or request.files.get("file")
    if not file:
        return jsonify({"ok": False, "error": "Thiếu file hình ảnh (image)"}), 400

    history_image_prompt = """
        Hãy quan sát kỹ bức ảnh lịch sử được cung cấp và thực hiện các yêu cầu sau:

        1. Nhận diện hình ảnh:
        - Đồ vật/Nhân vật trong ảnh có thể là ai. (Nếu nghi ngờ không phải nhân vật lịch sử thì bỏ qua)
        - Nếu là tranh bối cảnh, phong cảnh thì đưa ra bức ảnh có thể là phong cảnh ở đâu.

        2. Bối cảnh lịch sử:
        - Nêu thời kỳ hoặc triều đại liên quan.
        - Giải thích sự kiện hoặc truyền thuyết gắn liền với hình ảnh.
        - Nếu có nhân vật, hãy cho biết vai trò của họ trong lịch sử Việt Nam (hoặc lịch sử thế giới nếu không phải Việt Nam).

        3. Ý nghĩa & giá trị:
        - Nêu ý nghĩa lịch sử, văn hóa hoặc quân sự của đối tượng trong ảnh.
        - Giải thích vì sao hình ảnh này thường được nhắc đến hoặc trưng bày trong sách / bảo tàng / tư liệu giảng dạy.

        4. Thông tin mở rộng:
        - So sánh nếu có hình ảnh hoặc hiện vật tương tự trong lịch sử Việt Nam hay thế giới.
        - Cung cấp một số thông tin thú vị hoặc ít người biết về đối tượng này (nếu có).

        5. Tóm tắt ngắn gọn (2–3 câu):
        - Viết lại phần kết luận bằng lời dễ hiểu dành cho học sinh tiểu học hoặc THCS.
        """

    is_stream = get_is_stream()

    img_bytes = file.read()
    mime = file.mimetype or "image/jpeg"
    parts = [
        types.Part.from_bytes(
        data=img_bytes,
        mime_type=mime,
      ),
      history_image_prompt
    ]

    if is_stream:
        def generate():
            try:
                for chunk in client.models.generate_content_stream(
                    model=MODEL_NAME,
                    contents=parts,
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
            resp = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=parts,
                )
            return jsonify({
                "ok": True,
                "answer": getattr(resp, "text", None)
            })
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500


# =================== 3) pdf-qa ===================
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
    system_prompt = (
        "Bạn là một trợ lý đọc hiểu tài liệu lịch sử bằng tiếng Việt. "
        "Hãy đọc kỹ tài liệu PDF kèm theo và trả lời câu hỏi bên dưới bằng tiếng Việt rõ ràng, ngắn gọn, "
        "dễ hiểu và **chính xác về mặt lịch sử**. "
        "Không được xuyên tạc nội dung, không thêm ý kiến cá nhân, không bàn chính trị. "
        "Nếu có thể, hãy trích dẫn (ví dụ: 'theo trang 5 của tài liệu')."
    )
    prompt = system_prompt + "\n\nCâu hỏi: " + question

    parts = [
        types.Part.from_bytes(
            data=pdf_bytes,
            mime_type='application/pdf',
        ),
        prompt
    ]

    if is_stream:
        def generate():
            try:
                for chunk in client.models.generate_content_stream(
                    model=MODEL_NAME,
                    contents=parts,
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
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=parts,
            )
            return jsonify({
                "ok": True,
                "answer": getattr(resp, "text", None)
            })
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500



# =============== (Không cần app.run khi chạy WSGI) ===============
if __name__ == "__main__":
    app.run(debug=True)
