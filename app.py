import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai


try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    else:
        gemini_model = None
except Exception:
    gemini_model = None

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
        "desc": "API Hỏi đáp lịch sử / Vision / PDF QA",
        "version": "1.0.0",
        "ok": True
    })