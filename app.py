"""
app.py — FITFORM.AI Backend

Features:
  - User signup / login (SQLite + session-based auth)
  - Multi-exercise video analysis (push-up, squat, plank, lunge)
  - Beginner-friendly feedback
  - Chat with Claude for follow-up coaching
"""

import os
import re
import sqlite3
from datetime import timedelta
from functools import wraps

from flask import (
    Flask, request, jsonify, render_template,
    redirect, url_for, session, g,
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import anthropic

from analyzer import analyze
from model import train_model, predict

# ──────────────────────────────────────────────
# App configuration
# ──────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "fitform-ai-secret-key-change-in-production"

# Keep users logged in for 30 days (helps Chrome offer to save the password)
app.permanent_session_lifetime = timedelta(days=30)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
DATABASE = os.path.join(os.path.dirname(__file__), "users.db")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
SUPPORTED_EXERCISES = {"pushup", "squat", "plank", "lunge"}

# Minimum password length and allowed characters (letters + numbers only)
MIN_PASSWORD_LENGTH = 8

# ──────────────────────────────────────────────
# Localized auth messages
# ──────────────────────────────────────────────
AUTH_MSG = {
    "en": {
        "need_both": "Username and password are required.",
        "pw_short": "Password must be at least 8 characters long.",
        "pw_alnum": "Password can only contain letters and numbers — no spaces or symbols.",
        "user_rules": "Username can only contain letters and numbers.",
        "taken": "That username is already taken.",
        "created": "Account created! You can now log in.",
        "need_login": "Please enter your username and password.",
        "invalid": "Invalid username or password.",
    },
    "es": {
        "need_both": "El usuario y la contrasena son obligatorios.",
        "pw_short": "La contrasena debe tener al menos 8 caracteres.",
        "pw_alnum": "La contrasena solo puede contener letras y numeros, sin espacios ni simbolos.",
        "user_rules": "El usuario solo puede contener letras y numeros.",
        "taken": "Ese nombre de usuario ya esta en uso.",
        "created": "Cuenta creada! Ya puedes iniciar sesion.",
        "need_login": "Por favor introduce tu usuario y contrasena.",
        "invalid": "Usuario o contrasena incorrectos.",
    },
}


def _auth_lang() -> str:
    lang = request.form.get("language", "en")
    return lang if lang in AUTH_MSG else "en"


def validate_password(password: str) -> bool:
    """Password must be >= 8 chars and contain only letters and numbers."""
    return len(password) >= MIN_PASSWORD_LENGTH and password.isalnum()

# ──────────────────────────────────────────────
# Train the optional ML model once at startup
# ──────────────────────────────────────────────
print("Training ML model...")
ml_model, ml_accuracy = train_model()

# ──────────────────────────────────────────────
# Database
# ──────────────────────────────────────────────
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DATABASE)
    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.commit()
    db.close()


init_db()

# ──────────────────────────────────────────────
# Auth decorator
# ──────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return decorated


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ──────────────────────────────────────────────
# Claude feedback
# ──────────────────────────────────────────────
def get_claude_feedback(result: dict, language: str = "en") -> str:
    """Generate beginner-friendly feedback via Claude API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        return generate_fallback_feedback(result, language)

    exercise_names = {
        "pushup": "push-up", "squat": "squat",
        "plank": "plank", "lunge": "lunge",
    }
    exercise_name = exercise_names.get(result["exercise"], result["exercise"])
    reply_language = "Spanish" if language == "es" else "English"

    prompt = (
        f"You are a friendly, encouraging fitness coach.\n\n"
        f"A beginner just recorded themselves doing a {exercise_name}. "
        f"A computer vision system analyzed their form.\n\n"
        f"Overall rating: {result['overall']}\n"
        f"Strengths: {', '.join(result['strengths']) if result['strengths'] else 'None detected'}\n"
        f"Areas to improve: {', '.join(result['tips']) if result['tips'] else 'None — great form!'}\n\n"
        f"Write a short, warm, motivating coaching message (3-5 sentences). "
        f"Be specific about what they did well and what to focus on. "
        f"Use simple language — no jargon. Don't mention angles or numbers. "
        f"Write your entire reply in {reply_language}."
    )

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def generate_fallback_feedback(result: dict, language: str = "en") -> str:
    """Simple feedback when Claude API is not available."""
    intros = {
        "en": {
            "great": "Awesome work! Your form looks really solid.",
            "okay": "Nice effort! You're on the right track.",
            "needs work": "Keep going! Every rep is progress.",
            "unknown": "Let's take another look.",
        },
        "es": {
            "great": "Excelente trabajo! Tu tecnica se ve muy solida.",
            "okay": "Buen esfuerzo! Vas por buen camino.",
            "needs work": "Sigue asi! Cada repeticion es progreso.",
            "unknown": "Vamos a intentarlo de nuevo.",
        },
    }
    lang = language if language in intros else "en"
    parts = [intros[lang].get(result["overall"], intros[lang]["okay"])]

    for s in result.get("strengths", []):
        parts.append(s)
    for tip in result.get("tips", []):
        parts.append(tip)

    return " ".join(parts)


# ──────────────────────────────────────────────
# Routes — Auth
# ──────────────────────────────────────────────
@app.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("analyzer_page"))
    return render_template("home.html")


@app.route("/signup", methods=["POST"])
def signup():
    lang = _auth_lang()
    msg = AUTH_MSG[lang]
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    if not username or not password:
        return render_template("home.html", error=msg["need_both"])
    if not username.isalnum():
        return render_template("home.html", error=msg["user_rules"])
    if len(password) < MIN_PASSWORD_LENGTH:
        return render_template("home.html", error=msg["pw_short"])
    if not password.isalnum():
        return render_template("home.html", error=msg["pw_alnum"])

    db = get_db()
    existing = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if existing:
        return render_template("home.html", error=msg["taken"])

    hashed = generate_password_hash(password)
    db.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
    db.commit()

    return render_template("home.html", success=msg["created"])


@app.route("/login", methods=["POST"])
def login():
    lang = _auth_lang()
    msg = AUTH_MSG[lang]
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    if not username or not password:
        return render_template("home.html", error=msg["need_login"])

    db = get_db()
    user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

    if not user or not check_password_hash(user["password"], password):
        return render_template("home.html", error=msg["invalid"])

    session.permanent = True
    session["user_id"] = user["id"]
    session["username"] = user["username"]
    return redirect(url_for("analyzer_page"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# ──────────────────────────────────────────────
# Routes — Analyzer
# ──────────────────────────────────────────────
@app.route("/analyze")
@login_required
def analyzer_page():
    return render_template("analyzer.html", username=session.get("username"))


@app.route("/upload", methods=["POST"])
@login_required
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided."}), 400

    file = request.files["video"]
    exercise = request.form.get("exercise", "pushup")
    language = request.form.get("language", "en")

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file format."}), 400
    if exercise not in SUPPORTED_EXERCISES:
        return jsonify({"error": "Unsupported exercise."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        result = analyze(filepath, exercise, language)
    except Exception as e:
        try:
            os.remove(filepath)
        except OSError:
            pass
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

    # ML prediction (only for push-ups, since the model is trained on push-up data)
    if exercise == "pushup" and result.get("min_elbow_angle") is not None:
        ml_prediction = predict(ml_model, result["min_elbow_angle"], result["avg_body_alignment"])
        result["ml_prediction"] = ml_prediction

    # Generate coaching feedback
    try:
        feedback = get_claude_feedback(result, language)
    except Exception:
        feedback = generate_fallback_feedback(result, language)

    result["feedback"] = feedback

    try:
        os.remove(filepath)
    except OSError:
        pass

    return jsonify(result)


# ──────────────────────────────────────────────
# Routes — Chat
# ──────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided."}), 400

    user_message = data["message"]
    context = data.get("context", {})
    language = data.get("language", "en")
    reply_language = "Spanish" if language == "es" else "English"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        no_key = {
            "en": "Chat requires a Claude API key. Set the ANTHROPIC_API_KEY environment variable to enable this feature.",
            "es": "El chat requiere una clave de API de Claude. Configura la variable de entorno ANTHROPIC_API_KEY para habilitar esta funcion.",
        }
        return jsonify({"reply": no_key.get(language, no_key["en"])})

    exercise_names = {
        "pushup": "push-up", "squat": "squat",
        "plank": "plank", "lunge": "lunge",
    }

    system_prompt = (
        "You are a friendly, knowledgeable fitness coach on the FITFORM.AI app. "
        "Keep answers short (2-4 sentences), clear, and beginner-friendly. "
        "Never use technical jargon or mention angles/numbers. "
        "Be encouraging and practical. "
        f"Always reply in {reply_language}."
    )

    if context:
        ex_name = exercise_names.get(context.get("exercise", ""), "exercise")
        system_prompt += (
            f"\n\nThe user just had their {ex_name} analyzed. "
            f"Rating: {context.get('overall', 'unknown')}. "
            f"Strengths: {', '.join(context.get('strengths', []))}. "
            f"Tips: {', '.join(context.get('tips', []))}."
        )

    client = anthropic.Anthropic(api_key=api_key)

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        reply = message.content[0].text
    except Exception as e:
        reply = f"Sorry, I couldn't process that. ({str(e)})"

    return jsonify({"reply": reply})


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting FITFORM.AI server...")
    print(f"ML model accuracy: {ml_accuracy:.2%}")
    app.run(debug=True, port=5000)
