import os
import json
import hashlib
import shutil
import html
import re
import io
import wave
import time
import queue
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
from datetime import datetime, timedelta

import pyttsx3
import streamlit as st

from record import record_audio
from stt import transcribe_audio
from llm import get_ai_response
from tts import speak_text

# Gemini + browser microphone
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except Exception:
    genai = None
    types = None
    GEMINI_AVAILABLE = False

try:
    from streamlit_mic_recorder import mic_recorder
    MIC_RECORDER_AVAILABLE = True
except Exception:
    mic_recorder = None
    MIC_RECORDER_AVAILABLE = False

GEMINI_API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("GOOGLE_GEMINI_API_KEY")
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_AVAILABLE and GEMINI_API_KEY else None
except Exception:
    gemini_client = None

try:
    from pypdf import PdfReader
    PDF_READER_AVAILABLE = True
except ImportError:
    PDF_READER_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "users.json")
CONVERSATION_FILE = os.path.join(BASE_DIR, "conversation.txt")
INTERVIEWS_DIR = os.path.join(BASE_DIR, "interviews")


# ============================================================
# PAGE 
# ============================================================

st.set_page_config(
    page_title="PrepTalk AI",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at top left,
                #172554 0%,
                #0f172a 38%,
                #020617 100%
            );

        color: #f8fafc;
    }

    .block-container {
        max-width: 1000px;
        padding-top: 35px;
        padding-bottom: 60px;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
    }

    p, label {
        color: #cbd5e1 !important;
    }

    div[data-baseweb="input"] {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="input"] input {
        color: #f8fafc !important;
    }

    textarea {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
    }

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #334155;
        background: linear-gradient(
            135deg,
            #4f46e5,
            #2563eb
        );
        color: white;
        font-weight: 600;
        padding: 12px 20px;
    }

    .stButton > button:hover {
        border-color: #60a5fa;
        transform: translateY(-1px);
        box-shadow:
            0 8px 25px
            rgba(37, 99, 235, 0.30);
    }

    .card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid #334155;
        border-radius: 18px;
        padding: 28px;
        margin: 15px 0;
        box-shadow:
            0 15px 40px
            rgba(0,0,0,0.25);
    }

    .logo-circle {
        width: 92px;
        height: 92px;
        margin: auto;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 50%;

        background:
            linear-gradient(
                135deg,
                #6366f1,
                #2563eb
            );

        box-shadow:
            0 15px 35px
            rgba(37,99,235,0.35);

        font-size: 45px;
    }

    .brand-title {
        text-align: center;
        font-size: 38px;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 15px;
    }

    .brand-subtitle {
        text-align: center;
        font-size: 16px;
        color: #94a3b8;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    .question-card {
        background: #172554;
        border-left: 4px solid #6366f1;
        padding: 16px;
        border-radius: 10px;
        margin: 10px 0;
    }

    .answer-card {
        background: #0f172a;
        border-left: 4px solid #22c55e;
        padding: 16px;
        border-radius: 10px;
        margin: 10px 0;
    }

    .feedback-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 18px;
        margin: 10px 0;
    }

    .history-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 18px;
        margin: 12px 0;
    }

    hr {
        border-color: #334155 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# AUTH
# ============================================================

def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4)


def create_user(username, password):
    users = load_users()
    if username in users:
        return False, "Username already exists."
    users[username] = {"password": hash_password(password)}
    save_users(users)
    return True, "Account created successfully."


def authenticate_user(username, password):
    users = load_users()
    return username in users and users[username]["password"] == hash_password(password)


# ============================================================
# SESSION
# ============================================================

DEFAULTS = {
    "page": "login",
    "logged_in": False,
    "username": "",
    "auth_mode": "login",
    "interview_type": None,
    "conversation_started": False,
    "interview_completed": False,
    "transcription": "",
    "ai_response": "",
    "conversation_history": [],
    "current_question": "Tell me about yourself.",
    "feedback": None,
    "feedback_score": None,
    "start_time": None,
    "end_time": None,
    "interview_folder": None,
    "user_audio_folder": None,
    "ai_audio_folder": None,
    "user_audio_count": 0,
    "ai_audio_count": 0,
    "report_path": None,
    "resume_text": "",
    "resume_name": "",
    "company_name": "",
    "company_description": "",
    "company_suggestions": [],
    "awaiting_candidate_answer": False,
    "max_questions": 10,
    "answer_max_seconds": 120,
    "silence_seconds": 2.0,
    "initial_silence_timeout": 10.0,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# HELPERS
# ============================================================

def show_logo():
    st.markdown("""
    <div class="logo-circle">🤖</div>
    <div class="brand-title">PrepTalk AI</div>
    <div class="brand-subtitle">Your Personal AI Interview Coach</div>
    """, unsafe_allow_html=True)


def stars(value):
    try:
        value = max(0, min(5, int(value)))
    except Exception:
        value = 0
    return "⭐" * value + "☆" * (5 - value)


def rating(value, default=3):
    try:
        value = int(value)
    except Exception:
        value = default
    return max(1, min(5, value))


def duration_text():
    start = st.session_state.start_time
    end = st.session_state.end_time
    if not start or not end:
        return "N/A"
    seconds = max(0, int((end - start).total_seconds()))
    return f"{seconds // 60} min {seconds % 60} sec"


def question_count():
    return sum(x.get("role") == "assistant" for x in st.session_state.conversation_history)


def answer_count():
    return sum(x.get("role") == "user" for x in st.session_state.conversation_history)


def normalize_feedback(data):
    if not isinstance(data, dict):
        return None

    out = dict(data)

    try:
        out["score"] = max(0, min(100, int(out.get("score", 0))))
    except Exception:
        out["score"] = 0

    for key in [
        "communication", "confidence", "answer_quality",
        "relevance", "completeness", "specificity",
        "technical_knowledge", "two_person_communication"
    ]:
        out[key] = rating(out.get(key, 3))

    filler = out.get("filler_words", {})
    if not isinstance(filler, dict):
        filler = {}

    try:
        filler_count = max(0, int(filler.get("count", 0)))
    except Exception:
        filler_count = 0

    examples = filler.get("examples", [])
    if not isinstance(examples, list):
        examples = []

    out["filler_words"] = {
        "rating": rating(filler.get("rating", 3)),
        "count": filler_count,
        "examples": [str(x) for x in examples[:10]]
    }

    for key in ["strengths", "areas_for_improvement"]:
        if not isinstance(out.get(key), list):
            out[key] = []

    out["overall_feedback"] = str(out.get("overall_feedback", ""))
    return out


# ============================================================
# RESUME / CONTEXT
# ============================================================

def extract_resume_text(file):
    if not file:
        return ""

    if not PDF_READER_AVAILABLE:
        st.error("Install pypdf: pip install pypdf")
        return ""

    try:
        reader = PdfReader(file)
        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        ).strip()
    except Exception as e:
        st.error(f"Could not read resume: {e}")
        return ""


def interview_context():
    context = ""

    if st.session_state.interview_type == "Resume Based Interview":
        if st.session_state.resume_text:
            context += (
                "\n\nCANDIDATE RESUME:\n"
                + st.session_state.resume_text[:12000]
            )

    if st.session_state.interview_type == "Company Specific Interview":
        if st.session_state.company_name:
            context += (
                "\n\nTARGET COMPANY:\n"
                + st.session_state.company_name
            )
        if st.session_state.company_description:
            context += (
                "\nCOMPANY / JOB INFORMATION:\n"
                + st.session_state.company_description[:6000]
            )

    return context


# ============================================================
# INTERVIEW STORAGE
# ============================================================

def create_interview_folder():
    os.makedirs(INTERVIEWS_DIR, exist_ok=True)

    folder = os.path.join(
        INTERVIEWS_DIR,
        "interview_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    )

    user_audio = os.path.join(folder, "user_audio")
    ai_audio = os.path.join(folder, "ai_audio")

    os.makedirs(user_audio, exist_ok=True)
    os.makedirs(ai_audio, exist_ok=True)

    return folder, user_audio, ai_audio


def save_ai_audio(text, path):
    if not text:
        return False

    engine = None
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.setProperty("volume", 1.0)
        engine.save_to_file(text, path)
        engine.runAndWait()
        engine.stop()
        return os.path.exists(path)
    except Exception as e:
        print("AI audio save failed:", e)
        try:
            if engine:
                engine.stop()
        except Exception:
            pass
        return False


def save_conversation():
    history = st.session_state.conversation_history
    if not history:
        return

    folder = st.session_state.interview_folder
    path = (
        os.path.join(folder, "conversation.txt")
        if folder
        else CONVERSATION_FILE
    )

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("========================================\n")
            f.write("             PREPTALK AI\n")
            f.write("          INTERVIEW RECORD\n")
            f.write("========================================\n\n")
            f.write(f"Candidate: {st.session_state.username}\n")
            f.write(f"Interview Type: {st.session_state.interview_type}\n")
            f.write(f"Duration: {duration_text()}\n\n")

            for msg in history:
                role = "User" if msg.get("role") == "user" else "AI"
                f.write(f"{role}:\n{msg.get('content','')}\n\n")
                f.write("----------------------------------------\n\n")
    except Exception as e:
        print("Conversation save failed:", e)


def save_metadata():
    folder = st.session_state.interview_folder
    if not folder:
        return

    metadata = {
        "candidate": st.session_state.username,
        "interview_type": st.session_state.interview_type,
        "company_name": st.session_state.company_name,
        "company_description": st.session_state.company_description,
        "resume_name": st.session_state.resume_name,
        "start_time": (
            st.session_state.start_time.strftime("%Y-%m-%d %H:%M:%S")
            if st.session_state.start_time else None
        ),
        "end_time": (
            st.session_state.end_time.strftime("%Y-%m-%d %H:%M:%S")
            if st.session_state.end_time else None
        ),
        "duration": duration_text(),
        "questions": question_count(),
        "answers": answer_count(),
        "score": (
            st.session_state.feedback.get("score")
            if st.session_state.feedback else None
        ),
        "feedback": st.session_state.feedback,
    }

    try:
        with open(
            os.path.join(folder, "metadata.json"),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print("Metadata save failed:", e)


def interview_history():
    result = []

    if not os.path.exists(INTERVIEWS_DIR):
        return result

    for folder in os.listdir(INTERVIEWS_DIR):
        path = os.path.join(INTERVIEWS_DIR, folder)
        metadata_path = os.path.join(path, "metadata.json")

        if not os.path.isdir(path) or not os.path.exists(metadata_path):
            continue

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("candidate") == st.session_state.username:
                result.append(data)
        except Exception:
            pass

    result.sort(
        key=lambda x: x.get("start_time", ""),
        reverse=True
    )
    return result


def reset_interview():
    for key, value in {
        "conversation_started": False,
        "interview_completed": False,
        "transcription": "",
        "ai_response": "",
        "conversation_history": [],
        "current_question": "Tell me about yourself.",
        "feedback": None,
        "feedback_score": None,
        "start_time": None,
        "end_time": None,
        "interview_folder": None,
        "user_audio_folder": None,
        "ai_audio_folder": None,
        "user_audio_count": 0,
        "ai_audio_count": 0,
        "report_path": None,
        "resume_text": "",
        "resume_name": "",
        "company_name": "",
        "company_description": "",
        "company_suggestions": [],
        "human_recording_path": None,
        "human_recording_bytes": None,
        "human_transcript": [],
        "human_analysis": None,
        "human_start_time": None,
        "human_end_time": None,
        "human_report_path": None,
    }.items():
        st.session_state[key] = value


# ============================================================
# GEMINI FEEDBACK
# ============================================================

def clean_json_response(text):
    """Parse a Gemini JSON object response safely."""
    text = (text or "").strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.I)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def clean_json_any_response(text):
    """Parse either a JSON object or JSON array returned by Gemini."""
    text = (text or "").strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.I)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass

    object_match = re.search(r"\{.*\}", text, re.DOTALL)
    if object_match:
        try:
            return json.loads(object_match.group(0))
        except Exception:
            pass

    array_match = re.search(r"\[.*\]", text, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except Exception:
            pass

    return None


def analyze_saved_audio_features():
    folder = st.session_state.get("user_audio_folder")
    result = {"audio_files": 0, "audio_seconds": 0.0, "avg_rms": 0.0, "silence_ratio": 0.0}
    if not folder or not os.path.isdir(folder):
        return result
    try:
        import librosa
        import numpy as np
    except Exception:
        return result
    rms_values, silence_values = [], []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".wav"):
            continue
        path = os.path.join(folder, name)
        try:
            y, sr = librosa.load(path, sr=None, mono=True)
            if len(y) == 0:
                continue
            result["audio_files"] += 1
            result["audio_seconds"] += float(len(y) / (sr or 1))
            rms = librosa.feature.rms(y=y)[0]
            rms_values.append(float(np.mean(rms)))
            silence_values.append(float(np.mean(rms < max(0.005, np.mean(rms) * 0.25))))
        except Exception:
            continue
    if rms_values:
        result["avg_rms"] = round(sum(rms_values) / len(rms_values), 4)
    if silence_values:
        result["silence_ratio"] = round(sum(silence_values) / len(silence_values), 3)
    return result


def generate_ai_feedback():
    """
    REAL AI feedback using the same working get_ai_response()
    pipeline used for interview questions.

    No dummy feedback.
    """

    history = st.session_state.get("conversation_history", [])

    if not history:
        st.error("No interview conversation was found.")
        return None

    candidate_answers = [
        str(m.get("content", "")).strip()
        for m in history
        if m.get("role") == "user"
        and str(m.get("content", "")).strip()
    ]

    interviewer_questions = [
        str(m.get("content", "")).strip()
        for m in history
        if m.get("role") == "assistant"
        and str(m.get("content", "")).strip()
    ]

    if not candidate_answers:
        st.error("No candidate answers were found for feedback.")
        return None

    # ---------------------------------------------------------
    # BUILD REAL TRANSCRIPT
    # ---------------------------------------------------------

    transcript_parts = []

    for msg in history:
        content = str(msg.get("content", "")).strip()

        if not content:
            continue

        if msg.get("role") == "assistant":
            transcript_parts.append(
                f"INTERVIEWER: {content}"
            )
        else:
            transcript_parts.append(
                f"CANDIDATE: {content}"
            )

    transcript = "\n\n".join(transcript_parts)

    # ---------------------------------------------------------
    # LOCAL FILLER DETECTION
    # ---------------------------------------------------------

    filler_pattern = re.compile(
        r"\b(?:um+|uh+|er+|hmm+|like|actually|basically|"
        r"literally|you know|so|means|uhh+)\b",
        re.IGNORECASE
    )

    local_fillers = filler_pattern.findall(
        " ".join(candidate_answers)
    )

    local_filler_count = len(local_fillers)

    unique_fillers = list(
        dict.fromkeys(
            [str(x).lower() for x in local_fillers]
        )
    )[:10]

    # ---------------------------------------------------------
    # REAL GEMINI PROMPT
    # ---------------------------------------------------------

    prompt = f"""
You are the REAL AI Interview Evaluation Engine for PrepTalk AI.

Analyze ONLY the actual interview transcript provided below.

DO NOT invent:
- answers
- achievements
- technologies
- examples
- scores
- strengths
- weaknesses
- filler words
- evidence

The candidate actually interacted with an interviewer.
Evaluate the candidate based ONLY on the conversation.

INTERVIEW TYPE:
{st.session_state.get("interview_type", "General Interview")}

COMPANY:
{st.session_state.get("company_name", "N/A")}

NUMBER OF QUESTIONS:
{len(interviewer_questions)}

NUMBER OF CANDIDATE ANSWERS:
{len(candidate_answers)}

LOCALLY DETECTED FILLER COUNT:
{local_filler_count}

ACTUAL INTERVIEW TRANSCRIPT:

{transcript}

Evaluate:

1. Communication
2. Confidence
3. Answer Quality
4. Relevance
5. Completeness
6. Specificity
7. Technical Knowledge
8. Two-Person Communication
9. Filler Words
10. Strengths
11. Areas for Improvement
12. Overall Feedback

Two-Person Communication means:
- Did the candidate directly answer the interviewer?
- Did the candidate understand the question?
- Did the candidate handle follow-up questions?
- Did the candidate maintain conversational flow?
- Did the candidate avoid changing the topic?
- Were answers appropriately sized?
- Did the candidate respond naturally to the interviewer?

For HR interviews, technical knowledge should NOT be penalized
when technical knowledge is not relevant.

Return ONLY valid JSON.

Use EXACTLY this structure:

{{
    "score": 0,

    "communication": 1,
    "confidence": 1,
    "answer_quality": 1,
    "relevance": 1,
    "completeness": 1,
    "specificity": 1,
    "technical_knowledge": 1,
    "two_person_communication": 1,

    "filler_words": {{
        "rating": 1,
        "count": 0,
        "examples": []
    }},

    "strengths": [],

    "areas_for_improvement": [],

    "overall_feedback": ""
}}

Rules:

- score must be between 0 and 100.
- All rating fields must be between 1 and 5.
- filler_words.count must reflect actual candidate speech.
- filler_words.examples must contain only actual fillers found in candidate answers.
- strengths must be based on the transcript.
- areas_for_improvement must be based on the transcript.
- overall_feedback must be specific to this candidate.
- Never create dummy feedback.
- Never return markdown.
- Never return ```json.
"""

    # ---------------------------------------------------------
    # CALL THE SAME WORKING AI PIPELINE
    # ---------------------------------------------------------

    try:

        raw = get_ai_response(
            prompt,
            st.session_state.get(
                "interview_type",
                "General Interview"
            ),
            history
        )

        if not raw:
            st.error(
                "Gemini returned an empty feedback response."
            )
            return None

        raw = str(raw).strip()

        print("\n========== RAW AI FEEDBACK ==========")
        print(raw)
        print("=====================================\n")

        # -----------------------------------------------------
        # REMOVE MARKDOWN CODE FENCES IF GEMINI ADDS THEM
        # -----------------------------------------------------

        raw = re.sub(
            r"^```json\s*",
            "",
            raw,
            flags=re.IGNORECASE
        )

        raw = re.sub(
            r"^```\s*",
            "",
            raw
        )

        raw = re.sub(
            r"\s*```$",
            "",
            raw
        )

        raw = raw.strip()

        # -----------------------------------------------------
        # FIND JSON OBJECT
        # -----------------------------------------------------

        start = raw.find("{")
        end = raw.rfind("}")

        if start == -1 or end == -1 or end <= start:

            st.error(
                "AI feedback response was not valid JSON."
            )

            print(
                "INVALID FEEDBACK RESPONSE:",
                raw
            )

            return None

        json_text = raw[start:end + 1]

        # -----------------------------------------------------
        # PARSE JSON
        # -----------------------------------------------------

        try:

            data = json.loads(json_text)

        except json.JSONDecodeError as e:

            st.error(
                f"Feedback JSON parsing failed: {e}"
            )

            print(
                "JSON PARSE ERROR:",
                e
            )

            print(
                "RAW JSON:",
                json_text
            )

            return None

        if not isinstance(data, dict):

            st.error(
                "AI feedback did not return a JSON object."
            )

            return None

        # -----------------------------------------------------
        # NORMALIZE SCORE
        # -----------------------------------------------------

        def safe_int(value, default):

            try:
                return int(float(value))
            except Exception:
                return default

        data["score"] = max(
            0,
            min(
                100,
                safe_int(
                    data.get("score"),
                    0
                )
            )
        )

        # -----------------------------------------------------
        # NORMALIZE RATINGS
        # -----------------------------------------------------

        rating_fields = [
            "communication",
            "confidence",
            "answer_quality",
            "relevance",
            "completeness",
            "specificity",
            "technical_knowledge",
            "two_person_communication"
        ]

        for field in rating_fields:

            value = safe_int(
                data.get(field),
                1
            )

            data[field] = max(
                1,
                min(
                    5,
                    value
                )
            )

        # -----------------------------------------------------
        # FILLER WORDS
        # -----------------------------------------------------

        filler = data.get(
            "filler_words",
            {}
        )

        if not isinstance(filler, dict):
            filler = {}

        filler_count = safe_int(
            filler.get(
                "count",
                local_filler_count
            ),
            local_filler_count
        )

        # If local transcript detected fillers,
        # don't allow Gemini to incorrectly return zero.

        if local_filler_count > 0:
            filler_count = local_filler_count

        filler_rating = safe_int(
            filler.get(
                "rating",
                5 if filler_count == 0 else 3
            ),
            3
        )

        filler_rating = max(
            1,
            min(
                5,
                filler_rating
            )
        )

        examples = filler.get(
            "examples",
            []
        )

        if not isinstance(examples, list):
            examples = []

        # Prefer actual locally detected fillers.
        if unique_fillers:
            examples = unique_fillers
        else:
            examples = [
                str(x).strip()
                for x in examples
                if str(x).strip()
            ][:10]

        data["filler_words"] = {
            "rating": filler_rating,
            "count": filler_count,
            "examples": examples
        }

        # -----------------------------------------------------
        # STRENGTHS
        # -----------------------------------------------------

        strengths = data.get(
            "strengths",
            []
        )

        if not isinstance(strengths, list):
            strengths = []

        data["strengths"] = [
            str(x).strip()
            for x in strengths
            if str(x).strip()
        ]

        # -----------------------------------------------------
        # IMPROVEMENTS
        # -----------------------------------------------------

        improvements = data.get(
            "areas_for_improvement",
            []
        )

        if not isinstance(improvements, list):
            improvements = []

        data["areas_for_improvement"] = [
            str(x).strip()
            for x in improvements
            if str(x).strip()
        ]

        # -----------------------------------------------------
        # OVERALL FEEDBACK
        # -----------------------------------------------------

        overall = str(
            data.get(
                "overall_feedback",
                ""
            )
        ).strip()

        if not overall:

            st.error(
                "Gemini returned no overall feedback."
            )

            return None

        data["overall_feedback"] = overall

        # -----------------------------------------------------
        # SAVE SCORE IN SESSION
        # -----------------------------------------------------

        st.session_state.feedback_score = data["score"]

        print(
            "REAL AI FEEDBACK GENERATED:",
            data
        )

        return data

    except Exception as e:

        st.error(
            f"Real AI feedback generation failed: {e}"
        )

        print(
            "FEEDBACK ERROR:",
            repr(e)
        )

        return None

# ============================================================
# FEEDBACK UI
# ============================================================

def feedback_box(title, value, icon):
    st.markdown(
        f"""
        <div class="feedback-box">
            <div class="feedback-title">{icon} {title}</div>
            <div class="feedback-stars">{stars(value)}</div>
            <div>{value}/5</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_feedback(feedback):
    feedback = normalize_feedback(feedback)

    if not feedback:
        st.warning("AI feedback could not be generated.")
        return

    st.subheader("📋 AI Interview Feedback")

    st.metric("Overall Score", f"{feedback['score']}/100")

    st.markdown("### 📊 Detailed Evaluation")

    # 3 x 3 grid
    fields = [
        ("Communication", "communication", "🗣️"),
        ("Confidence", "confidence", "💪"),
        ("Answer Quality", "answer_quality", "🎯"),
        ("Relevance", "relevance", "🎯"),
        ("Completeness", "completeness", "📝"),
        ("Specificity", "specificity", "🔎"),
        ("Technical Knowledge", "technical_knowledge", "💻"),
        ("Two-Person Communication", "two_person_communication", "🤝"),
    ]

    for i in range(0, len(fields), 3):
        cols = st.columns(3)
        for col, (title, key, icon) in zip(cols, fields[i:i+3]):
            with col:
                feedback_box(title, feedback[key], icon)

    filler = feedback["filler_words"]

    cols = st.columns(3)
    with cols[0]:
        feedback_box("Filler Words", filler["rating"], "🗣️")
    with cols[1]:
        st.metric("Approx. Filler Count", filler["count"])
    with cols[2]:
        if filler["examples"]:
            st.write("**Examples:**")
            st.write(", ".join(filler["examples"]))
        else:
            st.write("**Examples:** None identified")

    st.markdown("### 🧠 Overall Feedback")
    st.markdown(
        f"""
        <div class="feedback-card">
            {html.escape(feedback["overall_feedback"])}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 💪 Strengths")
    for item in feedback["strengths"]:
        st.success("✓ " + item)

    if not feedback["strengths"]:
        st.info("No specific strengths identified.")

    st.markdown("### 🎯 Areas for Improvement")
    for item in feedback["areas_for_improvement"]:
        st.warning("• " + item)

    if not feedback["areas_for_improvement"]:
        st.info("No specific improvement areas identified.")


# ============================================================
# PDF REPORT
# ============================================================

def generate_report_pdf():
    if not PDF_AVAILABLE or not st.session_state.interview_folder:
        return None

    path = os.path.join(
        st.session_state.interview_folder,
        "report.pdf"
    )

    try:
        doc = SimpleDocTemplate(
            path,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        styles = getSampleStyleSheet()

        title = ParagraphStyle(
            "Title",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=20,
            spaceAfter=15
        )

        heading = ParagraphStyle(
            "Heading",
            parent=styles["Heading2"],
            fontSize=14,
            spaceBefore=12,
            spaceAfter=8
        )

        normal = ParagraphStyle(
            "Normal",
            parent=styles["Normal"],
            fontSize=10,
            leading=14
        )

        feedback = normalize_feedback(
            st.session_state.feedback
        )

        story = [
            Paragraph("PREPTALK AI", title),
            Paragraph("INTERVIEW REPORT", heading),
            Spacer(1, 8)
        ]

        summary = [
            ["Candidate", st.session_state.username],
            ["Interview Type", st.session_state.interview_type or "N/A"],
            ["Company", st.session_state.company_name or "N/A"],
            ["Resume", st.session_state.resume_name or "N/A"],
            ["Duration", duration_text()],
            ["Questions", str(question_count())],
            ["Answers", str(answer_count())],
            [
                "Overall Score",
                f"{feedback['score']}/100" if feedback else "N/A"
            ]
        ]

        table = Table(summary, colWidths=[160, 330])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#e2e8f0")),
            ("GRID", (0,0), (-1,-1), .5, colors.grey),
            ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
            ("VALIGN", (0,0), (-1,-1), "TOP")
        ]))

        story.append(table)
        story.append(Spacer(1, 18))
        story.append(Paragraph("AI Interview Feedback", heading))

        if feedback:
            filler = feedback["filler_words"]

            rows = [
                ["Category", "Rating / Details"],
                ["Communication", f"{feedback['communication']}/5"],
                ["Confidence", f"{feedback['confidence']}/5"],
                ["Answer Quality", f"{feedback['answer_quality']}/5"],
                ["Relevance", f"{feedback['relevance']}/5"],
                ["Completeness", f"{feedback['completeness']}/5"],
                ["Specificity", f"{feedback['specificity']}/5"],
                ["Technical Knowledge", f"{feedback['technical_knowledge']}/5"],
                ["Two-Person Communication", f"{feedback['two_person_communication']}/5"],
                ["Filler Words", f"{filler['rating']}/5 | Approx. {filler['count']}"],
            ]

            t = Table(rows, colWidths=[280, 210])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#172554")),
                ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                ("GRID", (0,0), (-1,-1), .5, colors.grey),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("ALIGN", (1,1), (1,-1), "CENTER")
            ]))
            story.append(t)
            story.append(Spacer(1, 12))

            story.append(
                Paragraph(
                    "<b>Overall Feedback:</b> " +
                    html.escape(feedback["overall_feedback"]),
                    normal
                )
            )

            story.append(Spacer(1, 8))
            story.append(Paragraph("<b>Strengths</b>", normal))

            for item in feedback["strengths"]:
                story.append(
                    Paragraph("• " + html.escape(str(item)), normal)
                )

            story.append(Spacer(1, 8))
            story.append(Paragraph("<b>Areas for Improvement</b>", normal))

            for item in feedback["areas_for_improvement"]:
                story.append(
                    Paragraph("• " + html.escape(str(item)), normal)
                )

        story.append(Spacer(1, 15))
        story.append(Paragraph("Conversation", heading))

        for msg in st.session_state.conversation_history:
            role = "AI Interviewer" if msg.get("role") == "assistant" else "Candidate"
            text = html.escape(str(msg.get("content", ""))).replace("\n", "<br/>")
            story.append(Paragraph(f"<b>{role}:</b>", normal))
            story.append(Paragraph(text, normal))
            story.append(Spacer(1, 7))

        doc.build(story)
        return path if os.path.exists(path) else None

    except Exception as e:
        print("PDF generation failed:", e)
        return None


# ============================================================
# INTERVIEW HISTORY
# ============================================================

def get_interview_history():

    history = []

    if not os.path.exists(
        INTERVIEWS_DIR
    ):
        return history

    try:

        folders = os.listdir(
            INTERVIEWS_DIR
        )

    except Exception:

        return history

    for folder in folders:

        folder_path = os.path.join(
            INTERVIEWS_DIR,
            folder
        )

        if not os.path.isdir(
            folder_path
        ):
            continue

        metadata_path = os.path.join(
            folder_path,
            "metadata.json"
        )

        if not os.path.exists(
            metadata_path
        ):
            continue

        try:

            with open(
                metadata_path,
                "r",
                encoding="utf-8"
            ) as f:

                metadata = json.load(f)

            if metadata.get(
                "candidate"
            ) != st.session_state.username:

                continue

            metadata["_folder"] = folder_path

            history.append(
                metadata
            )

        except Exception:

            continue

    history.sort(
        key=lambda x: x.get(
            "start_time",
            ""
        ),
        reverse=True
    )

    return history

# ============================================================
# DASHBOARD STATISTICS
# ============================================================

def get_dashboard_statistics():

    history = get_interview_history()

    sessions = len(history)

    scores = []

    for item in history:

        score = item.get(
            "score"
        )

        if score is not None:

            try:

                scores.append(
                    float(score)
                )

            except Exception:
                pass

    if scores:

        average_score = round(
            sum(scores) / len(scores)
        )

    else:

        average_score = None

    return (
        average_score,
        sessions
    )



# ============================================================
# LOGIN
# ============================================================

def login_page():

    show_logo()

    st.markdown(
        """
        <div class="card">

        <h2 style="text-align:center;">
            🔐 Welcome Back
        </h2>

        <p style="text-align:center;">
            Login to continue your interview practice.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    username = st.text_input(
        "Username",
        placeholder="Enter your username",
        key="login_username"
    )

    password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter your password",
        key="login_password"
    )

    if st.button(
        "🔐 Login",
        use_container_width=True
    ):

        if not username or not password:

            st.error(
                "Please enter username and password."
            )

        elif authenticate_user(
            username,
            password
        ):

            st.session_state.logged_in = True

            st.session_state.username = username

            st.session_state.page = "dashboard"

            st.rerun()

        else:

            st.error(
                "Invalid username or password."
            )

    st.write("")

    if st.button(
        "Create New Account",
        use_container_width=True
    ):

        st.session_state.auth_mode = "signup"

        st.rerun()


# ============================================================
# SIGNUP
# ============================================================

def signup_page():

    show_logo()

    st.markdown(
        """
        <div class="card">

        <h2 style="text-align:center;">
            ✨ Create Account
        </h2>

        </div>
        """,
        unsafe_allow_html=True
    )

    username = st.text_input(
        "Username",
        key="signup_username"
    )

    password = st.text_input(
        "Password",
        type="password",
        key="signup_password"
    )

    confirm_password = st.text_input(
        "Confirm Password",
        type="password",
        key="signup_confirm"
    )

    if st.button(
        "🚀 Sign Up",
        use_container_width=True
    ):

        if not username or not password:

            st.error(
                "Please fill all fields."
            )

        elif password != confirm_password:

            st.error(
                "Passwords do not match."
            )

        elif len(password) < 4:

            st.error(
                "Password should contain at least 4 characters."
            )

        else:

            success, message = create_user(
                username,
                password
            )

            if success:

                st.success(
                    message
                )

                st.session_state.auth_mode = "login"

                st.rerun()

            else:

                st.error(
                    message
                )

    if st.button(
        "⬅️ Back to Login",
        use_container_width=True
    ):

        st.session_state.auth_mode = "login"

        st.rerun()
# ============================================================
# DASHBOARD
# ============================================================

def dashboard_page():

    average_score, sessions = (
        get_dashboard_statistics()
    )

    st.markdown(
        f"""
        <div class="card">

        <h1>
            👋 Welcome,
            {html.escape(
                st.session_state.username
            )}
        </h1>

        <p>
            Ready to improve your interview skills?
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader(
        "📊 Your Progress"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Average Score",
            (
                f"{average_score}/100"
                if average_score is not None
                else "N/A"
            )
        )

    with col2:

        st.metric(
            "Interview Types",
            "4"
        )

    with col3:

        st.metric(
            "Practice Sessions",
            sessions
        )

    st.write("")

    if st.button(
        "📚 View Interview History",
        use_container_width=True
    ):

        st.session_state.page = "history"

        st.rerun()

    st.divider()

    st.subheader(
        "🎯 Practice Areas"
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "👤 HR Interview",
            use_container_width=True
        ):

            reset_interview()

            st.session_state.interview_type = (
                "HR Interview"
            )

            st.session_state.page = "interview"

            st.rerun()

        if st.button(
            "💻 Technical Interview",
            use_container_width=True
        ):

            reset_interview()

            st.session_state.interview_type = (
                "Technical Interview"
            )

            st.session_state.page = "interview"

            st.rerun()

    with col2:

        if st.button(
            "📄 Resume Based Interview",
            use_container_width=True
        ):

            reset_interview()

            st.session_state.interview_type = (
                "Resume Based Interview"
            )

            st.session_state.page = "interview"

            st.rerun()

        if st.button(
            "🏢 Company Specific Interview",
            use_container_width=True
        ):

            reset_interview()

            st.session_state.interview_type = (
                "Company Specific Interview"
            )

            st.session_state.page = "interview"

            st.rerun()

    st.write("")

    if st.button(
        "🚪 Logout",
        use_container_width=True
    ):

        reset_interview()

        st.session_state.logged_in = False

        st.session_state.username = ""

        st.session_state.page = "login"

        st.rerun()



# ============================================================
# HISTORY
# ============================================================

def history_page():
    st.title("📚 Interview History")

    data = interview_history()

    if not data:
        st.info("No interview history available yet.")
    else:
        for item in data:
            feedback = normalize_feedback(item.get("feedback"))

            st.markdown('<div class="history-card">', unsafe_allow_html=True)
            st.subheader("🎤 " + item.get("interview_type", "Interview"))

            st.write("**Score:**", f"{item.get('score', 'N/A')}/100")
            st.write("**Date:**", item.get("start_time", "N/A"))
            st.write("**Duration:**", item.get("duration", "N/A"))

            raw_feedback = item.get("feedback") or {}
            if item.get("interview_type") == "Two-Person Human Interview" and isinstance(raw_feedback, dict) and "two_person_communication" in raw_feedback:
                tpc = raw_feedback.get("two_person_communication", {})
                st.write("**Two-Person Communication:**", f"{tpc.get('score', 'N/A')}/5")
                st.write("**Feedback:**", tpc.get("feedback", "N/A"))

            if feedback:
                st.write("Communication:", stars(feedback["communication"]))
                st.write("Confidence:", stars(feedback["confidence"]))
                st.write("Answer Quality:", stars(feedback["answer_quality"]))
                st.write("Relevance:", stars(feedback["relevance"]))
                st.write("Completeness:", stars(feedback["completeness"]))
                st.write("Specificity:", stars(feedback["specificity"]))
                st.write("Technical Knowledge:", stars(feedback["technical_knowledge"]))
                st.write("Two-Person Communication:", stars(feedback["two_person_communication"]))
                st.write(
                    "Filler Words:",
                    stars(feedback["filler_words"]["rating"]),
                    f"(approx. {feedback['filler_words']['count']})"
                )

            st.markdown("</div>", unsafe_allow_html=True)

    if st.button("⬅️ Back to Dashboard"):
        st.session_state.page = "dashboard"
        st.rerun()


# ============================================================
# INTERVIEW
# ============================================================


def record_candidate_until_silence(
    output_path,
    max_seconds=120,
    silence_seconds=2.0,
    initial_silence_timeout=10.0,
    sample_rate=16000,
    block_seconds=0.1,
    silence_threshold=0.015,
):
    """
    Automatically record one candidate answer.

    Recording starts without a separate button after the AI asks a question.
    It ends when either:
      1) the candidate has been silent continuously for 2 seconds, or
      2) the hard 2-minute answer limit is reached.

    No fixed recording-duration slider is used.
    """
    audio_queue = queue.Queue()
    chunks = []
    speech_started = False
    silent_for = 0.0
    started_at = time.monotonic()

    def callback(indata, frames, time_info, status):
        if status:
            print("Microphone status:", status)
        audio_queue.put(indata.copy())

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=max(1, int(sample_rate * block_seconds)),
            callback=callback,
        ):
            st.info(
                "🎙️ Listening... Speak naturally. "
                "Your answer will stop after 2 seconds of silence "
                "or at the 2-minute limit."
            )
            progress = st.progress(0.0)
            timer_box = st.empty()

            while True:
                elapsed = time.monotonic() - started_at

                if elapsed >= max_seconds:
                    break

                try:
                    chunk = audio_queue.get(timeout=block_seconds + 0.05)
                except queue.Empty:
                    continue

                chunks.append(chunk)

                rms = (
                    float(np.sqrt(np.mean(np.square(chunk))))
                    if chunk.size
                    else 0.0
                )

                if rms >= silence_threshold:
                    speech_started = True
                    silent_for = 0.0
                elif speech_started:
                    silent_for += block_seconds

                progress.progress(min(1.0, elapsed / max_seconds))
                timer_box.caption(
                    f"🔴 Recording {int(elapsed)//60:02d}:"
                    f"{int(elapsed)%60:02d} / 02:00"
                )

                if speech_started and silent_for >= silence_seconds:
                    break

                if (
                    not speech_started
                    and elapsed >= initial_silence_timeout
                ):
                    break

        if not chunks or not speech_started:
            return False, (
                "No speech was detected. Please wait for the microphone "
                "to start listening and then speak."
            )

        audio = np.concatenate(chunks, axis=0)
        audio_int16 = np.clip(
            audio * 32767, -32768, 32767
        ).astype(np.int16)
        write_wav(
            output_path,
            sample_rate,
            audio_int16
        )

        return True, None

    except Exception as e:
        return False, (
            "Automatic microphone recording failed: "
            f"{e}"
        )


def interview_page():

    st.markdown(
        '<div class="card"><h1 style="text-align:center;">🤖 PrepTalk AI</h1></div>',
        unsafe_allow_html=True
    )

    st.caption("Interview Type: " + str(st.session_state.interview_type))

    # Resume
    if (
        st.session_state.interview_type == "Resume Based Interview"
        and not st.session_state.conversation_started
    ):
        st.subheader("📄 Upload Your Resume")

        file = st.file_uploader(
            "Upload PDF Resume",
            type=["pdf"],
            key="resume_uploader"
        )

        if file:
            st.session_state.resume_name = file.name

            if st.button("📖 Read Resume"):
                text = extract_resume_text(file)

                if text:
                    st.session_state.resume_text = text
                    st.success("✅ Resume uploaded and read successfully.")

                    with st.expander("Preview Resume Text"):
                        st.write(text[:5000])

    # Company
    if (
        st.session_state.interview_type == "Company Specific Interview"
        and not st.session_state.conversation_started
    ):
        st.subheader("🏢 Company Information")

        st.session_state.company_name = st.text_input(
            "Company Name",
            value=st.session_state.company_name,
            key="company_name_input"
        )

        st.session_state.company_description = st.text_area(
            "Company Information / Job Role",
            value=st.session_state.company_description,
            height=150,
            key="company_description_input"
        )

        st.caption("💡 You can also ask Gemini to suggest companies based on your target role/field.")
        if st.button("✨ Suggest Company Names", key="suggest_company_names"):
            if gemini_client is None:
                st.error("Gemini is not configured. Add GEMINI_API_KEY to .env.")
            else:
                suggestion_prompt = f"Suggest 8 realistic company names for a student preparing for a {st.session_state.company_description or 'software / AI / data'} role in India. Return ONLY a JSON array of company names. Do not invent obscure fake companies; prefer well-known or plausible employers."
                try:
                    response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=suggestion_prompt)
                    suggestions = clean_json_any_response(getattr(response, "text", "") or "")
                    if isinstance(suggestions, list):
                        st.session_state.company_suggestions = [
                            str(x).strip() for x in suggestions[:8] if str(x).strip()
                        ]
                        if not st.session_state.company_suggestions:
                            st.warning("Gemini returned no company suggestions.")
                    else:
                        st.warning("Could not parse company suggestions.")
                except Exception as e:
                    st.error(f"Company suggestion failed: {e}")
        if st.session_state.get("company_suggestions"):
            st.write("**Suggested companies:**")
            for company in st.session_state.company_suggestions:
                st.write(f"• {company}")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        disabled = st.session_state.conversation_started

        if (
            st.session_state.interview_type == "Resume Based Interview"
            and not st.session_state.resume_text
        ):
            disabled = True

        if (
            st.session_state.interview_type == "Company Specific Interview"
            and not st.session_state.company_name.strip()
        ):
            disabled = True

        if st.button("▶️ Start Conversation", disabled=disabled):

            folder, user_audio, ai_audio = create_interview_folder()

            st.session_state.interview_folder = folder
            st.session_state.user_audio_folder = user_audio
            st.session_state.ai_audio_folder = ai_audio
            st.session_state.user_audio_count = 0
            st.session_state.ai_audio_count = 0
            st.session_state.conversation_history = []
            st.session_state.feedback = None
            st.session_state.report_path = None
            st.session_state.start_time = datetime.now()
            st.session_state.end_time = None
            st.session_state.interview_completed = False
            st.session_state.conversation_started = True
            st.session_state.awaiting_candidate_answer = True

            if st.session_state.interview_type == "Resume Based Interview":
                q = (
                    "Hello! I have reviewed your resume. "
                    "Let's begin the interview. Could you briefly "
                    "introduce yourself and walk me through your background?"
                )
            elif st.session_state.interview_type == "Company Specific Interview":
                q = (
                    "Hello! Welcome to your company-specific interview. "
                    f"We will be focusing on {st.session_state.company_name}. "
                    "Please introduce yourself and tell me why you are interested "
                    "in this company and role."
                )
            elif st.session_state.interview_type == "Technical Interview":
                q = (
                    "Hello! Let's begin your technical interview. "
                    "Tell me about yourself and your technical background."
                )
            else:
                q = "Hello! Let's begin the interview. Tell me about yourself."

            st.session_state.current_question = q
            st.session_state.conversation_history.append(
                {"role": "assistant", "content": q}
            )

            st.session_state.ai_audio_count += 1
            audio_path = os.path.join(
                ai_audio,
                f"question_{st.session_state.ai_audio_count:03d}.wav"
            )

            save_ai_audio(q, audio_path)

            try:
                speak_text(q)
            except Exception as e:
                print("TTS error:", e)

            save_conversation()
            save_metadata()

            st.rerun()

    with c2:
        if st.button(
            "⏹️ Stop Conversation",
            disabled=not st.session_state.conversation_started
        ):

            st.session_state.conversation_started = False
            st.session_state.awaiting_candidate_answer = False
            st.session_state.end_time = datetime.now()
            st.session_state.interview_completed = True

            save_conversation()

            with st.spinner("🧠 AI is analyzing your interview..."):
                st.session_state.feedback = generate_ai_feedback()

            if st.session_state.feedback:
                st.session_state.feedback_score = (
                    st.session_state.feedback["score"]
                )

            save_metadata()

            if PDF_AVAILABLE:
                with st.spinner("📄 Generating interview report..."):
                    st.session_state.report_path = generate_report_pdf()

            st.rerun()

    if st.session_state.conversation_started and st.session_state.awaiting_candidate_answer:

        st.divider()
        st.subheader("🎙️ Candidate Answer")
        st.caption("Voice input starts automatically after each AI question. No Tap-to-Speak button is required.")

        number = st.session_state.user_audio_count + 1
        temp_path = os.path.join(BASE_DIR, "recording.wav")
        answer_path = os.path.join(
            st.session_state.user_audio_folder,
            f"answer_{number:03d}.wav"
        )

        st.session_state.awaiting_candidate_answer = False
        st.session_state.user_audio_count = number

        with st.spinner("🎙️ Listening for your answer..."):
            ok, record_error = record_candidate_until_silence(
                temp_path,
                max_seconds=st.session_state.answer_max_seconds,
                silence_seconds=st.session_state.silence_seconds,
                initial_silence_timeout=st.session_state.initial_silence_timeout,
            )

        if not ok:
            st.error(f"❌ {record_error}")
            st.session_state.user_audio_count -= 1
            st.session_state.awaiting_candidate_answer = True
        elif not os.path.exists(temp_path):
            st.error("❌ Recording file was not created.")
            st.session_state.user_audio_count -= 1
            st.session_state.awaiting_candidate_answer = True
        else:
            try:
                shutil.copy2(temp_path, answer_path)
            except Exception as e:
                st.error(f"Could not save answer audio: {e}")
                st.session_state.user_audio_count -= 1
                st.session_state.awaiting_candidate_answer = True
            else:
                with st.spinner("📝 Converting your answer to text..."):
                    text = transcribe_audio(answer_path)

                if not text or not str(text).strip():
                    st.error("❌ I could not understand the answer. Please try again.")
                    st.session_state.user_audio_count -= 1
                    st.session_state.awaiting_candidate_answer = True
                else:
                    text = str(text).strip()
                    st.session_state.transcription = text
                    st.session_state.conversation_history.append(
                        {"role": "user", "content": text}
                    )

                    if st.session_state.user_audio_count >= st.session_state.max_questions:
                        st.session_state.conversation_started = False
                        st.session_state.interview_completed = True
                        st.session_state.end_time = datetime.now()
                        st.session_state.awaiting_candidate_answer = False

                        save_conversation()
                        save_metadata()
                        with st.spinner("🧠 Analyzing your 10-question interview..."):
                            st.session_state.feedback = generate_ai_feedback()

                        if st.session_state.feedback:
                            st.session_state.feedback_score = st.session_state.feedback.get("score")
                        save_metadata()

                        if PDF_AVAILABLE:
                            with st.spinner("📄 Generating interview report..."):
                                st.session_state.report_path = generate_report_pdf()

                        st.success("🎉 10 questions completed. Your interview is finished.")
                        st.rerun()
                    else:
                        # Hard guard: never allow question 11.
                        if question_count() >= st.session_state.max_questions:
                            st.session_state.conversation_started = False
                            st.session_state.interview_completed = True
                            st.session_state.end_time = datetime.now()
                            st.session_state.awaiting_candidate_answer = False
                            save_conversation()
                            with st.spinner("🧠 Analyzing your 10-question interview..."):
                                st.session_state.feedback = generate_ai_feedback()
                            if st.session_state.feedback:
                                st.session_state.feedback_score = st.session_state.feedback.get("score")
                            save_metadata()
                            if PDF_AVAILABLE:
                                with st.spinner("📄 Generating interview report..."):
                                    st.session_state.report_path = generate_report_pdf()
                            st.rerun()

                        with st.spinner("🤖 Generating the next interview question..."):
                            response = get_ai_response(
                                text + interview_context() +
                                f"\n\nIMPORTANT: This interview has a hard maximum of {st.session_state.max_questions} questions. "
                                f"The candidate has answered {st.session_state.user_audio_count}. Ask exactly one next question and never exceed 10 total questions.",
                                st.session_state.interview_type,
                                st.session_state.conversation_history
                            )

                        if response and question_count() < st.session_state.max_questions:
                            response = str(response).strip()
                            st.session_state.ai_response = response
                            st.session_state.conversation_history.append(
                                {"role": "assistant", "content": response}
                            )

                            st.session_state.ai_audio_count += 1
                            qno = st.session_state.ai_audio_count
                            ai_path = os.path.join(
                                st.session_state.ai_audio_folder,
                                f"question_{qno:03d}.wav"
                            )
                            save_ai_audio(response, ai_path)
                            try:
                                speak_text(response)
                            except Exception as e:
                                print("TTS error:", e)

                            save_conversation()
                            save_metadata()
                            st.session_state.awaiting_candidate_answer = True
                            st.rerun()
                        else:
                            st.error("❌ AI could not generate the next question.")
                            st.session_state.awaiting_candidate_answer = True

    elif st.session_state.conversation_started:
        st.info("🤖 AI interviewer is preparing the next question...")
    elif not st.session_state.interview_completed:
        st.info("Click 'Start Conversation' to begin your interview.")

    # Conversation
    st.divider()
    st.subheader("💬 Conversation")

    if not st.session_state.conversation_history:
        st.markdown(
            '<div class="question-card"><b>🤖 AI Interviewer</b><br><br>'
            'Tell me about yourself.</div>',
            unsafe_allow_html=True
        )
    else:
        for msg in st.session_state.conversation_history:
            content = html.escape(str(msg.get("content", "")))

            if msg.get("role") == "user":
                st.markdown(
                    f'<div class="answer-card"><b>👤 You</b><br><br>{content}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="question-card"><b>🤖 AI Interviewer</b><br><br>{content}</div>',
                    unsafe_allow_html=True
                )

    # Feedback
    if st.session_state.interview_completed:

        st.divider()
        show_feedback(st.session_state.feedback)

        st.divider()
        st.subheader("📝 Interview Summary")

        st.write("**Interview Type:**", st.session_state.interview_type)
        st.write("**Duration:**", duration_text())
        st.write("**Questions:**", question_count())
        st.write("**Answers:**", answer_count())

        if st.session_state.feedback:
            st.write(
                "**Overall Score:**",
                f"{st.session_state.feedback['score']}/100"
            )

        st.divider()
        st.subheader("📄 Interview Report")

        if st.session_state.report_path and os.path.exists(
            st.session_state.report_path
        ):
            with open(st.session_state.report_path, "rb") as f:
                st.download_button(
                    "📥 Download Interview Report",
                    f.read(),
                    "PrepTalk_Interview_Report.pdf",
                    "application/pdf"
                )
        else:
            st.warning("Report PDF could not be generated.")

    # Downloads
    if st.session_state.interview_folder:

        st.divider()

        conversation_path = os.path.join(
            st.session_state.interview_folder,
            "conversation.txt"
        )

        metadata_path = os.path.join(
            st.session_state.interview_folder,
            "metadata.json"
        )

        if os.path.exists(conversation_path):
            with open(conversation_path, "r", encoding="utf-8") as f:
                st.download_button(
                    "📥 Download Conversation",
                    f.read(),
                    "conversation.txt",
                    "text/plain"
                )

        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                st.download_button(
                    "📋 Download Metadata",
                    f.read(),
                    "metadata.json",
                    "application/json"
                )

    if st.button("⬅️ Back to Dashboard"):

        if st.session_state.conversation_started:
            st.warning("⏹️ Please stop the interview first.")
        else:
            save_conversation()
            save_metadata()
            st.session_state.page = "dashboard"
            st.rerun()



# ============================================================
# ROUTING
# ============================================================

if not st.session_state.logged_in:
    if st.session_state.auth_mode == "login":
        login_page()
    else:
        signup_page()
else:
    if st.session_state.page == "dashboard":
        dashboard_page()
    elif st.session_state.page == "interview":
        interview_page()
    elif st.session_state.page == "history":
        history_page()
    else:
        dashboard_page()
