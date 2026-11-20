import os
import json
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. Please add it to your .env file."
    )

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-3.7-flash"


def clean_json_response(text):
    """
    Removes markdown code fences if Gemini returns JSON inside ```json ... ```
    """

    if not text:
        return "{}"

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "")

    return text.strip()


def analyze_interview(transcript, interview_type="HR Interview"):
    """
    Analyze complete interview transcript using Gemini.
    """

    if not transcript or not transcript.strip():
        return {
            "overall_score": 0,
            "summary": "No transcript was available for analysis.",
            "communication": {
                "score": 0,
                "clarity": 0,
                "sentence_structure": 0,
                "relevance": 0,
                "speaking_pattern": 0,
                "feedback": "No answer was available."
            },
            "confidence": {
                "score": 0,
                "pace": 0,
                "pauses": 0,
                "hesitation": 0,
                "volume_variation": 0,
                "feedback": "Voice analysis could not be performed."
            },
            "filler_words": {
                "total": 0,
                "words": {}
            },
            "answer_quality": {
                "score": 0,
                "relevance": 0,
                "clarity": 0,
                "completeness": 0,
                "specificity": 0,
                "feedback": "No answer was available."
            },
            "strengths": [],
            "improvements": [],
            "recommendation": ""
        }

    prompt = f"""
You are an expert AI interview evaluator for PrepTalk AI.

Analyze the following interview transcript.

Interview Type:
{interview_type}

Transcript:
-------------------------
{transcript}
-------------------------

Evaluate the candidate professionally.

IMPORTANT:
- Do NOT give a fixed or dummy score.
- Scores must be based only on the transcript.
- Do not invent information.
- Give realistic scores from 0 to 100.
- Communication should consider clarity, sentence structure,
  relevance and speaking pattern.
- Confidence can be estimated from textual indicators such as
  hesitation, uncertainty and long pauses if pause information
  exists in the transcript.
- Filler words must be counted from the transcript.
- Answer quality should consider relevance, clarity,
  completeness and specificity.
- Provide actionable feedback.

Return ONLY valid JSON in this exact structure:

{{
    "overall_score": 0,

    "summary": "",

    "communication": {{
        "score": 0,
        "clarity": 0,
        "sentence_structure": 0,
        "relevance": 0,
        "speaking_pattern": 0,
        "feedback": ""
    }},

    "confidence": {{
        "score": 0,
        "pace": 0,
        "pauses": 0,
        "hesitation": 0,
        "volume_variation": 0,
        "feedback": ""
    }},

    "filler_words": {{
        "total": 0,
        "words": {{
            "um": 0,
            "uh": 0,
            "like": 0,
            "actually": 0,
            "basically": 0,
            "you know": 0
        }}
    }},

    "answer_quality": {{
        "score": 0,
        "relevance": 0,
        "clarity": 0,
        "completeness": 0,
        "specificity": 0,
        "feedback": ""
    }},

    "strengths": [],

    "improvements": [],

    "recommendation": ""
}}
"""

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2
            )
        )

        raw_result = response.text

        cleaned_result = clean_json_response(raw_result)

        result = json.loads(cleaned_result)

        return result

    except Exception as e:

        print("Gemini analysis error:", e)

        return {
            "overall_score": 0,
            "summary": "AI analysis failed.",

            "communication": {
                "score": 0,
                "clarity": 0,
                "sentence_structure": 0,
                "relevance": 0,
                "speaking_pattern": 0,
                "feedback": str(e)
            },

            "confidence": {
                "score": 0,
                "pace": 0,
                "pauses": 0,
                "hesitation": 0,
                "volume_variation": 0,
                "feedback": "Analysis unavailable."
            },

            "filler_words": {
                "total": 0,
                "words": {}
            },

            "answer_quality": {
                "score": 0,
                "relevance": 0,
                "clarity": 0,
                "completeness": 0,
                "specificity": 0,
                "feedback": "Analysis unavailable."
            },

            "strengths": [],
            "improvements": [],
            "recommendation": ""
        }


def calculate_overall_score(result):
    """
    Calculates final score from:
    Communication
    Confidence
    Answer Quality
    Filler Words
    """

    communication = result.get("communication", {}).get("score", 0)

    confidence = result.get("confidence", {}).get("score", 0)

    answer_quality = result.get("answer_quality", {}).get("score", 0)

    filler_total = result.get("filler_words", {}).get("total", 0)

    # Filler penalty
    if filler_total <= 2:
        filler_score = 100

    elif filler_total <= 5:
        filler_score = 90

    elif filler_total <= 10:
        filler_score = 75

    elif filler_total <= 15:
        filler_score = 60

    else:
        filler_score = 45

    overall = (
        communication * 0.30
        + confidence * 0.20
        + answer_quality * 0.40
        + filler_score * 0.10
    )

    return round(overall)


def final_evaluation(transcript, interview_type="HR Interview"):

    result = analyze_interview(
        transcript,
        interview_type
    )

    calculated_score = calculate_overall_score(result)

    result["overall_score"] = calculated_score

    return result