import os
from dotenv import load_dotenv
from google import genai


# ============================================================
# LOAD .ENV
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        f"GEMINI_API_KEY not found in .env\n"
        f"Expected .env at: {ENV_PATH}"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "gemini-3.1-flash-lite"


# ============================================================
# AI RESPONSE FUNCTION
# ============================================================

def get_ai_response(
    user_text,
    interview_type=None,
    conversation_history=None
):

    if conversation_history is None:
        conversation_history = []

    try:

        # ----------------------------------------------------
        # BUILD CONVERSATION HISTORY
        # ----------------------------------------------------

        history_text = ""

        for message in conversation_history:

            role = message.get("role", "")
            content = message.get("content", "")

            if role == "user":

                history_text += (
                    f"Candidate: {content}\n"
                )

            elif role == "assistant":

                history_text += (
                    f"Interviewer: {content}\n"
                )


        # ----------------------------------------------------
        # PROMPT
        # ----------------------------------------------------

        prompt = f"""
You are an AI interviewer for an interview practice
application called PrepTalk AI.

Interview Type:
{interview_type}

Previous conversation:
{history_text}

Candidate's latest answer:
{user_text}

Your job:

1. Act like a real interviewer.
2. Consider the previous conversation.
3. Respond naturally to the candidate's answer.
4. Ask the next relevant interview question.
5. Keep the response concise.
6. Do not give unnecessary explanations.
7. Do not mention that you are an AI.
8. Continue the interview naturally.
9. Ask only one question at a time.

Give ONLY the interviewer's response.
"""


        # ----------------------------------------------------
        # GEMINI REQUEST
        # ----------------------------------------------------

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        if response.text:

            return response.text.strip()

        return "Could you tell me more about that?"


    except Exception as e:

        print(f"AI response failed: {e}")

        return f"AI response failed: {e}"


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("Testing Gemini...")

    answer = get_ai_response(
        "I am a Computer Science student interested in AI and Machine Learning.",
        "HR Interview",
        []
    )

    print("\nAI RESPONSE:")
    print(answer)