import os

# ==============================
# FFmpeg PATH
# ==============================

FFMPEG_PATH = r"C:\Users\khush\Downloads\ffmpeg\ffmpeg-9.0-essentials_build\bin"

os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ.get("PATH", "")


# ==============================
# Whisper
# ==============================

import whisper


# Load Whisper model
print("Loading Whisper model...")
model = whisper.load_model("base")
print("Whisper model loaded successfully!")


# ==============================
# Speech To Text
# ==============================

def transcribe_audio(audio_file):

    try:

        print(f"Transcribing: {audio_file}")

        result = model.transcribe(
            audio_file,
            fp16=False
        )

        text = result["text"].strip()

        print("Transcription:", text)

        return text

    except Exception as e:

        print("Transcription failed:", e)

        return ""