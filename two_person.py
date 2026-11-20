import os
import json
import re
from datetime import datetime

from dotenv import load_dotenv

import whisper

load_dotenv()


# ============================================================
# WHISPER MODEL
# ============================================================

print("Loading Whisper model for Two-Person Interview...")

whisper_model = whisper.load_model("base")

print("Two-Person Whisper model loaded successfully!")


# ============================================================
# SAVE AUDIO
# ============================================================

def save_uploaded_audio(audio_bytes, output_path):

    try:

        with open(
            output_path,
            "wb"
        ) as f:

            f.write(audio_bytes)

        return os.path.exists(
            output_path
        )

    except Exception as e:

        print(
            "Audio save error:",
            e
        )

        return False


# ============================================================
# WHISPER TRANSCRIPTION WITH TIMESTAMPS
# ============================================================

def transcribe_with_timestamps(audio_path):

    try:

        print(
            f"Transcribing two-person interview: {audio_path}"
        )

        result = whisper_model.transcribe(
            audio_path,
            fp16=False
        )

        segments = []

        for segment in result.get(
            "segments",
            []
        ):

            segments.append(
                {
                    "start": float(
                        segment.get(
                            "start",
                            0
                        )
                    ),

                    "end": float(
                        segment.get(
                            "end",
                            0
                        )
                    ),

                    "text": str(
                        segment.get(
                            "text",
                            ""
                        )
                    ).strip()
                }
            )

        return segments

    except Exception as e:

        print(
            "Two-person transcription error:",
            e
        )

        return []


# ============================================================
# SPEAKER DIARIZATION
# ============================================================

def diarize_audio(audio_path):

    try:

        from pyannote.audio import Pipeline

    except ImportError:

        print(
            "pyannote.audio is not installed."
        )

        return []

    token = os.getenv(
        "HUGGINGFACE_TOKEN"
    )

    if not token:

        print(
            "HUGGINGFACE_TOKEN is missing."
        )

        return []

    try:

        print(
            "Loading speaker diarization model..."
        )

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token
        )

        print(
            "Running speaker diarization..."
        )

        diarization = pipeline(
            audio_path
        )

        speaker_segments = []

        for turn, _, speaker in diarization.itertracks(
            yield_label=True
        ):

            speaker_segments.append(
                {
                    "start": float(
                        turn.start
                    ),

                    "end": float(
                        turn.end
                    ),

                    "speaker": speaker
                }
            )

        return speaker_segments

    except Exception as e:

        print(
            "Diarization error:",
            e
        )

        return []


# ============================================================
# FIND SPEAKER FOR TRANSCRIPT SEGMENT
# ============================================================

def find_speaker(
    transcript_segment,
    speaker_segments
):

    if not speaker_segments:

        return "Unknown"

    start = transcript_segment[
        "start"
    ]

    end = transcript_segment[
        "end"
    ]

    best_speaker = None

    best_overlap = 0

    for speaker_segment in speaker_segments:

        speaker_start = speaker_segment[
            "start"
        ]

        speaker_end = speaker_segment[
            "end"
        ]

        overlap_start = max(
            start,
            speaker_start
        )

        overlap_end = min(
            end,
            speaker_end
        )

        overlap = max(
            0,
            overlap_end - overlap_start
        )

        if overlap > best_overlap:

            best_overlap = overlap

            best_speaker = speaker_segment[
                "speaker"
            ]

    if best_speaker is None:

        return "Unknown"

    return best_speaker


# ============================================================
# NORMALIZE SPEAKER NAMES
# ============================================================

def normalize_speakers(conversation):

    speakers = []

    for item in conversation:

        speaker = item.get(
            "speaker",
            "Unknown"
        )

        if speaker != "Unknown" and speaker not in speakers:

            speakers.append(
                speaker
            )

    # First detected speaker = Interviewer
    # Second detected speaker = Candidate

    mapping = {}

    if len(speakers) >= 1:

        mapping[
            speakers[0]
        ] = "Interviewer"

    if len(speakers) >= 2:

        mapping[
            speakers[1]
        ] = "Candidate"

    # Additional speakers are kept visible
    # instead of pretending they are candidate/interviewer.

    for speaker in speakers[2:]:

        mapping[
            speaker
        ] = speaker

    for item in conversation:

        original = item.get(
            "speaker",
            "Unknown"
        )

        item["speaker"] = mapping.get(
            original,
            "Unknown"
        )

    return conversation


# ============================================================
# CREATE COMBINED CONVERSATION
# ============================================================

def create_two_person_conversation(
    transcript_segments,
    speaker_segments
):

    conversation = []

    for segment in transcript_segments:

        text = segment.get(
            "text",
            ""
        ).strip()

        if not text:

            continue

        speaker = find_speaker(
            segment,
            speaker_segments
        )

        conversation.append(
            {
                "start": segment.get(
                    "start",
                    0
                ),

                "end": segment.get(
                    "end",
                    0
                ),

                "speaker": speaker,

                "text": text
            }
        )

    conversation = normalize_speakers(
        conversation
    )

    return conversation


# ============================================================
# FORMAT TIME
# ============================================================

def format_timestamp(seconds):

    try:

        seconds = float(
            seconds
        )

    except Exception:

        seconds = 0

    minutes = int(
        seconds // 60
    )

    remaining_seconds = int(
        seconds % 60
    )

    return (
        f"{minutes:02d}:"
        f"{remaining_seconds:02d}"
    )


# ============================================================
# GENERATE TRANSCRIPT TEXT
# ============================================================

def generate_transcript_text(
    conversation
):

    lines = []

    lines.append(
        "========================================"
    )

    lines.append(
        "       PREPTALK AI"
    )

    lines.append(
        "   TWO-PERSON INTERVIEW TRANSCRIPT"
    )

    lines.append(
        "========================================"
    )

    lines.append("")

    for item in conversation:

        timestamp = format_timestamp(
            item.get(
                "start",
                0
            )
        )

        speaker = item.get(
            "speaker",
            "Unknown"
        )

        text = item.get(
            "text",
            ""
        )

        lines.append(
            f"[{timestamp}] {speaker}:"
        )

        lines.append(
            text
        )

        lines.append("")

    return "\n".join(
        lines
    )


# ============================================================
# SAVE TWO-PERSON TRANSCRIPT
# ============================================================

def save_two_person_transcript(
    conversation,
    output_path
):

    try:

        transcript = generate_transcript_text(
            conversation
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                transcript
            )

        return True

    except Exception as e:

        print(
            "Transcript save error:",
            e
        )

        return False


# ============================================================
# SAVE TWO-PERSON METADATA
# ============================================================

def save_two_person_metadata(
    metadata,
    output_path
):

    try:

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4,
                ensure_ascii=False
            )

        return True

    except Exception as e:

        print(
            "Metadata save error:",
            e
        )

        return False