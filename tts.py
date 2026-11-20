import pyttsx3
import os
import time


def speak_text(
    text,
    output_path=None
):

    if not text:
        return

    engine = None

    try:

        engine = pyttsx3.init()

        engine.setProperty(
            "rate",
            165
        )

        engine.setProperty(
            "volume",
            1.0
        )

        if output_path:

            os.makedirs(
                os.path.dirname(output_path),
                exist_ok=True
            )

            engine.save_to_file(
                text,
                output_path
            )

        engine.say(text)

        engine.runAndWait()

        time.sleep(0.5)

        if output_path:

            print(
                f"🔊 AI audio saved: {output_path}"
            )

    except Exception as e:

        print(
            f"TTS failed: {e}"
        )

    finally:

        try:

            if engine:
                engine.stop()

        except Exception:
            pass