import sounddevice as sd
from scipy.io.wavfile import write


def record_audio(duration, output_path="recording.wav"):
    """
    Record microphone audio and save it as a WAV file.
    """

    sample_rate = 44100

    print("🎤 Recording started...")

    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    write(
        output_path,
        sample_rate,
        recording
    )

    print(f"✅ Recording saved as {output_path}")

    return output_path