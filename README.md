# 🎤 PrepTalk AI – Your Personal AI Interview Coach

**PrepTalk AI** is a voice-based AI interview practice application developed to help students, freshers, and job seekers improve their interview preparation through interactive conversations with an AI assistant.

The application combines Speech-to-Text (STT), Large Language Models (LLMs), and Text-to-Speech (TTS) technologies to provide an interactive interview practice experience.

## 🚀 Features

- **Voice-Based Interaction:** Record voice input using a microphone.
- **Speech Recognition:** Convert spoken audio into text using OpenAI Whisper.
- **AI-Powered Responses:** Generate relevant responses using Google Gemini API.
- **Conversational Interaction:** Support continued interview practice through follow-up interactions.
- **Text-to-Speech:** Convert generated text responses into spoken audio.
- **AI Feedback Module:** Includes a separate module for AI-based feedback functionality.
- **Two-Person Module:** Includes a separate Python module for two-person interaction.
- **Interactive Interface:** Application interface developed using Streamlit.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Application interface |
| OpenAI Whisper | Speech-to-Text conversion |
| Google Gemini API | AI response generation |
| Text-to-Speech | Spoken response generation |
| SoundDevice | Audio recording |
| SciPy | Audio processing |
| Python-dotenv | Environment variable management |
| Git & GitHub | Version control |

## ⚙️ Working Process

1. The user opens the PrepTalk AI application.
2. The user records an interview question using the microphone.
3. The recorded audio is saved in WAV format.
4. Whisper converts the recorded speech into text.
5. The transcribed text is processed using the Gemini API.
6. Gemini generates a relevant AI response.
7. The generated response is displayed on the application.
8. The Text-to-Speech module converts the response into audio.
9. The user listens to the response and continues practising.

## 📂 Project Structure

```text
preptalkAI-2026/
│
├── app.py
├── ai_feedback.py
├── llm.py
├── record.py
├── stt.py
├── tts.py
├── two_person.py
│
├── requirements.txt
├── .dockerignore
├── dockerfile.txt
│
├── backend.zip
├── conversation.txt
├── recording.wav
├── users.json
├── users.txt
│
└── README.md
```

*Note: This structure is based on the files visible in the project repository. Sensitive files and temporary data should be excluded from the final repository.*

## 💻 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/lowlatency2022/preptalkAI-2026.git
```

**Note:** This is a private repository. You must have authorised GitHub access to clone it.

### 2. Navigate to the Project Directory

```bash
cd preptalkAI-2026
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install Required Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure API Credentials

Create a `.env` file in the project directory and add the required Gemini API key:

```env
GEMINI_API_KEY=your_api_key_here
```

Keep your API credentials private. Never upload your `.env` file to GitHub.

### 7. Run the Application

```bash
streamlit run app.py
```

## 🎯 Project Objectives

- To develop an AI-powered interview practice assistant.
- To integrate voice input with speech recognition technology.
- To generate relevant interview responses using an LLM.
- To provide spoken responses using Text-to-Speech.
- To explore conversational AI and practical AI/ML applications.
- To gain hands-on experience in Python and AI-based application development.

## 🔮 Future Enhancements

- Detailed interview performance analysis.
- Personalised interview questions based on job roles.
- Pronunciation and communication improvement suggestions.
- Interview progress tracking.
- Multilingual interview support.
- Downloadable interview feedback reports.

## 👩‍💻 Internship Details

**Organisation:** Low Latency Technologies Private Limited

**Internship Role:** Cloud Data Engineering and AI/ML Intern

**Project Title:** PrepTalk AI – Your Personal AI Interview Coach

**Project Domain:** Artificial Intelligence and Conversational AI

**Internship Duration:** 19 June 2026 – 19 November 2026

## 🤝 Project Team

Developed collaboratively by a team of two members as part of the internship project at Low Latency Technologies Private Limited.

## 📌 Repository Information

**GitHub Repository:** [preptalkAI-2026](https://github.com/lowlatency2022/preptalkAI-2026)

**Repository Owner:** lowlatency2022

**Visibility:** Private

## 📄 License

This project was developed for internship and academic purposes.
