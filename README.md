# 🏥 AI Doctor with Vision and Voice

An AI-powered medical consultation application that combines computer vision and natural language processing to analyze medical images and provide intelligent voice-based responses. This project leverages cutting-edge AI technologies to assist users with medical guidance.

## 🎯 Overview

**AI Doctor** is a state-of-the-art web application that enables users to:
- Upload medical images for AI-powered analysis
- Ask questions via voice input
- Receive professional medical insights from an AI model
- Get voice responses in natural, conversational language

The application uses **Groq's fast LLM inference** combined with **multimodal capabilities** to analyze medical conditions and provide evidence-based recommendations.

## ✨ Key Features

### 🖼️ **Medical Image Analysis**
- Upload skin condition, rash, acne, or other medical images
- AI analyzes the image and identifies potential conditions
- Provides differential diagnoses and suggested remedies
- Utilizes Meta's Llama-4 Scout 17B vision model

### 🎤 **Voice Input & Output**
- Record questions directly through microphone
- Speech-to-text transcription using Groq's Whisper-Large-v3
- AI-generated voice responses
- Support for dual TTS providers:
  - **ElevenLabs** (Premium, natural-sounding voices)
  - **Google Text-to-Speech** (Fallback option)

### 🔊 **Natural Conversation**
- AI responds as a professional doctor, not an AI bot
- Concise, actionable medical advice
- Conversational tone with proper medical terminology
- Combines patient's voice input with visual analysis

### 🎨 **User-Friendly Interface**
- Built with **Gradio** for intuitive UI
- Clean, accessible design with emojis for clarity
- Real-time processing and instant feedback
- Mobile and desktop compatible

## 🛠 Technology Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Gradio 5.12.0 |
| **Vision & LLM** | Groq API (Meta Llama-4 Scout 17B-16e-instruct) |
| **Speech-to-Text** | Groq Whisper-Large-v3 |
| **Text-to-Speech** | ElevenLabs API, Google Text-to-Speech (gTTS) |
| **Audio Processing** | PyAudio, pydub, ffmpeg |
| **Backend** | Python, FastAPI, Uvicorn |
| **Dependencies** | See requirements.txt |

## 📋 Project Structure

```
AI_Doctor/
├── gradio_app.py                 # Main Gradio application
├── brain_of_the_doctor.py        # Image analysis logic
├── voice_of_the_doctor.py        # Doctor's voice generation
├── voice_of_the_patient.py       # Patient's voice processing
├── requirements.txt              # Python dependencies
├── Pipfile & Pipfile.lock        # Pipenv configuration
├── .env                          # Environment variables (not in repo)
├── acne.jpg                      # Sample medical image
├── skin_rash.jpg                 # Sample medical image
├── dandruff-optimized.webp       # Sample medical image
└── presentation.pdf              # Project presentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip or pipenv
- API Keys for:
  - **Groq API** (Free tier available)
  - **ElevenLabs** (Optional, for premium voice)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Satyamgupta8670/AI_Doctor.git
   cd AI_Doctor
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or using Pipenv:
   ```bash
   pipenv install
   pipenv shell
   ```

4. **Setup Environment Variables**
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ELEVEN_API_KEY=your_elevenlabs_api_key_here  # Optional
   ```

### Getting API Keys

#### 🔑 Groq API Key
1. Visit [Groq Console](https://console.groq.com)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy and paste into `.env`

#### 🎵 ElevenLabs API Key (Optional)
1. Visit [ElevenLabs](https://elevenlabs.io)
2. Create a free account
3. Go to API section
4. Copy your API key
5. Add to `.env` file

## ▶️ Running the Application

```bash
python gradio_app.py
```

The application will launch at `http://127.0.0.1:7860`

### Using the Application
1. **Record Your Question**: Click the microphone icon and ask your medical question
2. **Upload Image**: Upload a medical image (skin condition, rash, etc.)
3. **Get Analysis**: Click Submit to get:
   - Transcribed question (📝)
   - Doctor's analysis (🩺)
   - Voice response (🔊)

## 📊 Core Functions

### analyze_image_with_query(query, model, encoded_image)
Analyzes medical images using Groq's multimodal LLM
- Input: Text query, image encoded in base64
- Output: Medical analysis and diagnosis

### transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY)
Converts speech to text using Whisper-Large-v3
- Input: Audio file path
- Output: Transcribed text

### text_to_speech_with_elevenlabs(input_text, output_filepath)
Generates natural-sounding voice responses
- Input: Text to convert to speech
- Output: MP3 audio file

### text_to_speech_with_gtts(input_text, output_filepath)
Fallback TTS using Google Text-to-Speech
- Automatically triggered if ElevenLabs unavailable
- No API key required

## ⚙️ Configuration

### Supported Models
- Vision LLM: meta-llama/llama-4-scout-17b-16e-instruct
- Speech-to-Text: whisper-large-v3
- TTS: ElevenLabs voice "Aria" or gTTS

### System Prompt
The AI doctor operates with this system prompt:
"You have to act as a professional doctor. What's in this image? Do you find anything wrong with it medically? If you make a differential, suggest some remedies. Keep your answer concise (max 2 sentences). Respond naturally as an actual doctor would, not as an AI bot."

## 🔄 Workflow

```
User Input (Voice + Image)
         ↓
    Speech-to-Text (Groq Whisper)
         ↓
    Image Encoding (Base64)
         ↓
    Multimodal Analysis (Groq Llama)
         ↓
    AI Doctor Response
         ↓
    Text-to-Speech (ElevenLabs/gTTS)
         ↓
    Voice Output to User
```

## 📦 Dependencies

Key packages used:
- gradio - Web interface
- groq - AI inference
- elevenlabs - Premium text-to-speech
- gtts - Fallback TTS
- pyaudio - Audio input/output
- pydub - Audio processing
- python-dotenv - Environment configuration

See requirements.txt for complete list with versions.

## 🎓 Educational Purpose

This project is designed for:
- Learning AI integration in healthcare applications
- Understanding multimodal LLM capabilities
- Exploring voice and vision APIs
- Building accessible healthcare technology

⚠️ **Disclaimer**: This application is for educational purposes only. It is NOT a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical concerns.

## 🤝 Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request

## 📝 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

## 👨‍💻 Author

**Satyam Gupta**
GitHub: @Satyamgupta8670

## 🙏 Acknowledgments

- **Groq** - For fast, reliable LLM inference
- **Meta** - For Llama models
- **ElevenLabs** - For natural text-to-speech
- **Google** - For Speech Recognition APIs
- **Gradio** - For the web interface framework

## 📞 Support

For issues, questions, or suggestions, please:
1. Check existing issues on GitHub
2. Create a new issue with detailed description
3. Contact the author via GitHub

## 🚦 Status

- ✅ Image analysis functional
- ✅ Voice input/output working
- ✅ Web interface live
- ⏳ Mobile app optimization (in progress)
- ⏳ Multi-language support (planned)

---

**Last Updated**: April 2026
**Version**: 1.0.0
