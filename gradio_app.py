from dotenv import load_dotenv
load_dotenv()

import os
import gradio as gr
import shutil
import base64
import tempfile
from gtts import gTTS
from groq import Groq

# Try to import ElevenLabs, fallback to gTTS if not available
try:
    import elevenlabs
    from elevenlabs.client import ElevenLabs
    ELEVENLABS_API_KEY = os.environ.get("ELEVEN_API_KEY")
    ELEVENLABS_AVAILABLE = bool(ELEVENLABS_API_KEY)
    print(f"ElevenLabs available: {ELEVENLABS_AVAILABLE}")
except ImportError:
    ELEVENLABS_AVAILABLE = False
    print("ElevenLabs not available, using gTTS")

# Get API keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

system_prompt = """You have to act as a professional doctor, i know you are not but this is for learning purpose. 
            What's in this image?. Do you find anything wrong with it medically? 
            If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
            your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
            Donot say 'In the image I see' but say 'With what I see, I think you have ....'
            Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
            Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""

def encode_image(image_path):   
    """Encode image to base64"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def analyze_image_with_query(query, model, encoded_image):
    """Analyze image using Groq API"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": query
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                        },
                    },
                ],
            }
        ]
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY):
    """Transcribe audio using Groq API"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        with open(audio_filepath, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=stt_model,
                file=audio_file,
                language="en"
            )
        return transcription.text
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

def text_to_speech_with_elevenlabs(input_text, output_filepath):
    """Generate speech using ElevenLabs API"""
    try:
        if not ELEVENLABS_AVAILABLE:
            raise Exception("ElevenLabs not available")
            
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        audio = client.generate(
            text=input_text,
            voice="Aria",
            output_format="mp3_22050_32",
            model="eleven_turbo_v2"
        )
        elevenlabs.save(audio, output_filepath)
        return output_filepath
        
    except Exception as e:
        print(f"ElevenLabs error: {e}. Falling back to gTTS.")
        return text_to_speech_with_gtts(input_text, output_filepath)

def text_to_speech_with_gtts(input_text, output_filepath):
    """Fallback TTS using Google Text-to-Speech"""
    try:
        language = "en"
        audioobj = gTTS(
            text=input_text,
            lang=language,
            slow=False
        )
        audioobj.save(output_filepath)
        return output_filepath
    except Exception as e:
        print(f"gTTS error: {e}")
        return None

def process_inputs(audio_filepath, image_filepath):
    """Main processing function for Gradio interface"""
    
    # Initialize responses
    speech_to_text_output = "No audio provided"
    doctor_response = "No analysis available"
    audio_file = None
    
    # Transcribe audio if provided
    if audio_filepath:
        try:
            speech_to_text_output = transcribe_with_groq(
                stt_model="whisper-large-v3",
                audio_filepath=audio_filepath,
                GROQ_API_KEY=GROQ_API_KEY
            )
        except Exception as e:
            speech_to_text_output = f"Error transcribing audio: {str(e)}"

    # Process image if provided
    if image_filepath:
        try:
            # Create a temporary copy of the image to avoid permission issues
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_image_path = temp_file.name
                
            # Copy the uploaded image
            shutil.copy2(image_filepath, temp_image_path)
            
            # Encode and analyze the image
            encoded_image = encode_image(temp_image_path)
            if encoded_image:
                doctor_response = analyze_image_with_query(
                    query=system_prompt + " " + speech_to_text_output, 
                    encoded_image=encoded_image, 
                    model="meta-llama/llama-4-scout-17b-16e-instruct"
                )
            else:
                doctor_response = "Error encoding image for analysis"
                
            # Clean up temporary image
            try:
                os.unlink(temp_image_path)
            except:
                pass
                
        except Exception as e:
            doctor_response = f"Error processing image: {str(e)}"
    else:
        doctor_response = "No image provided for analysis. Please upload an image."

    # Generate audio response
    if doctor_response and not doctor_response.startswith("Error") and not doctor_response.startswith("No image"):
        try:
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
                output_audio_path = temp_audio.name
                
            audio_file = text_to_speech_with_elevenlabs(
                input_text=doctor_response, 
                output_filepath=output_audio_path
            )
            
        except Exception as e:
            print(f"Error generating audio: {e}")
            audio_file = None

    return speech_to_text_output, doctor_response, audio_file

# Create Gradio interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="🎤 Record your question"),
        gr.Image(type="filepath", label="📷 Upload medical image")
    ],
    outputs=[
        gr.Textbox(label="📝 Your Question (Speech to Text)", lines=3),
        gr.Textbox(label="🩺 Doctor's Analysis", lines=5),
        gr.Audio(label="🔊 Doctor's Voice Response")
    ],
    title="🏥 AI Doctor with Vision and Voice",
    description="Upload a medical image and ask a question using your microphone. The AI doctor will analyze the image and provide a voice response.",
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    # Check if required API keys are available
    if not GROQ_API_KEY:
        print("⚠️  WARNING: GROQ_API_KEY not found in environment variables")
    
    print("🚀 Starting AI Doctor application...")
    print("📡 Server will be available at: http://127.0.0.1:7860")
    
    iface.launch(
        debug=True, 
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        quiet=False
    )