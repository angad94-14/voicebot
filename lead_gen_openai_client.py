import streamlit as st
import openai
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os
import json
from datetime import datetime
import base64
from dotenv import load_dotenv
import requests
from google.cloud import texttospeech

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Voice Loan Assistant / वॉइस लोन असिस्टेंट",
    page_icon="🎙️",
    layout="centered"
)


# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Initialize Google Cloud TTS client
@st.cache_resource
def get_google_tts_client():
    try:
        # Initialize Google Cloud TTS client
        # Requires GOOGLE_APPLICATION_CREDENTIALS environment variable
        return texttospeech.TextToSpeechClient()
    except Exception as e:
        st.warning("Google Cloud TTS not configured. Using OpenAI TTS as fallback.")
        return None


# Initialize ElevenLabs client (optional)
def get_elevenlabs_headers():
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if api_key:
        return {"Accept": "audio/mpeg", "xi-api-key": api_key}
    return None


client = get_openai_client()
google_tts_client = get_google_tts_client()
elevenlabs_headers = get_elevenlabs_headers()

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "leads" not in st.session_state:
    st.session_state.leads = []
if "current_lead" not in st.session_state:
    st.session_state.current_lead = {}
if "tts_provider" not in st.session_state:
    st.session_state.tts_provider = "auto"

# TTS Provider Configuration
TTS_PROVIDERS = {
    "auto": "Auto-select",
    "google": "Google Cloud TTS",
    "elevenlabs": "ElevenLabs",
    "openai": "OpenAI TTS"
}

# Hindi voice configurations
GOOGLE_HINDI_VOICES = {
    "hi-IN-Chirp3-HD-Sadachbia": "Hindi Female (Chirp3-HD) - Most Natural",
    "hi-IN-Wavenet-A": "Hindi Female (WaveNet)",
    "hi-IN-Wavenet-B": "Hindi Male (WaveNet)",
    "hi-IN-Wavenet-C": "Hindi Female (WaveNet)",
    "hi-IN-Wavenet-D": "Hindi Male (WaveNet)",
    "hi-IN-Neural2-A": "Hindi Female (Neural2)",
    "hi-IN-Neural2-B": "Hindi Male (Neural2)",
    "hi-IN-Neural2-C": "Hindi Female (Neural2)",
    "hi-IN-Neural2-D": "Hindi Male (Neural2)",

}

ELEVENLABS_HINDI_VOICES = {
    "pNInz6obpgDQGcFmaJgB": "Hindi Female (Adam)",
    "EXAVITQu4vr4xnSDxMaL": "Hindi Male (Sarah)"
}

# System prompts (same as before)
HINDI_SYSTEM_PROMPT = """You are a helpful bilingual voice assistant for a financial services company in India. You can communicate in both Hindi and English based on user preference.

Our services include:
हमारी सेवाएं / Our Services:
- व्यक्तिगत लोन (Personal Loans): ₹50,000 तक का असुरक्षित लोन / Unsecured loans up to ₹50,000
- व्यापारिक लोन (Business Loans): व्यापार विस्तार के लिए फंडिंग / Funding for business expansion  
- ऋण समेकन (Debt Consolidation): कई ऋणों को एक भुगतान में मिलाना / Combine multiple debts into one payment

IMPORTANT INSTRUCTIONS:
- Respond in the same language the user speaks (Hindi or English)
- Keep responses conversational and under 2 sentences for voice delivery
- For Hindi speakers, use familiar terms like "लोन", "ब्याज दर", "किस्त"
- When someone shows interest, offer to collect their information
- Be respectful and use appropriate Hindi honorifics like "जी", "आप"
"""


def detect_language(text):
    """Enhanced language detection"""
    hindi_chars = sum(1 for char in text if ord(char) >= 0x0900 and ord(char) <= 0x097F)
    total_chars = len([c for c in text if c.isalpha()])

    if total_chars == 0:
        return "English"

    hindi_ratio = hindi_chars / total_chars
    return "Hindi" if hindi_ratio > 0.3 else "English"


def google_text_to_speech(text, language_hint="Hindi"):
    """Convert text to speech using Google Cloud TTS with high-quality Hindi voices"""
    if not google_tts_client:
        return None

    try:
        # Select voice based on language
        if language_hint == "Hindi":
            voice_name = "hi-IN-Chirp3-HD-Sadachbia"  # High-quality Hindi
            language_code = "hi-IN"
        else:
            voice_name = "en-IN-Chirp3-HD-Puck"  # Indian English voice
            language_code = "en-IN"

        # Set up the synthesis input
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Build the voice request
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )

        # Select the audio config
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,  # Normal speed
            pitch=0.0  # Normal pitch
        )

        # Perform the text-to-speech request
        response = google_tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.write(response.audio_content)
        temp_file.close()

        return temp_file.name

    except Exception as e:
        st.error(f"Google TTS failed: {str(e)}")
        return None


def elevenlabs_text_to_speech(text, language_hint="Hindi"):
    """Convert text to speech using ElevenLabs API"""
    if not elevenlabs_headers:
        return None

    try:
        # Select voice based on language
        voice_id = "pNInz6obpgDQGcFmaJgB" if language_hint == "Hindi" else "21m00Tcm4TlvDq8ikWAM"

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        response = requests.post(url, json=data, headers=elevenlabs_headers)

        if response.status_code == 200:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_file.write(response.content)
            temp_file.close()
            return temp_file.name
        else:
            st.error(f"ElevenLabs TTS failed: {response.status_code}")
            return None

    except Exception as e:
        st.error(f"ElevenLabs TTS failed: {str(e)}")
        return None


def openai_text_to_speech(text, language_hint="Hindi"):
    """Fallback OpenAI TTS"""
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        response.stream_to_file(temp_file.name)
        return temp_file.name
    except Exception as e:
        st.error(f"OpenAI TTS failed: {str(e)}")
        return None


def smart_text_to_speech(text, language_hint="Hindi"):
    """Intelligently choose the best TTS provider based on language and availability"""

    if st.session_state.tts_provider == "google" and google_tts_client:
        return google_text_to_speech(text, language_hint)
    elif st.session_state.tts_provider == "elevenlabs" and elevenlabs_headers:
        return elevenlabs_text_to_speech(text, language_hint)
    elif st.session_state.tts_provider == "openai":
        return openai_text_to_speech(text, language_hint)
    else:  # Auto-select
        # Priority: Google Cloud (best Hindi) > ElevenLabs > OpenAI
        if language_hint == "Hindi":
            if google_tts_client:
                return google_text_to_speech(text, language_hint)
            elif elevenlabs_headers:
                return elevenlabs_text_to_speech(text, language_hint)
            else:
                return openai_text_to_speech(text, language_hint)
        else:  # English
            if elevenlabs_headers:
                return elevenlabs_text_to_speech(text, language_hint)
            elif google_tts_client:
                return google_text_to_speech(text, language_hint)
            else:
                return openai_text_to_speech(text, language_hint)


# Lead capture functions (same as before)
lead_capture_functions = [
    {
        "name": "capture_lead_info",
        "description": "Capture customer lead information when they express interest in applying for a loan",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Customer's full name"},
                "email": {"type": "string", "description": "Customer's email address"},
                "phone": {"type": "string", "description": "Customer's phone number"},
                "loan_type": {"type": "string", "description": "Type of loan they're interested in"},
                "loan_amount": {"type": "number", "description": "Desired loan amount"},
                "language": {"type": "string", "description": "Preferred language (Hindi/English)"}
            },
            "required": ["name", "phone", "loan_type"]
        }
    }
]


def handle_function_call(function_name, arguments, detected_language):
    """Handle function calls with bilingual responses"""
    if function_name == "capture_lead_info":
        lead_data = {
            **arguments,
            "timestamp": datetime.now().isoformat(),
            "status": "new",
            "detected_language": detected_language
        }
        st.session_state.leads.append(lead_data)
        st.session_state.current_lead = lead_data

        name = arguments.get('name', '')
        loan_type = arguments.get('loan_type', 'loan')

        if detected_language == "Hindi":
            return f"धन्यवाद {name} जी! मैंने आपकी {loan_type} की जानकारी ले ली है। हमारा लोन एडवाइजर 24 घंटे में आपसे संपर्क करेगा।"
        else:
            return f"Thank you {name}! I've captured your information for a {loan_type}. A loan advisor will contact you within 24 hours."

    return "I'm sorry, I couldn't process that request."


def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone"""
    try:
        audio_data = sd.rec(int(duration * sample_rate),
                            samplerate=sample_rate,
                            channels=1,
                            dtype=np.float32)
        sd.wait()
        return audio_data.flatten()
    except Exception as e:
        st.error(f"Recording failed: {str(e)}")
        return None


def save_audio_to_temp(audio_data, sample_rate=16000):
    """Save audio data to temporary file"""
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, audio_data, sample_rate)
        return temp_file.name
    except Exception as e:
        st.error(f"Failed to save audio: {str(e)}")
        return None


def transcribe_audio(audio_file_path, language_hint=None):
    """Convert speech to text using OpenAI Whisper with language hint"""
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="hi" if language_hint == "Hindi" else None
            )
        return transcript.text
    except Exception as e:
        st.error(f"Transcription failed: {str(e)}")
        return None


def get_ai_response(user_message, conversation_history, detected_language):
    """Get response from OpenAI GPT-4 with language awareness"""
    try:
        system_prompt = HINDI_SYSTEM_PROMPT

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        for msg in conversation_history[-6:]:
            messages.append({"role": "user", "content": msg["user"]})
            messages.append({"role": "assistant", "content": msg["assistant"]})

        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            functions=lead_capture_functions,
            function_call="auto",
            max_tokens=200,
            temperature=0.7
        )

        message = response.choices[0].message

        if message.function_call:
            function_name = message.function_call.name
            arguments = json.loads(message.function_call.arguments)
            function_response = handle_function_call(function_name, arguments, detected_language)
            return function_response
        else:
            return message.content

    except Exception as e:
        if detected_language == "Hindi":
            return "क्षमा करें, मुझे आपका अनुरोध प्रोसेस करने में कुछ समस्या हो रही है।"
        else:
            return "I'm sorry, I'm having trouble processing your request right now."


def play_audio_in_browser(audio_file_path):
    """Create audio player for browser"""
    try:
        with open(audio_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Audio playback failed: {str(e)}")


# Main UI
st.title("🎙️ Voice Loan Assistant / वॉइस लोन असिस्टेंट")
st.markdown("**Professional Hindi TTS with multiple provider support!**")

# TTS Provider Configuration
col1, col2 = st.columns([2, 1])
with col1:
    st.session_state.tts_provider = st.selectbox(
        "TTS Provider / TTS प्रदाता:",
        list(TTS_PROVIDERS.keys()),
        format_func=lambda x: TTS_PROVIDERS[x],
        index=0
    )

with col2:
    # Show provider status
    if st.session_state.tts_provider == "google" or st.session_state.tts_provider == "auto":
        if google_tts_client:
            st.success("✅ Google Cloud")
        else:
            st.warning("⚠️ Google Cloud - Not configured")

    if st.session_state.tts_provider == "elevenlabs" or st.session_state.tts_provider == "auto":
        if elevenlabs_headers:
            st.success("✅ ElevenLabs")
        else:
            st.info("⚠️ ElevnLabs - Not Configured")

# Voice interaction section
col1, col2 = st.columns([1, 1])

with col1:
    recording_duration = st.selectbox("Recording Duration / रिकॉर्डिंग समय", [5, 8, 10], index=0)
    if st.button("🎤 Start Recording / रिकॉर्डिंग शुरू करें", type="primary", disabled=st.session_state.is_recording):
        st.session_state.is_recording = True
        with st.spinner(
                f"Recording for {recording_duration} seconds... / {recording_duration} सेकंड रिकॉर्ड हो रहा है..."):
            audio_data = record_audio(duration=recording_duration)
            if audio_data is not None:
                st.session_state.audio_data = audio_data
        st.session_state.is_recording = False

with col2:
    if st.button("⏹️ Process Voice / आवाज़ प्रोसेस करें", disabled=st.session_state.audio_data is None):
        if st.session_state.audio_data is not None:
            with st.spinner("Processing your voice... / आपकी आवाज़ प्रोसेस हो रही है..."):
                temp_audio_path = save_audio_to_temp(st.session_state.audio_data)

                if temp_audio_path:
                    user_text = transcribe_audio(temp_audio_path)

                    if user_text:
                        detected_language = detect_language(user_text)
                        ai_response = get_ai_response(user_text, st.session_state.conversation, detected_language)

                        # Use smart TTS selection
                        tts_audio_path = smart_text_to_speech(ai_response, detected_language)

                        st.session_state.conversation.append({
                            "user": user_text,
                            "assistant": ai_response,
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "language": detected_language,
                            "tts_provider": st.session_state.tts_provider
                        })

                        if tts_audio_path:
                            play_audio_in_browser(tts_audio_path)
                            os.unlink(tts_audio_path)

                    os.unlink(temp_audio_path)

            st.session_state.audio_data = None

# Lead capture status
if st.session_state.current_lead:
    name = st.session_state.current_lead.get('name', 'Unknown')
    loan_type = st.session_state.current_lead.get('loan_type', 'Loan')
    lang = st.session_state.current_lead.get('detected_language', 'English')

    if lang == "Hindi":
        st.success(f"✅ लीड कैप्चर हुआ: {name} - {loan_type} आवेदन")
    else:
        st.success(f"✅ Lead captured: {name} - {loan_type} application")

# Conversation History
if st.session_state.conversation:
    st.markdown("---")
    st.subheader("💬 Conversation / बातचीत")

    for msg in reversed(st.session_state.conversation):
        with st.container():
            col1, col2 = st.columns([1, 4])
            with col1:
                lang_flag = "🇮🇳" if msg.get('language') == 'Hindi' else "🇺🇸"
                provider_icon = "🔵" if msg.get('tts_provider') == 'google' else "⚫"
                st.markdown(f"**{msg['timestamp']}** {lang_flag} {provider_icon}")
            with col2:
                st.markdown(f"**You:** {msg['user']}")
                st.markdown(f"**Assistant:** {msg['assistant']}")
            st.markdown("")

# Control buttons
if st.session_state.conversation and st.button("🗑️ Clear / साफ़ करें"):
    st.session_state.conversation = []
    st.session_state.current_lead = {}
    st.rerun()

# Sample User Journey
with st.expander("🎯 Sample User Journey / नमूना यात्रा"):
    st.markdown("""
    **English User Journey:**
    1. "What services do you offer?"
    2. "Tell me about personal loans"
    3. "What's the interest rate?"
    4. "How much can I borrow?"
    5. "What documents do I need?"
    6. "I'm interested in a ₹25,000 loan"
    7. "Yes, I want to apply"
    8. [Provides name, phone, email when asked]

    **Hindi User Journey / हिंदी यात्रा:**
    1. "आप क्या सेवाएं देते हैं?"
    2. "व्यक्तिगत लोन के बारे में बताएं"
    3. "ब्याज दर क्या है?"
    4. "कितना लोन मिल सकता है?"
    5. "कौन से डॉक्यूमेंट चाहिए?"
    6. "मुझे ₹25,000 का लोन चाहिए"
    7. "हां, मैं अप्लाई करना चाहता हूं"
    8. [नाम, फोन, ईमेल देता है]

    **Mixed Language Journey:**
    1. "What loan options हैं?"
    2. "Personal loan ke लिए क्या करना होगा?"
    3. "Interest rate kitna है?"
    4. "I need ₹30,000 ka loan"
    5. "Apply kaise करूं?"

    **Account Lookup Journey:**
    1. "Check my account balance"
    2. "My account ID is demo123"
    3. "When is my next payment due?"
    4. "What's my loan status?"
    """)

# Environment check
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")