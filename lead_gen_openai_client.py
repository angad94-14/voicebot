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


client = get_openai_client()

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
if "language_preference" not in st.session_state:
    st.session_state.language_preference = "English"

# Hindi-English financial services knowledge
HINDI_SYSTEM_PROMPT = """You are a helpful bilingual voice assistant for a financial services company in India. You can communicate in both Hindi and English based on user preference.

Our services include:
हमारी सेवाएं / Our Services:
- व्यक्तिगत लोन (Personal Loans): ₹50,000 तक का असुरक्षित लोन / Unsecured loans up to ₹50,000
- व्यापारिक लोन (Business Loans): व्यापार विस्तार के लिए फंडिंग / Funding for business expansion  
- ऋण समेकन (Debt Consolidation): कई ऋणों को एक भुगतान में मिलाना / Combine multiple debts into one payment

Key eligibility / पात्रता मापदंड:
- 18+ वर्ष आयु / 18+ years old
- भारतीय निवासी / Indian resident  
- स्थिर आय / Steady income
- क्रेडिट स्कोर 600+ / Credit score 600+

IMPORTANT INSTRUCTIONS:
- Respond in the same language the user speaks (Hindi or English)
- If user speaks Hindi, respond in Hindi with some English financial terms
- If user speaks English, respond in English
- Keep responses conversational and under 2 sentences for voice delivery
- For Hindi speakers, use familiar terms like "लोन", "ब्याज दर", "किस्त"
- When someone shows interest, offer to collect their information
- Be respectful and use appropriate Hindi honorifics like "जी", "आप"

Available demo account IDs for testing: demo123, biz456, consol789"""

ENGLISH_SYSTEM_PROMPT = """You are a helpful voice assistant for a financial services company specializing in loans.

Our services include:
- Personal Loans: Unsecured loans up to $50,000 for various needs
- Business Loans: Funding solutions for business growth and expansion  
- Debt Consolidation: Combine multiple debts into one manageable payment

Key eligibility requirements:
- 18+ years old, US/India resident
- Steady income, Credit score 600+
- Debt-to-income ratio below 40%

Keep responses conversational and under 2 sentences for voice delivery.
When someone shows interest in applying, offer to collect their information.
Available demo account IDs: demo123, biz456, consol789"""

# Function definitions for OpenAI
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
    },
    {
        "name": "lookup_account_info",
        "description": "Look up account information using account ID",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string", "description": "Account ID or access code"}
            },
            "required": ["account_id"]
        }
    }
]


def detect_language(text):
    """Simple language detection"""
    hindi_chars = any(ord(char) >= 0x0900 and ord(char) <= 0x097F for char in text)
    return "Hindi" if hindi_chars else "English"


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

        # Save to file
        save_leads_to_file()

        name = arguments.get('name', '')
        loan_type = arguments.get('loan_type', 'loan')

        if detected_language == "Hindi":
            return f"धन्यवाद {name} जी! मैंने आपकी {loan_type} की जानकारी ले ली है। हमारा लोन एडवाइजर 24 घंटे में आपसे संपर्क करेगा।"
        else:
            return f"Thank you {name}! I've captured your information for a {loan_type}. A loan advisor will contact you within 24 hours."

    elif function_name == "lookup_account_info":
        account_id = arguments.get("account_id", "").lower()
        # Mock account data
        accounts = {
            "demo123": {
                "balance": 18750,
                "payment": 485,
                "due_date": "2025-06-15"
            }
        }

        account = accounts.get(account_id)
        if account:
            if detected_language == "Hindi":
                return f"आपका खाता विवरण: वर्तमान शेष ₹{account['balance']:,}, अगली किस्त ₹{account['payment']} दिनांक {account['due_date']} को देनी है।"
            else:
                return f"Your account info: Current balance ₹{account['balance']:,}, next payment ₹{account['payment']} due on {account['due_date']}."
        else:
            if detected_language == "Hindi":
                return "खुशी! इस आईडी से कोई खाता नहीं मिला। कृपया अपना खाता आईडी जांचें।"
            else:
                return "Sorry! I couldn't find an account with that ID. Please check your account ID."

    return "I'm sorry, I couldn't process that request."


def save_leads_to_file():
    """Save leads to JSON file"""
    try:
        with open('captured_leads.json', 'w', encoding='utf-8') as f:
            json.dump(st.session_state.leads, f, indent=2, ensure_ascii=False)
    except Exception as e:
        pass  # Silent fail for POC


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
            # Add language parameter for better Hindi recognition
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="hi" if language_hint == "Hindi" else None  # Hindi language code
            )
        return transcript.text
    except Exception as e:
        st.error(f"Transcription failed: {str(e)}")
        return None


def get_ai_response(user_message, conversation_history, detected_language):
    """Get response from OpenAI GPT-4 with language awareness"""
    try:
        # Choose system prompt based on detected language
        system_prompt = HINDI_SYSTEM_PROMPT if detected_language == "Hindi" else ENGLISH_SYSTEM_PROMPT

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        for msg in conversation_history[-6:]:  # Keep last 6 messages for context
            messages.append({"role": "user", "content": msg["user"]})
            messages.append({"role": "assistant", "content": msg["assistant"]})

        # Add current message
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            functions=lead_capture_functions,
            function_call="auto",
            max_tokens=200,
            temperature=0.7
        )

        message = response.choices[0].message

        # Handle function calls
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


def text_to_speech(text, language_hint=None):
    """Convert text to speech - note: OpenAI TTS may not sound natural for Hindi"""
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",  # Best voice for multilingual
            input=text
        )

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        response.stream_to_file(temp_file.name)
        return temp_file.name
    except Exception as e:
        st.error(f"Text-to-speech failed: {str(e)}")
        return None


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
st.markdown("**Ask me about loans in Hindi or English! / हिंदी या अंग्रेजी में लोन के बारे में पूछें!**")

# Language preference selector
col1, col2 = st.columns([2, 1])
with col1:
    st.session_state.language_preference = st.selectbox(
        "Preferred Language / भाषा चुनें:",
        ["Auto-detect", "English", "Hindi / हिंदी"],
        index=0
    )

with col2:
    if st.session_state.language_preference == "Hindi / हिंदी":
        st.info("⚠️ TTS may sound English-accented")

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
                    # Determine language hint for Whisper
                    language_hint = None
                    if st.session_state.language_preference == "Hindi / हिंदी":
                        language_hint = "Hindi"

                    user_text = transcribe_audio(temp_audio_path, language_hint)

                    if user_text:
                        # Detect actual language used
                        detected_language = detect_language(user_text)
                        if st.session_state.language_preference == "Auto-detect":
                            detected_language = detect_language(user_text)
                        elif st.session_state.language_preference == "Hindi / हिंदी":
                            detected_language = "Hindi"
                        else:
                            detected_language = "English"

                        ai_response = get_ai_response(user_text, st.session_state.conversation, detected_language)
                        tts_audio_path = text_to_speech(ai_response, detected_language)

                        st.session_state.conversation.append({
                            "user": user_text,
                            "assistant": ai_response,
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "language": detected_language
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
                st.markdown(f"**{msg['timestamp']}** {lang_flag}")
            with col2:
                st.markdown(f"**You:** {msg['user']}")
                st.markdown(f"**Assistant:** {msg['assistant']}")
            st.markdown("")

# Control buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.session_state.conversation and st.button("🗑️ Clear / साफ़ करें"):
        st.session_state.conversation = []
        st.session_state.current_lead = {}
        st.rerun()

# Instructions
with st.expander("ℹ️ How to Use / उपयोग करने का तरीका"):
    st.markdown("""
    **English Instructions:**
    1. Select your preferred language or use auto-detect
    2. Click "Start Recording" and speak clearly for 5-10 seconds
    3. Click "Process Voice" to get AI response
    4. The assistant will respond in the same language you spoke

    **हिंदी निर्देश:**
    1. अपनी पसंदीदा भाषा चुनें या ऑटो-डिटेक्ट का उपयोग करें
    2. "रिकॉर्डिंग शुरू करें" पर क्लिक करें और 5-10 सेकंड स्पष्ट रूप से बोलें
    3. AI उत्तर पाने के लिए "आवाज़ प्रोसेस करें" पर क्लिक करें
    4. असिस्टेंट उसी भाषा में जवाब देगा जिसमें आपने बात की है

    **Sample Questions / नमूना प्रश्न:**
    - "What loan options do you have?" / "आपके पास कौन से लोन के विकल्प हैं?"
    - "I need a personal loan" / "मुझे व्यक्तिगत लोन चाहिए"
    - "Account ID demo123 ki jankari dijiye"

    **Note:** Hindi TTS may sound English-accented due to OpenAI limitations.
    """)

# Environment check
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")