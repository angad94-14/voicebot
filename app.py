"""
Streamlit Voice Agent POC for Loan Company Customer Service
Using OpenAI Agents Framework with Tracing and Custom TTS
"""
import streamlit as st
import numpy as np
import sounddevice as sd
import asyncio
import time
from loan_agent import loan_service_agent, triage_agent, CUSTOMER_DB
from agents.voice import AudioInput, SingleAgentVoiceWorkflow, VoicePipeline, VoicePipelineConfig, TTSModelSettings
from agents import set_default_openai_key, trace
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="ABC Lending - Voice Assistant",
    page_icon="🎙️",
    layout="wide"
)

# Set OpenAI key
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    set_default_openai_key(openai_key)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'voice_pipeline' not in st.session_state:
    st.session_state.voice_pipeline = None

# Define custom TTS model settings for natural, friendly voice
custom_tts_settings = TTSModelSettings(
    instructions=(
        "Never use words or phrases from other languages - English only."
        "Personality: Professional, friendly, and helpful loan company customer service representative. "
        "Tone: Warm, reassuring, and trustworthy, making customers feel comfortable discussing their financial needs. "
        "Pronunciation: Clear and articulate, ensuring financial terms and numbers are easily understood. "
        "Tempo: Speak at a moderate pace with brief pauses after important information like account balances or payment dates. "
        "Emotion: Empathetic and supportive, especially when discussing payment difficulties or loan applications. "
        "Style: Conversational yet professional, like a knowledgeable friend helping with financial matters."
    )
)



def initialize_voice_pipeline():
    """Initialize the voice pipeline with OpenAI agents and custom TTS settings"""
    try:
        # Create voice pipeline configuration with custom TTS
        voice_pipeline_config = VoicePipelineConfig(tts_settings=custom_tts_settings)

        # Create workflow and pipeline
        workflow = SingleAgentVoiceWorkflow(triage_agent)
        pipeline = VoicePipeline(workflow=workflow, config=voice_pipeline_config)

        st.success("✅ Voice pipeline initialized with custom TTS settings")
        return pipeline
    except Exception as e:
        st.error(f"Failed to initialize voice pipeline: {e}")
        return None

async def process_voice_input_streaming(recording):
    """Process voice input using the original OpenAI agents approach with tracing"""
    try:
        print('---- Processing voice input ----')

        # voice_pipeline = initialize_voice_pipeline()

        # Initialize voice pipeline if not already done
        st.session_state.voice_pipeline = initialize_voice_pipeline()  # Store in session state for later use
        if not st.session_state.voice_pipeline:
            st.error("❌ Voice pipeline initialization failed")
            return "Voice pipeline initialization failed"
        print("Voice pipeline ready")

        # Create AudioInput
        print("Creating AudioInput from buffer...")
        audio_input = AudioInput(buffer=recording)

        # Process through pipeline with OpenAI tracing
        print("Running voice pipeline with tracing...")

        with trace("ABC Lending Voice Assistant"):
            # Run the voice pipeline on the Audio Input
            result = await st.session_state.voice_pipeline.run(audio_input)
            print("Pipeline processing complete")

        # Transfer the streamed result into chunks of audio (like original)
        response_chunks = []
        response_text = ""
        event_count = 0

        async for event in result.stream():
            event_count += 1
            print(f" Event #{event_count}: {type(event).__name__}")

            # Handle different event types
            if hasattr(event, 'type'):
                print(f" ----   Event type: {event.type} ----")

                if event.type == "voice_stream_event_audio":
                    response_chunks.append(event.data)
                    print(f"Audio chunk: {len(event.data)} samples")

                elif event.type == "voice_stream_event_text" or "text" in str(event.type).lower():
                    if hasattr(event, 'data'):
                        response_text += str(event.data)
                        print(f" **** Text: {event.data}")

        st.write(f"📊 Total events: {event_count}")
        st.write(f"📊 Audio chunks: {len(response_chunks)}")
        st.write(f"📊 Response text length: {len(response_text)}")

        # Play response audio (like original)
        if response_chunks:
            st.write("🔊 Playing assistant response...")
            try:
                response_audio = np.concatenate(response_chunks, axis=0)
                st.write(f"🎵 Response audio shape: {response_audio.shape}")

                # Play using sounddevice (like original)
                sd.play(response_audio, samplerate=24000)
                print("Audio response playing with custom TTS voice...")

            except Exception as audio_error:
                st.warning(f"⚠️ Audio playback failed: {audio_error}")

        # Return text response if available
        final_response = response_text if response_text else "I processed your voice input successfully."
        st.write(f"✅ Final response: {final_response}")
        return final_response

    except Exception as e:
        error_msg = f"❌ Voice processing error: {e}"
        st.error(error_msg)
        import traceback
        st.error(f"🔍 Traceback: {traceback.format_exc()}")
        return error_msg

def record_audio_streaming(duration=5):
    """Record audio using streaming approach"""
    try:
        # Get sample rate
        samplerate = int(sd.query_devices(kind='input')['default_samplerate'])
        st.write(f"🎵 Using sample rate: {samplerate}")

        st.info(f"🎤 Recording for {duration} seconds... Speak now!")

        recorded_chunks = []

        with sd.InputStream(
            samplerate=samplerate,
            channels=1,
            dtype='int16',
            callback=lambda indata, frames, time, status: recorded_chunks.append(indata.copy())
        ):
            # Record for the specified duration
            import time as time_module
            time_module.sleep(duration)

        # Concatenate chunks into single buffer
        if recorded_chunks:
            recording = np.concatenate(recorded_chunks, axis=0)
            st.success("✅ Recording complete!")
            st.write(f"📊 Recorded {len(recording)} samples")
            st.write(f"📊 Recording shape: {recording.shape}")
            st.write(f"📊 Sample rate: {samplerate}")
            return recording, samplerate
        else:
            st.error("❌ No audio data recorded")
            return None, None

    except Exception as e:
        st.error(f"❌ Recording failed: {e}")
        return None, None

def main():
    # Header
    st.title("🎙️ ABC Lending Voice Assistant")
    st.markdown("*AI-powered customer service for your loan account*")

    # API Key check
    if not openai_key:
        st.error("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in your environment.")
        st.stop()

    # Sidebar - Customer Info
    with st.sidebar:
        st.header("📋 Demo Customer Accounts")
        st.markdown("*Use these account numbers when speaking to the agent*")

        for user_id, account in CUSTOMER_DB.items():
            with st.expander(f"{account['name']} - {user_id}"):
                st.write(f"**Balance:** ${account['loan_balance']:,.2f}")
                st.write(f"**Payment:** ${account['monthly_payment']:,.2f}")
                st.write(f"**Status:** {account['account_status']}")
                st.write(f"**Credit Score:** {account['credit_score']}")

        st.markdown("---")
        st.markdown("**Sample Questions:**")
        st.markdown("- *What's my account balance?*")
        st.markdown("- *When is my next payment due?*")
        st.markdown("- *I want to apply for a loan*")
        st.markdown("- *Can I make a payment?*")

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("🎙️ Voice Interaction")

        # Recording controls
        col_a, col_b = st.columns(2)

        with col_a:
            duration = st.slider("Recording Duration (seconds)", 3, 10, 5)

        with col_b:
            if st.button("🎤 Record & Process",
                        disabled=st.session_state.is_recording,
                        type="primary"):

                try:
                    # Set recording state to true when button is clicked
                    st.session_state.is_recording = True

                    # Record audio using streaming approach
                    recording, samplerate = record_audio_streaming(duration)

                    if recording is not None and samplerate is not None:
                        # Add user indicator to conversation
                        st.session_state.conversation_history.append({
                            "type": "user",
                            "message": "[Voice Input Recorded]",
                            "timestamp": time.time()
                        })

                        # Process voice input using original approach
                        with st.spinner("🤖 Processing your request..."):
                            try:
                                # Run async function
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                response = loop.run_until_complete(
                                    process_voice_input_streaming(recording, samplerate)
                                )
                                loop.close()

                                # Add response to conversation
                                st.session_state.conversation_history.append({
                                    "type": "assistant",
                                    "message": response,
                                    "timestamp": time.time()
                                })

                            except Exception as e:
                                st.error(f"Voice processing failed: {e}")
                                # Add error response to conversation
                                st.session_state.conversation_history.append({
                                    "type": "assistant",
                                    "message": f"Error: {str(e)}",
                                    "timestamp": time.time()
                                })
                    else:
                        st.error("Failed to record audio")

                except Exception as e:
                    st.error(f"Recording failed: {e}")

                finally:
                    # Always reset recording state, regardless of success/failure
                    st.session_state.is_recording = False
                    st.rerun()

        if st.session_state.is_recording:
            st.warning("🔴 Processing... Please wait")

    with col2:
        st.header("🤖 Agent Status")

        # Agent status
        if st.session_state.voice_pipeline:
            st.success("✅ Voice Agent Online")
            st.info("🎤 Custom TTS: Professional & Friendly")
        else:
            st.warning("⚠️ Initializing Agent...")

        # System info
        st.markdown("**Capabilities:**")
        st.markdown("- Account balance inquiries")
        st.markdown("- Payment information")
        st.markdown("- Payment processing")
        st.markdown("- Loan applications")
        st.markdown("- Credit eligibility checks")

        st.markdown("---")
        st.subheader("🔧 Debug & Testing")

        if st.button("🎤 Test Microphone Only"):
            recording, samplerate = record_audio_streaming(3)
            if recording is not None:
                st.success("Microphone test successful!")
                st.write(f"Recorded {len(recording)} samples at {samplerate}Hz")
                # Play back the recording
                import io
                import soundfile as sf

                # Save to an in-memory WAV buffer
                buffer = io.BytesIO()
                sf.write(buffer, recording, samplerate, format='WAV')
                st.audio(buffer)

                # st.audio(recording.astype(np.float32), sample_rate=samplerate)
            else:
                st.error("Microphone test failed!")

        if st.button("🤖 Test Agent Only"):
            if initialize_voice_pipeline():
                st.success("Agent initialization successful!")
            else:
                st.error("Agent initialization failed!")

        if st.button("📋 Show System Info"):
            st.write("**Python Audio Devices:**")
            try:
                devices = sd.query_devices()
                st.write(devices)
            except Exception as e:
                st.error(f"Could not query audio devices: {e}")

            st.write("**Environment Variables:**")
            st.write(f"OpenAI Key Set: {'Yes' if openai_key else 'No'}")


    # Conversation History
    st.header("💬 Conversation History")

    if st.session_state.conversation_history:
        # Display conversation
        for entry in st.session_state.conversation_history:
            if entry["type"] == "user":
                st.chat_message("user").write(entry["message"])
            elif entry["type"] == "assistant":
                st.chat_message("assistant").write(entry["message"])

        # Clear conversation
        if st.button("🗑️ Clear History"):
            st.session_state.conversation_history = []
            st.rerun()
    else:
        st.info("💡 Start a conversation by recording your voice or typing a message!")
        st.markdown("**Try saying:** *'Hello, I need help with my account'*")

if __name__ == "__main__":
    main()