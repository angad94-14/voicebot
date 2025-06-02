import asyncio
import os
import json
import logging
from datetime import datetime
from typing import AsyncGenerator, Optional
from dotenv import load_dotenv
from pipecat.audio.vad.vad_analyzer import VADParams

# Pipecat real-time framework
from pipecat.frames.frames import (
    AudioRawFrame,
    TextFrame,
    TranscriptionFrame,
    Frame,
    UserAudioRawFrame
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.audio.vad.silero import SileroVADAnalyzer

# Your existing business logic
import openai

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeHindiLoanAgent(FrameProcessor):
    """
    Real-time voice loan agent with all your current features:
    - Hindi/English bilingual support
    - Lead capture with function calling
    - Enhanced account lookup
    - Conversation history
    - Language detection
    """

    def __init__(self, context: OpenAILLMContext):
        super().__init__()

        # OpenAI client for function calling
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.context = context

        # Your existing data structures
        self.conversation_history = []
        self.leads = []
        self.current_lead = {}
        self.session_id = None

        # Load your enhanced account data (same as current app)
        self.enhanced_accounts = self._load_enhanced_accounts()

        # Initialize with your system prompt
        self._setup_system_prompt()

        # Lead capture and account lookup functions (same as current)
        self.functions = [
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
                "description": "Look up detailed account information using account ID. Use when customer asks about balance, payments, account status, or loan details.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "account_id": {"type": "string", "description": "Account ID provided by customer"},
                        "info_type": {"type": "string",
                                      "description": "Type of info requested: balance, payment, status, details, history"}
                    },
                    "required": ["account_id"]
                }
            }
        ]

    def _load_enhanced_accounts(self):
        """Load the same enhanced account data from your current app"""
        return {
            "accounts": {
                "demo123": {
                    "customer_info": {
                        "name": "Rajesh Kumar",
                        "email": "rajesh.kumar@email.com",
                        "phone": "+91-9876543210"
                    },
                    "loan_details": {
                        "loan_type": "Personal Loan",
                        "original_amount": 350000,
                        "current_balance": 187500,
                        "interest_rate": "12.5%",
                        "monthly_emi": 8750,
                        "remaining_payments": 24,
                        "next_payment_date": "2025-06-15",
                        "account_status": "Active"
                    }
                },
                "biz456": {
                    "customer_info": {
                        "name": "Priya's Fashion Store",
                        "contact_person": "Priya Sharma",
                        "email": "priya@fashionstore.com",
                        "phone": "+91-8765432109"
                    },
                    "loan_details": {
                        "loan_type": "Business Loan",
                        "original_amount": 750000,
                        "current_balance": 525000,
                        "interest_rate": "14.5%",
                        "monthly_emi": 18500,
                        "remaining_payments": 36,
                        "next_payment_date": "2025-06-01",
                        "account_status": "Active"
                    }
                },
                "consol789": {
                    "customer_info": {
                        "name": "Amit Singh",
                        "email": "amit.singh@email.com",
                        "phone": "+91-7654321098"
                    },
                    "loan_details": {
                        "loan_type": "Debt Consolidation",
                        "original_amount": 450000,
                        "current_balance": 280000,
                        "interest_rate": "15.5%",
                        "monthly_emi": 12500,
                        "remaining_payments": 28,
                        "next_payment_date": "2025-06-10",
                        "account_status": "Active"
                    }
                },
                "new890": {
                    "customer_info": {
                        "name": "Sunita Patel",
                        "email": "sunita.patel@email.com",
                        "phone": "+91-6543210987"
                    },
                    "loan_details": {
                        "loan_type": "Personal Loan",
                        "original_amount": 200000,
                        "current_balance": 195000,
                        "interest_rate": "11.5%",
                        "monthly_emi": 5500,
                        "remaining_payments": 41,
                        "next_payment_date": "2025-06-25",
                        "account_status": "Active - New Customer"
                    }
                }
            }
        }

    def _setup_system_prompt(self):
        """Initialize the conversation with your system prompt"""
        system_prompt = """You are a helpful bilingual voice assistant for a financial services company in India. You can communicate in both Hindi and English based on user preference.

Our services include:
हमारी सेवाएं / Our Services:
- व्यक्तिगत लोन (Personal Loans): ₹50,000 तक का असुरक्षित लोन / Unsecured loans up to ₹50,000
- व्यापारिक लोन (Business Loans): व्यापार विस्तार के लिए फंडिंग / Funding for business expansion  
- ऋण समेकन (Debt Consolidation): कई ऋणों को एक भुगतान में मिलाना / Combine multiple debts into one payment

Available demo accounts: demo123 (Rajesh Kumar), biz456 (Priya's Fashion Store), consol789 (Amit Singh), new890 (Sunita Patel)

CRITICAL REAL-TIME INSTRUCTIONS:
- ALWAYS respond in maximum 8-10 words ONLY for natural conversation flow
- Respond in the same language the user speaks (Hindi or English)
- For Hindi speakers, use familiar terms like "लोन", "ब्याज दर", "किस्त"
- When someone shows interest, say: "Name and phone please" or "नाम और फोन नंबर बताएं"
- Be respectful and use "जी", "आप" in Hindi
- For account queries, use the lookup_account_info function
- Keep responses extremely brief for real-time voice delivery
"""

        # Add system message to context
        self.context.add_message({"role": "system", "content": system_prompt})

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames - SIMPLIFIED VERSION"""

        try:
            # Only process TranscriptionFrame - let everything else pass through
            if isinstance(frame, TranscriptionFrame):
                # User spoke - process their speech in real-time
                user_text = frame.text
                logger.info(f"🗣️ User said: {user_text}")

                # Detect language (your existing logic)
                detected_language = self._detect_language(user_text)

                # Add user message to context
                self.context.add_message({"role": "user", "content": user_text})

                # Check if this requires function calling
                needs_function_call = await self._check_for_function_call(user_text, detected_language)

                if needs_function_call:
                    # Handle function call and respond
                    response = await self._handle_function_call_from_text(user_text, detected_language)
                else:
                    # Regular conversational response
                    response = await self._get_regular_response(user_text, detected_language)

                # Add to conversation history
                self._add_to_conversation_history(user_text, response, detected_language)

                # Send response back to TTS
                logger.info(f"🤖 Responding: {response}")
                await self.push_frame(TextFrame(response), direction)

            else:
                # Pass through ALL other frame types without any checks
                await self.push_frame(frame, direction)

        except Exception as e:
            logger.error(f"Error in process_frame: {e}")
            # Always pass through the frame even if there's an error
            await self.push_frame(frame, direction)

    def _detect_language(self, text):
        """Your existing language detection logic"""
        hindi_chars = sum(1 for char in text if ord(char) >= 0x0900 and ord(char) <= 0x097F)
        total_chars = len([c for c in text if c.isalpha()])

        if total_chars == 0:
            return "English"

        hindi_ratio = hindi_chars / total_chars
        return "Hindi" if hindi_ratio > 0.3 else "English"

    async def _check_for_function_call(self, user_text, detected_language):
        """Check if user input requires function calling"""
        # Quick check for keywords that indicate function calls needed
        function_keywords = [
            # Lead capture keywords
            "interested", "apply", "want loan", "need loan", "चाहिए", "अप्लाई",
            # Account lookup keywords
            "account", "balance", "payment", "due", "status", "अकाउंट", "बैलेंस", "भुगतान"
        ]

        return any(keyword.lower() in user_text.lower() for keyword in function_keywords)

    async def _handle_function_call_from_text(self, user_text, detected_language):
        """Handle function calling for lead capture or account lookup"""
        try:
            # Use GPT-4o to determine which function to call
            messages = [
                {"role": "system",
                 "content": "You are a function calling assistant. Determine if the user wants to apply for a loan (capture_lead_info) or check account information (lookup_account_info)."},
                {"role": "user", "content": user_text}
            ]

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                functions=self.functions,
                function_call="auto",
                max_tokens=50,
                temperature=0.5
            )

            message = response.choices[0].message

            if message.function_call:
                function_name = message.function_call.name
                arguments = json.loads(message.function_call.arguments)
                return self._handle_function_call(function_name, arguments, detected_language)
            else:
                return message.content

        except Exception as e:
            logger.error(f"Function call error: {e}")
            if detected_language == "Hindi":
                return "क्षमा करें, कुछ समस्या है।"
            else:
                return "Sorry, there's an issue."

    async def _get_regular_response(self, user_text, detected_language):
        """Get regular conversational response"""
        try:
            # Get recent conversation context
            recent_messages = []
            for msg in self.conversation_history[-3:]:  # Last 3 exchanges
                recent_messages.append({"role": "user", "content": msg["user"]})
                recent_messages.append({"role": "assistant", "content": msg["assistant"]})

            recent_messages.append({"role": "user", "content": user_text})

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": f"Respond in maximum 8 words in {'Hindi' if detected_language == 'Hindi' else 'English'}. Be helpful about loans."},
                    *recent_messages
                ],
                max_tokens=30,
                temperature=0.5
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Regular response error: {e}")
            if detected_language == "Hindi":
                return "समझ नहीं आया, दोबारा बताएं।"
            else:
                return "Sorry, please repeat that."

    def _handle_function_call(self, function_name, arguments, detected_language):
        """Handle function calls with bilingual responses (same logic as current app)"""
        if function_name == "capture_lead_info":
            lead_data = {
                **arguments,
                "timestamp": datetime.now().isoformat(),
                "status": "new",
                "detected_language": detected_language,
                "session_id": self.session_id
            }
            self.leads.append(lead_data)
            self.current_lead = lead_data

            # Save lead to file (optional)
            self._save_lead_to_file(lead_data)

            name = arguments.get('name', '')

            if detected_language == "Hindi":
                return f"धन्यवाद {name} जी! एडवाइजर कॉल करेगा।"
            else:
                return f"Thank you {name}! Advisor will call soon."

        elif function_name == "lookup_account_info":
            account_id = arguments.get("account_id", "").lower()
            info_type = arguments.get("info_type", "details")

            account = self.enhanced_accounts["accounts"].get(account_id)

            if account:
                customer = account["customer_info"]
                loan = account["loan_details"]

                if detected_language == "Hindi":
                    return f"{customer['name']} जी, बैलेंस ₹{loan['current_balance']:,}।"
                else:
                    return f"Hi {customer['name']}, balance ₹{loan['current_balance']:,}."
            else:
                if detected_language == "Hindi":
                    return "अकाउंट नहीं मिला, ID चेक करें।"
                else:
                    return "Account not found, check ID."

        return "I'm sorry, I couldn't process that request."

    def _add_to_conversation_history(self, user_text, assistant_response, detected_language):
        """Add exchange to conversation history"""
        self.conversation_history.append({
            "user": user_text,
            "assistant": assistant_response,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "language": detected_language,
            "session_id": self.session_id
        })

    def _save_lead_to_file(self, lead_data):
        """Save lead to JSON file (optional for POC)"""
        try:
            # Save to leads file
            leads_file = "realtime_leads.json"
            existing_leads = []

            if os.path.exists(leads_file):
                with open(leads_file, 'r', encoding='utf-8') as f:
                    existing_leads = json.load(f)

            existing_leads.append(lead_data)

            with open(leads_file, 'w', encoding='utf-8') as f:
                json.dump(existing_leads, f, indent=2, ensure_ascii=False)

            logger.info(f"Lead saved: {lead_data['name']}")

        except Exception as e:
            logger.error(f"Error saving lead: {e}")


async def main():
    """Main function to set up and run the real-time voice agent"""

    # Environment check
    required_env_vars = ["OPENAI_API_KEY"]
    for var in required_env_vars:
        if not os.getenv(var):
            logger.error(f"Missing required environment variable: {var}")
            return

    try:
        # LLM Context for conversation management
        context = OpenAILLMContext()

        # Your custom Hindi voice loan agent
        loan_agent = RealTimeHindiLoanAgent(context)

        # LLM Service
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o"
        )

        # STT Service - Try Deepgram first, fallback to other options
        stt = None
        if os.getenv("DEEPGRAM_API_KEY"):
            stt = DeepgramSTTService(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                model="nova-2",
                language="hi-en",  # Hindi-English bilingual model
                smart_format=True,
                interim_results=True
            )
            logger.info("Using Deepgram STT for ultra-low latency")
        else:
            logger.warning("Deepgram API key not found. Please set DEEPGRAM_API_KEY for optimal performance.")
            # You might want to implement a fallback STT service here

        # TTS Service - Google Cloud with your Chirp3-HD voice
        tts = None
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            tts = GoogleTTSService(
                credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
                voice_name="hi-IN-Chirp3-HD-Sadachbia",  # Your preferred Hindi voice
                language_code="hi-IN",
                speaking_rate=1.0,
                pitch=0.0
            )
            logger.info("Using Google Chirp3-HD TTS")
        else:
            logger.warning("Google Cloud credentials not found. Please set GOOGLE_APPLICATION_CREDENTIALS.")

        # VAD - Voice Activity Detection for natural conversation flow
        vad = SileroVADAnalyzer(
            sample_rate=16000,
            params=VADParams(
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=100
            )
        )

        # Transport - Daily.co for WebRTC (professional voice quality)
        transport = DailyTransport(
            room_url=os.getenv("DAILY_ROOM_URL") or "https://yourdomain.daily.co/your-room",
            token=os.getenv("DAILY_TOKEN"),  # Add token if needed
            bot_name="Hindi Voice Loan Assistant",
            params=DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                transcription_enabled=False,  # We're using Deepgram instead
                vad_enabled=True,
                vad_analyzer=vad
            )
        )

        # Build the real-time pipeline - ensure all services are available
        pipeline_stages = [transport.input()]

        if stt:
            pipeline_stages.append(stt)
        else:
            logger.error("No STT service available. Cannot proceed.")
            return

        pipeline_stages.extend([
            loan_agent,  # Process with your loan agent logic
            llm,  # Generate responses
        ])

        if tts:
            pipeline_stages.append(tts)
        else:
            logger.error("No TTS service available. Cannot proceed.")
            return

        pipeline_stages.append(transport.output())

        pipeline = Pipeline(pipeline_stages)

        # Pipeline task configuration
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True
            )
        )

        # Run the pipeline
        runner = PipelineRunner()

        logger.info("🎙️ Real-Time Hindi Voice Loan Assistant Ready!")
        logger.info("📞 Users can now have natural voice conversations")
        logger.info("🌟 Features: Hindi/English, Lead Capture, Account Lookup")
        logger.info("⚡ Ultra-low latency: ~1-3 seconds response time")

        # Set session ID for this conversation
        loan_agent.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        await runner.run(task)

    except Exception as e:
        logger.error(f"Error starting voice agent: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())