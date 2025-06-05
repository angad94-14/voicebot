import asyncio
import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from pipecat.audio.vad.vad_analyzer import VADParams

# Pipecat real-time framework
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.audio.vad.silero import SileroVADAnalyzer

# Pipecat function calling schemas
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoanBusinessLogic:
    """
    Simple class to handle business logic - NO FrameProcessor inheritance!
    This is just a regular Python class with your business methods.
    """

    def __init__(self):
        # Your existing data structures
        self.conversation_history = []
        self.leads = []
        self.current_lead = {}
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Load your enhanced account data
        self.enhanced_accounts = self._load_enhanced_accounts()



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

    def detect_language(self, text):
        """Language detection - now just a regular method"""
        hindi_chars = sum(1 for char in text if ord(char) >= 0x0900 and ord(char) <= 0x097F)
        total_chars = len([c for c in text if c.isalpha()])

        if total_chars == 0:
            return "English"

        hindi_ratio = hindi_chars / total_chars
        return "Hindi" if hindi_ratio > 0.3 else "English"

    # Function implementations that will be called by Pipecat
    async def capture_lead_info(self, name: str, phone: str, loan_type: str, email: str = "", loan_amount: float = 0,
                                language: str = "English"):
        """Function called by OpenAI when user wants to apply for loan"""
        try:
            lead_data = {
                "name": name,
                "phone": phone,
                "loan_type": loan_type,
                "email": email,
                "loan_amount": loan_amount,
                "language": language,
                "timestamp": datetime.now().isoformat(),
                "status": "new",
                "session_id": self.session_id
            }

            self.leads.append(lead_data)
            self.current_lead = lead_data

            # Save lead to file
            self._save_lead_to_file(lead_data)

            # Return response based on language
            if language == "Hindi":
                return f"धन्यवाद {name} जी! हमारा एडवाइजर जल्दी ही आपको कॉल करेगा।"
            else:
                return f"Thank you {name}! Our advisor will call you soon."

        except Exception as e:
            logger.error(f"Error capturing lead: {e}")
            return "Sorry, there was an error. Please try again."

    async def lookup_account_info(self, account_id: str, info_type: str = "balance"):
        """Function called by OpenAI when user wants account information"""
        try:
            account_id = account_id.lower()
            account = self.enhanced_accounts["accounts"].get(account_id)

            if not account:
                return "Account not found. Please check your account ID."

            customer = account["customer_info"]
            loan = account["loan_details"]

            if info_type == "balance":
                return f"Hi {customer['name']}, your current balance is ₹{loan['current_balance']:,}."
            elif info_type == "payment":
                return f"Your next payment of ₹{loan['monthly_emi']:,} is due on {loan['next_payment_date']}."
            elif info_type == "status":
                return f"Your account status is: {loan['account_status']}"
            else:
                # Full details
                return f"Hi {customer['name']}, Account: {loan['loan_type']}, Balance: ₹{loan['current_balance']:,}, Next payment: ₹{loan['monthly_emi']:,} on {loan['next_payment_date']}"

        except Exception as e:
            logger.error(f"Error looking up account: {e}")
            return "Sorry, there was an error looking up your account."

    def _save_lead_to_file(self, lead_data):
        """Save lead to JSON file - with better error handling"""
        try:
            leads_file = "realtime_leads.json"
            existing_leads = []

            # Better file reading with error handling
            if os.path.exists(leads_file):
                try:
                    with open(leads_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:  # Only parse if file has content
                            existing_leads = json.loads(content)
                        else:
                            existing_leads = []
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(f"Corrupted leads file, starting fresh: {e}")
                    existing_leads = []

            existing_leads.append(lead_data)

            # Write with proper error handling
            with open(leads_file, 'w', encoding='utf-8') as f:
                json.dump(existing_leads, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ Lead saved: {lead_data['name']}")

        except Exception as e:
            logger.error(f"❌ Error saving lead: {e}")
            # Don't fail the function call, just log the error


def create_function_schemas():
    """Create Pipecat function schemas for your business logic"""

    # Lead capture function
    capture_lead_function = FunctionSchema(
        name="capture_lead_info",
        description="Capture customer lead information when they express interest in applying for a loan",
        properties={
            "name": {
                "type": "string",
                "description": "Customer's full name"
            },
            "phone": {
                "type": "string",
                "description": "Customer's phone number"
            },
            "loan_type": {
                "type": "string",
                "description": "Type of loan they're interested in"
            },
            "email": {
                "type": "string",
                "description": "Customer's email address (optional)"
            },
            "loan_amount": {
                "type": "number",
                "description": "Desired loan amount (optional)"
            },
            "language": {
                "type": "string",
                "description": "Preferred language (Hindi/English)"
            }
        },
        required=["name", "phone", "loan_type"]
    )

    # Account lookup function
    lookup_account_function = FunctionSchema(
        name="lookup_account_info",
        description="Look up detailed account information using account ID. Use when customer asks about balance, payments, account status, or loan details.",
        properties={
            "account_id": {
                "type": "string",
                "description": "Account ID provided by customer"
            },
            "info_type": {
                "type": "string",
                "description": "Type of info requested: balance, payment, status, details",
                "enum": ["balance", "payment", "status", "details"]
            }
        },
        required=["account_id"]
    )

    # Create tools schema
    tools = ToolsSchema(standard_tools=[capture_lead_function, lookup_account_function])

    return tools


async def main():
    """Main function to set up and run the real-time voice agent"""

    # Environment check
    required_env_vars = ["OPENAI_API_KEY"]
    for var in required_env_vars:
        if not os.getenv(var):
            logger.error(f"Missing required environment variable: {var}")
            return

    try:
        # Create business logic handler - NO FrameProcessor inheritance!
        business_logic = LoanBusinessLogic()

        # Create function schemas
        tools = create_function_schemas()

        # Set up system prompt with language detection instructions
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
- When someone shows interest, use capture_lead_info function
- For account queries, use lookup_account_info function
- Be respectful and use "जी", "आप" in Hindi
- Keep responses extremely brief for real-time voice delivery

When users express interest in loans, immediately call capture_lead_info.
When users ask about account information, call lookup_account_info.
Always detect the user's language and respond in the same language.
"""

        # LLM Context with functions
        context = OpenAILLMContext(
            messages=[{"role": "system", "content": system_prompt}],
            tools=tools
        )

        # LLM Service with function calling support
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
            context=context
        )

        # CRITICAL: Create context aggregator for proper conversation flow
        context_aggregator = llm.create_context_aggregator(context)

        # NEW: Register function handlers using the modern approach
        async def handle_capture_lead(params: FunctionCallParams):
            """Modern function call handler for lead capture"""
            try:
                result = await business_logic.capture_lead_info(**params.arguments)
                await params.result_callback(result)
            except Exception as e:
                logger.error(f"Error in capture_lead_info: {e}")
                await params.result_callback("Sorry, there was an error capturing your information.")

        async def handle_lookup_account(params: FunctionCallParams):
            """Modern function call handler for account lookup"""
            try:
                result = await business_logic.lookup_account_info(**params.arguments)
                await params.result_callback(result)
            except Exception as e:
                logger.error(f"Error in lookup_account_info: {e}")
                await params.result_callback("Sorry, there was an error looking up your account.")

        # Register functions with new method
        llm.register_function("capture_lead_info", handle_capture_lead)
        llm.register_function("lookup_account_info", handle_lookup_account)

        logger.info("✅ OpenAI LLM configured with context aggregator and functions")

        # STT Service
        stt = None
        if os.getenv("DEEPGRAM_API_KEY"):
            stt = DeepgramSTTService(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                model="nova-2",
                language="hi-en",
                smart_format=True,
                interim_results=True
            )
            logger.info("✅ Deepgram STT configured")
        else:
            logger.error("❌ Deepgram API key required")
            return

        # TTS Service
        tts = None
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            tts = GoogleTTSService(
                credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
                voice_name="hi-IN-Chirp3-HD-Sadachbia",
                language_code="hi-IN",
                speaking_rate=1.0,
                pitch=0.0
            )
            logger.info("✅ Google Chirp3-HD TTS configured")
        else:
            logger.error("❌ Google Cloud credentials required")
            return

        # VAD Configuration
        vad_analyzer = SileroVADAnalyzer(
            sample_rate=16000,
            params=VADParams(
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=100
            )
        )

        # Transport
        transport = DailyTransport(
            room_url=os.getenv("DAILY_ROOM_URL") or "https://yourdomain.daily.co/simple-voice",
            token=os.getenv("DAILY_TOKEN"),
            bot_name="Hindi Voice Loan Assistant",
            params=DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                transcription_enabled=False,
                vad_enabled=True,
                vad_analyzer=vad_analyzer
            )
        )

        # SIMPLE PIPELINE - No custom FrameProcessor needed!
        pipeline = Pipeline([
            transport.input(),  # Audio input from user
            stt,  # Speech to text (TranscriptionFrame)
            context_aggregator.user(),  # Add user message to context
            llm,  # LLM response with function calling
            tts,  # Text to speech
            context_aggregator.assistant(),  # Add assistant response to context
            transport.output()  # Audio output to user
        ])

        # Pipeline task
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
        logger.info("📞 Features: Hindi/English, Function Calling, Lead Capture")
        logger.info("⚡ Using Pipecat's built-in function calling system")
        logger.info("🚀 NO custom FrameProcessor - clean and simple!")

        await runner.run(task)

    except Exception as e:
        logger.error(f"Error starting voice agent: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())