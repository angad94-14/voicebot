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
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GoogleTTSService, Language
from pipecat.services.openai.stt import OpenAISTTService
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

# Configuration option - set to "openai" or "google"
STT_PROVIDER = os.getenv("STT_PROVIDER", "openai")  # Default to OpenAI


class LoanBusinessLogic:
    """
    Simple class to handle business logic - NO FrameProcessor inheritance!
    Returns responses in English - LLM will translate to Marathi.
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
        """Load account data - keeping names in English for simplicity"""
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

    # Function implementations - all return English, LLM translates to Marathi
    async def capture_lead_info(self, name: str, phone: str, loan_type: str, email: str = "", loan_amount: float = 0):
        """Function called by OpenAI when user wants to apply for loan - Returns English"""
        try:
            lead_data = {
                "name": name,
                "phone": phone,
                "loan_type": loan_type,
                "email": email,
                "loan_amount": loan_amount,
                "language": "Marathi",
                "timestamp": datetime.now().isoformat(),
                "status": "new",
                "session_id": self.session_id
            }

            self.leads.append(lead_data)
            self.current_lead = lead_data

            # Save lead to file
            self._save_lead_to_file(lead_data)

            # Return response in English - LLM will translate
            return f"Perfect {name}! I've saved your details. Our team will contact you within 24 hours for your {loan_type}."

        except Exception as e:
            logger.error(f"Error capturing lead: {e}")
            return "Sorry, something went wrong. Please try again."

    async def lookup_account_info(self, account_id: str, info_type: str = "balance"):
        """Function called by OpenAI when user wants account information - Returns English"""
        try:
            account_id = account_id.lower()
            account = self.enhanced_accounts["accounts"].get(account_id)

            if not account:
                return "I cannot find that account. Please check your account ID."

            customer = account["customer_info"]
            loan = account["loan_details"]

            # Return info in English - LLM will translate to Marathi
            if info_type == "balance":
                return f"Hello {customer['name'].split()[0]}, your current balance is ₹{loan['current_balance']:,}."
            elif info_type == "payment":
                return f"Your next payment of ₹{loan['monthly_emi']:,} is due on {loan['next_payment_date']}."
            elif info_type == "status":
                return f"Good news! Your account is {loan['account_status']}."
            elif info_type == "interest":
                return f"Your interest rate is {loan['interest_rate']} per annum."
            elif info_type == "emi":
                return f"Your EMI is ₹{loan['monthly_emi']:,} per month, with {loan['remaining_payments']} payments remaining."
            elif info_type == "history":
                return f"You started with ₹{loan['original_amount']:,}, now at ₹{loan['current_balance']:,}. Just {loan['remaining_payments']} payments to go!"
            else:
                # Full details in English
                return f"Hi {customer['name'].split()[0]}, your {loan['loan_type']} started at ₹{loan['original_amount']:,}. Current balance is ₹{loan['current_balance']:,} with {loan['remaining_payments']} payments left. Next payment of ₹{loan['monthly_emi']:,} is on {loan['next_payment_date']}."

        except Exception as e:
            logger.error(f"Error looking up account: {e}")
            return "Sorry, I'm having trouble accessing your account right now."

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
            }
        },
        required=["name", "phone", "loan_type"]
    )

    # Account lookup function
    lookup_account_function = FunctionSchema(
        name="lookup_account_info",
        description="Look up detailed account information using account ID. Use when customer asks about balance, payments, account status, loan details, interest rate, EMI, or any account information.",
        properties={
            "account_id": {
                "type": "string",
                "description": "Account ID provided by customer"
            },
            "info_type": {
                "type": "string",
                "description": "Type of info requested: balance, payment, status, details, interest, emi, history",
                "enum": ["balance", "payment", "status", "details", "interest", "emi", "history"]
            }
        },
        required=["account_id"]
    )

    # Create tools schema
    tools = ToolsSchema(standard_tools=[capture_lead_function, lookup_account_function])

    return tools


def create_stt_service(stt_provider: str):
    """Create STT service based on provider choice"""

    if stt_provider.lower() == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key required for OpenAI STT")

        stt = OpenAISTTService(
            model="whisper-1",  # Standard Whisper model
            api_key=os.getenv("OPENAI_API_KEY"),
            language=Language.MR,  # Set Marathi as primary language
            prompt="You are a multilingual voice agent for a financial services company in India. You will hear conversations in Marathi, English, or Hindi. Common loan-related terms: कर्ज (loan), EMI, हप्ता (installment), खाता (account), शिल्लक (balance). Transcribe accurately preserving the original language. Format numbers as digits. Account IDs may include: demo123, biz456, consol789, new890."
        )
        logger.info("✅ OpenAI STT (Whisper) configured for multilingual support")
        return stt

    elif stt_provider.lower() == "google":
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            raise ValueError("Google Cloud credentials required for Google STT")

        stt = GoogleSTTService(
            location="global",
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            sample_rate=16000,
            params=GoogleSTTService.InputParams(
                languages=[Language.MR, Language.EN_IN],  # Marathi and Indian English
                model="latest_long",
                enable_automatic_punctuation=False,
                enable_interim_results=True,
                enable_voice_activity_events=False,
                profanity_filter=False
            )
        )
        logger.info("✅ Google STT configured for Marathi and Indian English")
        return stt

    else:
        raise ValueError(f"Unsupported STT provider: {stt_provider}. Use 'openai' or 'google'")


async def main():
    """Main function to set up and run the real-time voice agent"""

    # Log STT provider choice
    logger.info(f"🎤 STT Provider selected: {STT_PROVIDER}")

    # Environment check based on STT provider
    if STT_PROVIDER.lower() == "openai":
        required_env_vars = ["OPENAI_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"]
    else:
        required_env_vars = ["GOOGLE_APPLICATION_CREDENTIALS"]

    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return

    try:
        # Create business logic handler - NO FrameProcessor inheritance!
        business_logic = LoanBusinessLogic()

        # Create function schemas
        tools = create_function_schemas()

        # Set up system prompt - Instructs to ALWAYS respond in Marathi
        system_prompt = """You are संदीप (Sandeep), a friendly loan advisor at a financial services company in India. You're having a phone conversation with customers.

PERSONALITY:
- Warm, approachable, and genuinely helpful
- Use conversational Marathi with common English terms (लोन, EMI, इंटरेस्ट रेट)
- Sound natural and caring
- Use respectful language ("साहेब", "ताई", "आपण")

SERVICES:
- Personal Loans (वैयक्तिक कर्ज): Quick loans up to ₹50,000
- Business Loans (व्यवसाय कर्ज): Help businesses grow
- Debt Consolidation (कर्ज एकत्रीकरण): Combine multiple loans into one

DEMO ACCOUNTS:
demo123 (राजेश), biz456 (प्रिया स्टोअर), consol789 (अमित), new890 (सुनीता)

CRITICAL INSTRUCTIONS:
- ALWAYS respond in Marathi (मराठी), regardless of input language
- You will receive transcriptions in the original language spoken (English, Hindi, or Marathi)
- Even if the customer speaks in English or Hindi, ALWAYS respond in Marathi
- Keep ALL responses under 15 words
- Sound conversational, not robotic
- Use natural fillers ("म्हणजे", "बघा", "हो")
- Ask one thing at a time
- When looking up info say "एक मिनिट" or "बघतो"
- After function calls, just deliver the result - don't repeat

CONVERSATION FLOW:
- Greeting: "नमस्कार! मी संदीप, QuickLoans मधून बोलतोय"
- For loans: "नक्की! कोणते कर्ज हवे आहे?"
- Get details one by one
- For account lookup: "एक मिनिट, बघतो"
- End with: "अजून काही मदत हवी का?"

IMPORTANT: You will receive function results in English, but you MUST translate and respond in Marathi.

Remember: Natural conversation in Marathi only!"""

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

        # Function Call Handlers - Simple English error messages
        async def handle_capture_lead(params: FunctionCallParams):
            """Modern function call handler for lead capture"""
            try:
                result = await business_logic.capture_lead_info(**params.arguments)
                await params.result_callback(result)
            except Exception as e:
                logger.error(f"Error in capture_lead_info: {e}")
                await params.result_callback("Sorry, something went wrong. Please try again.")

        async def handle_lookup_account(params: FunctionCallParams):
            """Modern function call handler for account lookup"""
            try:
                result = await business_logic.lookup_account_info(**params.arguments)
                await params.result_callback(result)
            except Exception as e:
                logger.error(f"Error in lookup_account_info: {e}")
                await params.result_callback("Sorry, having trouble with account lookup. One moment?")

        # Register functions with new method
        llm.register_function("capture_lead_info", handle_capture_lead)
        llm.register_function("lookup_account_info", handle_lookup_account)

        logger.info("✅ OpenAI LLM configured with context aggregator and functions")

        # STT Service - Create based on provider choice
        try:
            stt = create_stt_service(STT_PROVIDER)
        except Exception as e:
            logger.error(f"❌ Failed to create STT service: {e}")
            return

        # TTS Service - Marathi Chirp voice
        tts = None
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            tts = GoogleTTSService(
                credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
                voice_id="mr-IN-Chirp3-HD-Achird",  # Marathi Chirp voice
                params=GoogleTTSService.InputParams(
                    language=Language.MR  # Marathi language
                )
            )
            logger.info("✅ Google Marathi Chirp TTS configured")
        else:
            logger.error("❌ Google Cloud credentials required for TTS")
            return

        # VAD Configuration
        vad_analyzer = SileroVADAnalyzer(
            sample_rate=16000,
            params=VADParams(
                # Default parameters
            )
        )

        # Transport
        transport = DailyTransport(
            room_url=os.getenv("DAILY_ROOM_URL") or "https://yourdomain.daily.co/simple-voice",
            token=os.getenv("DAILY_TOKEN"),
            bot_name="संदीप - कर्ज सल्लागार",
            params=DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                transcription_enabled=False,
                vad_analyzer=vad_analyzer
            )
        )

        # SIMPLE PIPELINE - No custom FrameProcessor needed!
        pipeline = Pipeline([
            transport.input(),  # Audio input from user
            stt,  # Speech to text (OpenAI or Google based on config)
            context_aggregator.user(),  # Add user message to context
            llm,  # LLM response with function calling
            tts,  # Text to speech (Marathi)
            context_aggregator.assistant(),  # Add assistant response to context
            transport.output(),  # Audio output to user
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

        logger.info("🎙️ Real-Time Voice Loan Assistant Ready!")
        logger.info(
            f"🌐 STT: {STT_PROVIDER.upper()} ({'Whisper' if STT_PROVIDER.lower() == 'openai' else 'Cloud Speech-to-Text'})")

        await runner.run(task)

    except Exception as e:
        logger.error(f"Error starting voice agent: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())