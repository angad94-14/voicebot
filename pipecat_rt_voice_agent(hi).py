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
from deepgram import LiveOptions
from pipecat.services.google.tts import GoogleTTSService, Language
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
    Returns responses in English - LLM will translate to Hindi.
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

    # Function implementations - all return English, LLM translates to Hindi
    async def capture_lead_info(self, name: str, phone: str, loan_type: str, email: str = "", loan_amount: float = 0):
        """Function called by OpenAI when user wants to apply for loan - Returns English"""
        try:
            lead_data = {
                "name": name,
                "phone": phone,
                "loan_type": loan_type,
                "email": email,
                "loan_amount": loan_amount,
                "language": "Hindi",
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

            # Return info in English - LLM will translate to Hindi
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

        # Set up system prompt - Instructs to ALWAYS respond in Hindi
        system_prompt = """You are राज (Raj), a friendly loan advisor at a financial services company in India. You're having a phone conversation with customers.

PERSONALITY:
- Warm, approachable, and genuinely helpful
- Use conversational Hindi with common English terms (लोन, EMI, इंटरेस्ट रेट)
- Sound natural and caring
- Use respectful language ("जी", "आप")

SERVICES:
- Personal Loans: Quick loans up to ₹50,000
- Business Loans: Help businesses grow
- Debt Consolidation: Combine multiple loans into one

DEMO ACCOUNTS:
demo123 (Rajesh), biz456 (Priya's Store), consol789 (Amit), new890 (Sunita)

CRITICAL INSTRUCTIONS:
- ALWAYS respond in Hindi, regardless of input language
- Keep ALL responses under 15 words
- Sound conversational, not robotic
- Use natural fillers ("तो", "देखिए", "अच्छा")
- Ask one thing at a time
- When looking up info say "एक मिनट" or "देखता हूं"
- After function calls, just deliver the result - don't repeat
- Even if user speaks English, ALWAYS respond in Hindi

CONVERSATION FLOW:
- Greeting: "नमस्ते! मैं राज, QuickLoans से बोल रहा हूं"
- For loans: "जी बिल्कुल! कौन सा लोन चाहिए?"
- Get details one by one
- For account lookup: "एक मिनट, देखता हूं"
- End with: "और कुछ मदद चाहिए?"

IMPORTANT: You will receive function results in English, but you MUST translate and respond in Hindi.

Remember: Natural conversation in Hindi only!"""

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

        # STT Service - Multi-language support for Hindi/English
        stt = None
        if os.getenv("DEEPGRAM_API_KEY"):
            stt = DeepgramSTTService(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                live_options=LiveOptions(
                    model="nova-2-general",
                    language="hi",  # Multi-language for Hindi/English mix
                    smart_format=True,
                    vad_events=True,
                    interim_results=True
                )
            )
            logger.info("✅ Deepgram STT configured for multi-language (Hindi/English)")
        else:
            logger.error("❌ Deepgram API key required")
            return

        # TTS Service - Hindi Chirp voice
        tts = None
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            tts = GoogleTTSService(
                credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
                voice_id="hi-IN-Chirp3-HD-Achird",  # Hindi male Chirp voice
                params=GoogleTTSService.InputParams(
                    language=Language.HI  # Hindi language
                )
            )
            logger.info("✅ Google Hindi Chirp TTS configured")
        else:
            logger.error("❌ Google Cloud credentials required")
            return

        # VAD Configuration
        vad_analyzer = SileroVADAnalyzer(
            sample_rate=16000,
            params=VADParams()
        )

        # Transport
        transport = DailyTransport(
            room_url=os.getenv("DAILY_ROOM_URL") or "https://yourdomain.daily.co/simple-voice",
            token=os.getenv("DAILY_TOKEN"),
            bot_name="राज - लोन सलाहकार",
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
            stt,  # Speech to text (multi-language)
            context_aggregator.user(),  # Add user message to context
            llm,  # LLM response with function calling
            tts,  # Text to speech (Hindi)
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

        logger.info("🎙️ Real-Time Hindi Voice Loan Assistant Ready!")
        logger.info("🌐 STT: Multi-language (Hindi/English input)")

        await runner.run(task)

    except Exception as e:
        logger.error(f"Error starting voice agent: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())