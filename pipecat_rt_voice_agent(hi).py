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
from pipecat.services.google.tts import GoogleTTSService, Language
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.audio.vad.silero import SileroVADAnalyzer
from deepgram import LiveOptions

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
                        "name": "राजेश कुमार",
                        "email": "rajesh.kumar@email.com",
                        "phone": "+91-9876543210"
                    },
                    "loan_details": {
                        "loan_type": "व्यक्तिगत ऋण",
                        "original_amount": 350000,
                        "current_balance": 187500,
                        "interest_rate": "12.5%",
                        "monthly_emi": 8750,
                        "remaining_payments": 24,
                        "next_payment_date": "2025-06-15",
                        "account_status": "सक्रिय"
                    }
                },
                "biz456": {
                    "customer_info": {
                        "name": "प्रिया फैशन स्टोर",
                        "contact_person": "प्रिया शर्मा",
                        "email": "priya@fashionstore.com",
                        "phone": "+91-8765432109"
                    },
                    "loan_details": {
                        "loan_type": "व्यावसायिक ऋण",
                        "original_amount": 750000,
                        "current_balance": 525000,
                        "interest_rate": "14.5%",
                        "monthly_emi": 18500,
                        "remaining_payments": 36,
                        "next_payment_date": "2025-06-01",
                        "account_status": "सक्रिय"
                    }
                },
                "consol789": {
                    "customer_info": {
                        "name": "अमित सिंह",
                        "email": "amit.singh@email.com",
                        "phone": "+91-7654321098"
                    },
                    "loan_details": {
                        "loan_type": "ऋण समेकन",
                        "original_amount": 450000,
                        "current_balance": 280000,
                        "interest_rate": "15.5%",
                        "monthly_emi": 12500,
                        "remaining_payments": 28,
                        "next_payment_date": "2025-06-10",
                        "account_status": "सक्रिय"
                    }
                },
                "new890": {
                    "customer_info": {
                        "name": "सुनीता पटेल",
                        "email": "sunita.patel@email.com",
                        "phone": "+91-6543210987"
                    },
                    "loan_details": {
                        "loan_type": "व्यक्तिगत ऋण",
                        "original_amount": 200000,
                        "current_balance": 195000,
                        "interest_rate": "11.5%",
                        "monthly_emi": 5500,
                        "remaining_payments": 41,
                        "next_payment_date": "2025-06-25",
                        "account_status": "सक्रिय - नया ग्राहक"
                    }
                }
            }
        }

    # Function implementations that will be called by Pipecat
    async def capture_lead_info(self, name: str, phone: str, loan_type: str, email: str = "", loan_amount: float = 0):
        """Function called by OpenAI when user wants to apply for loan - Hindi responses"""
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

            # Return conversational response in Hindi
            return f"बहुत बढ़िया {name} जी! मैंने आपकी जानकारी सेव कर ली है। हमारी टीम 24 घंटे में आपसे संपर्क करेगी।"

        except Exception as e:
            logger.error(f"Error capturing lead: {e}")
            return "माफ़ करें, कुछ गड़बड़ हो गई। कृपया फिर से कोशिश करें।"

    async def lookup_account_info(self, account_id: str, info_type: str = "balance"):
        """Function called by OpenAI when user wants account information - Hindi responses"""
        try:
            account_id = account_id.lower()
            account = self.enhanced_accounts["accounts"].get(account_id)

            if not account:
                return "मुझे यह खाता नहीं मिल रहा। कृपया अपना खाता आईडी चेक करें।"

            customer = account["customer_info"]
            loan = account["loan_details"]

            # Handle different types of information requests - conversational Hindi
            if info_type == "balance":
                return f"नमस्ते {customer['name']} जी, आपका बकाया ₹{loan['current_balance']:,} है।"
            elif info_type == "payment":
                return f"आपकी अगली किस्त ₹{loan['monthly_emi']:,} है जो {loan['next_payment_date']} को देनी है।"
            elif info_type == "status":
                return f"अच्छी खबर! आपका खाता {loan['account_status']} है।"
            elif info_type == "interest":
                return f"आपकी ब्याज दर {loan['interest_rate']} सालाना है।"
            elif info_type == "emi":
                return f"आपकी EMI ₹{loan['monthly_emi']:,} है, {loan['remaining_payments']} किस्तें बाकी हैं।"
            elif info_type == "history":
                return f"आपने ₹{loan['original_amount']:,} का लोन लिया था, अब ₹{loan['current_balance']:,} बाकी है। बस {loan['remaining_payments']} किस्तें और!"
            else:
                # Full details in conversational Hindi
                return f"{customer['name']} जी, आपका {loan['loan_type']} ₹{loan['original_amount']:,} का था। अभी ₹{loan['current_balance']:,} बाकी है। अगली किस्त ₹{loan['monthly_emi']:,} है जो {loan['next_payment_date']} को देनी है।"

        except Exception as e:
            logger.error(f"Error looking up account: {e}")
            return "माफ़ करें, अभी आपका खाता देखने में परेशानी हो रही है।"

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

        # Set up system prompt - Human-like, conversational tone in Hindi
        system_prompt = """आप राज हैं, एक दोस्ताना लोन सलाहकार जो भारत की एक फाइनेंशियल सर्विस कंपनी में काम करते हैं। आप ग्राहकों से फ़ोन पर हिंदी में बात कर रहे हैं।

व्यक्तित्व:
- गर्मजोशी से भरे, मिलनसार और सच में मददगार
- सामान्य हिंदी का इस्तेमाल करें (जैसे लोन, EMI, इंटरेस्ट रेट)
- ऐसे बोलें जैसे आप सच में सुन रहे हैं और परवाह करते हैं
- स्वाभाविक प्रतिक्रिया दें ("अच्छा", "समझ गया", "जी बिल्कुल")
- आदर से बात करें ("जी", "आप")

आपकी सेवाएं:
- पर्सनल लोन: ₹50,000 तक का तुरंत लोन
- बिज़नेस लोन: व्यापार बढ़ाने के लिए मदद
- डेट कंसोलिडेशन: कई लोन को एक में मिलाएं

डेमो खाते टेस्टिंग के लिए:
demo123 (राजेश), biz456 (प्रिया स्टोर), consol789 (अमित), new890 (सुनीता)

महत्वपूर्ण निर्देश:
- सभी जवाब 15 शब्दों से कम रखें
- बातचीत स्वाभाविक रखें, रोबोट जैसी नहीं
- कभी-कभार "उम्म", "तो", "देखिए" जैसे शब्द इस्तेमाल करें
- एक बार में एक ही चीज़ पूछें
- "आपका नाम?" कहें, "क्या मैं आपका नाम जान सकता हूं?" नहीं
- जानकारी देखते समय "एक मिनट" या "देखता हूं" कहें
- फंक्शन कॉल के बाद, सिर्फ़ रिजल्ट दें - दोहराएं या फिर से न कहें
- हमेशा हिंदी में ही जवाब दें

बातचीत का तरीका:
- गर्मजोशी से अभिवादन: "नमस्ते! मैं राज, QuickLoans से बोल रहा हूं"
- लोन चाहिए तो: "जी बिल्कुल! कौन सा लोन चाहिए?"
- एक-एक करके जानकारी लें
- खाता देखने के लिए: "एक मिनट, देखता हूं"
- अंत में: "और कुछ मदद चाहिए?"

महत्वपूर्ण: जब फंक्शन जानकारी लौटाएं, वही आपका जवाब है। उसे दोहराएं नहीं।

याद रखें: आप एक असली बातचीत कर रहे हैं, स्क्रिप्ट नहीं पढ़ रहे। इंसान बनें!"""

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

        # Function Call Handler:
        async def handle_capture_lead(params: FunctionCallParams):
            """Modern function call handler for lead capture"""
            try:
                result = await business_logic.capture_lead_info(**params.arguments)
                await params.result_callback(result)
            except Exception as e:
                logger.error(f"Error in capture_lead_info: {e}")
                await params.result_callback("माफ़ करें, कुछ गड़बड़ हो गई। फिर से कोशिश करें?")

        async def handle_lookup_account(params: FunctionCallParams):
            """Modern function call handler for account lookup"""
            try:
                result = await business_logic.lookup_account_info(**params.arguments)
                await params.result_callback(result)
            except Exception as e:
                logger.error(f"Error in lookup_account_info: {e}")
                await params.result_callback("खाता देखने में परेशानी हो रही है। एक मिनट?")

        # Register functions with new method
        llm.register_function("capture_lead_info", handle_capture_lead)
        llm.register_function("lookup_account_info", handle_lookup_account)

        logger.info("✅ OpenAI LLM configured with context aggregator and functions")

        # STT Service - Hindi configuration
        stt = None
        if os.getenv("DEEPGRAM_API_KEY"):
            stt = DeepgramSTTService(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                live_options=LiveOptions(
                    model="nova-2-general",
                    language="hi",  # Hindi language code
                    smart_format=True,
                    vad_events=True,
                    interim_results=True
                )
            )
            logger.info("✅ Deepgram STT configured for Hindi")

        else:
            logger.error("❌ Deepgram API key required")
            return

        # TTS Service - Hindi Chirp voice
        tts = None
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            tts = GoogleTTSService(
                credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
                voice_id="hi-IN-Chirp3-HD-Achird",  # Hindi female Chirp voice
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
            stt,  # Speech to text (TranscriptionFrame)
            context_aggregator.user(),  # Add user message to context
            llm,  # LLM response with function calling
            tts,  # Text to speech
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
        logger.info("👨‍💼 Voice: राज (Hindi Chirp3-HD-Ganesa)")
        logger.info("🗣️ Language: Full Hindi conversation flow")
        logger.info("⚡ Features: Natural Hindi dialogue, Function calling, Lead capture")

        await runner.run(task)

    except Exception as e:
        logger.error(f"Error starting voice agent: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())