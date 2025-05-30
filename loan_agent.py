"""
Loan Company Voice Agent - OpenAI Agents Framework
"""
import asyncio
import numpy as np
from typing import Dict, Any, Optional
from agents import Agent, function_tool
import json

# Common system prompt for voice output best practices:
voice_system_prompt = """
[Output Structure]
Your output will be delivered in an audio voice response, please ensure that every response meets these guidelines:
1. Use a friendly, human tone that will sound natural when spoken aloud.
2. Keep responses short and segmented—ideally one to two concise sentences per step.
3. Avoid technical jargon; use plain language so that instructions are easy to understand.
4. Provide only essential details so as not to overwhelm the listener.
"""

# Customer account data structure
CUSTOMER_DB = {
    "123456": {
        "user_id": "123456",
        "name": "John Smith",
        "loan_balance": 15750.00,
        "monthly_payment": 525.00,
        "next_payment_date": "2025-06-15",
        "account_status": "Current",
        "credit_score": 720,
        "available_credit": 5000.00
    },
    "789012": {
        "user_id": "789012",
        "name": "Sarah Johnson",
        "loan_balance": 23400.00,
        "monthly_payment": 780.00,
        "next_payment_date": "2025-06-10",
        "account_status": "Current",
        "credit_score": 685,
        "available_credit": 2500.00
    },
    "345678": {
        "user_id": "345678",
        "name": "Mike Chen",
        "loan_balance": 8900.00,
        "monthly_payment": 295.00,
        "next_payment_date": "2025-06-20",
        "account_status": "Past Due",
        "credit_score": 650,
        "available_credit": 0.00
    }
}


# --- Tool Functions ---

@function_tool
def get_account_info(user_id: str) -> dict:
    """Get comprehensive account information for a customer."""
    if user_id in CUSTOMER_DB:
        account = CUSTOMER_DB[user_id]
        return {
            "success": True,
            "customer_name": account["name"],
            "loan_balance": account["loan_balance"],
            "monthly_payment": account["monthly_payment"],
            "next_payment_date": account["next_payment_date"],
            "account_status": account["account_status"],
            "credit_score": account["credit_score"],
            "available_credit": account["available_credit"]
        }
    else:
        return {
            "success": False,
            "error": "Customer account not found. Please verify your account number."
        }


@function_tool
def process_payment(user_id: str, amount: float) -> dict:
    """Process a payment for a customer account."""
    if user_id not in CUSTOMER_DB:
        return {"success": False, "error": "Account not found"}

    account = CUSTOMER_DB[user_id]

    if amount <= 0:
        return {"success": False, "error": "Payment amount must be greater than zero"}

    if amount > account["loan_balance"]:
        return {
            "success": False,
            "error": f"Payment amount ${amount:.2f} exceeds loan balance ${account['loan_balance']:.2f}"
        }

    # Process payment
    new_balance = account["loan_balance"] - amount
    CUSTOMER_DB[user_id]["loan_balance"] = new_balance

    return {
        "success": True,
        "message": f"Payment of ${amount:.2f} processed successfully",
        "new_balance": new_balance,
        "customer_name": account["name"]
    }


@function_tool
def check_loan_eligibility(user_id: str, requested_amount: float) -> dict:
    """Check if a customer is eligible for additional credit."""
    if user_id not in CUSTOMER_DB:
        return {"success": False, "error": "Account not found"}

    account = CUSTOMER_DB[user_id]

    # Eligibility checks
    if account["account_status"] != "Current":
        return {
            "success": True,
            "eligible": False,
            "reason": "Account must be current to qualify for additional credit"
        }

    if account["credit_score"] < 650:
        return {
            "success": True,
            "eligible": False,
            "reason": "Credit score too low for additional credit"
        }

    if requested_amount > account["available_credit"]:
        return {
            "success": True,
            "eligible": False,
            "reason": f"Requested amount ${requested_amount:.2f} exceeds available credit ${account['available_credit']:.2f}"
        }

    return {
        "success": True,
        "eligible": True,
        "approved_amount": requested_amount,
        "message": f"You are pre-approved for ${requested_amount:.2f}"
    }


# --- Main Loan Service Agent ---

loan_service_agent = Agent(
    name="LoanServiceAgent",
    instructions=voice_system_prompt + """You are a helpful customer service agent for ABC Lending Company. 

You help customers with:
- Account balance and payment information
- Processing payments 
- Loan applications and eligibility checks
- General account inquiries

Always be professional, empathetic, and clear in your responses. When customers ask about their account, always ask for their account number if they haven't provided it.

For account information requests, use the get_account_info tool.
For payment processing, use the process_payment tool. 
For loan applications, use the check_loan_eligibility tool.

Be conversational and friendly. Address customers by name when you have their information.
""",
    tools=[get_account_info, process_payment, check_loan_eligibility],
)

# --- Triage Agent (optional, for routing different types of queries) ---

triage_agent = Agent(
    name="TriageAgent",
    instructions="""You are a customer service triage agent for ABC Lending Company.

Listen to the customer's request and provide helpful responses about:
- Account balances and payment information
- Payment processing
- Loan applications and eligibility
- General account help

You have access to customer account tools. Always ask for account numbers when needed for account-specific requests.

Be warm, professional, and helpful. Keep responses conversational and clear.
""",
    handoffs=[loan_service_agent],
)