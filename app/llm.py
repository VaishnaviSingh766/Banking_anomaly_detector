import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError(
        "HF_TOKEN is not set. "
        "Please add it to your .env file."
    )


client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=hf_token
)


MODEL = "openai/gpt-oss-120b"


def generate_explanation(
    transaction,
    anomalies,
    rag_context
):

    anomaly_names = [
        anomaly["type"]
        for anomaly in anomalies
    ]

    retrieved_rules = "\n\n".join(
        rag_context
    )

    prompt = f"""
You are a banking transaction anomaly explanation assistant.

Explain why the transaction was flagged.

Do NOT claim that the transaction is confirmed fraud.

TRANSACTION:
Transaction ID: {transaction["transaction_id"]}
Amount: ₹{transaction["amount"]}
Customer average: ₹{transaction["customer_average"]}
Location: {transaction["location"]}
Usual location: {transaction["usual_location"]}
Time: {transaction["time"]}
Transactions in last hour:
{transaction.get("transactions_last_hour", 0)}

DETECTED ANOMALIES:
{", ".join(anomaly_names)}

RETRIEVED RULES:
{retrieved_rules}

Explain:
1. What was unusual.
2. Which indicators were triggered.
3. How the retrieved rules relate to the transaction.
4. What review action may be appropriate.

Do not invent information.
Do not claim confirmed fraud.
Use "flagged for review" where appropriate.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful banking anomaly "
                    "explanation assistant."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


def chat_with_transaction(
    transaction,
    anomalies,
    rag_context,
    user_question,
    chat_history=None
):
    """
    Answer a user's natural-language question
    about the selected transaction.
    """

    anomaly_names = [
        anomaly["type"]
        for anomaly in anomalies
    ]

    retrieved_rules = "\n\n".join(
        rag_context
    )

    history_text = ""

    if chat_history:

        for message in chat_history[-6:]:

            role = message["role"]
            content = message["content"]

            history_text += (
                f"{role.upper()}: {content}\n"
            )

    prompt = f"""
You are a banking transaction anomaly explanation assistant.

You are having a conversation with a banking analyst.

Answer the user's question using ONLY:
1. The transaction information
2. The detected anomaly information
3. The retrieved knowledge-base rules

Do not invent information.

IMPORTANT:
- A transaction being flagged does NOT mean confirmed fraud.
- Never say that the transaction is definitely fraudulent.
- Say "flagged for review" when appropriate.
- Keep the answer clear and concise.
- If the user asks about a rule, mention its rule ID.
- If the user asks what action should be taken, use the
  recommended review actions from the retrieved rules.
- If the answer cannot be determined from the provided
  information, say so.

TRANSACTION:
Transaction ID: {transaction["transaction_id"]}
Amount: ₹{transaction["amount"]}
Customer average transaction amount: ₹{transaction["customer_average"]}
Location: {transaction["location"]}
Usual location: {transaction["usual_location"]}
Time: {transaction["time"]}
Transactions in last hour:
{transaction.get("transactions_last_hour", 0)}

DETECTED ANOMALIES:
{", ".join(anomaly_names) if anomaly_names else "None"}

RETRIEVED KNOWLEDGE-BASE RULES:
{retrieved_rules}

PREVIOUS CONVERSATION:
{history_text}

USER QUESTION:
{user_question}

Answer the user's question directly.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a trustworthy banking anomaly "
                    "explanation assistant."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content