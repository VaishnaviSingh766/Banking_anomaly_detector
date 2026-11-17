import json
from pathlib import Path
import streamlit as st

from app.anomaly_detector import detect_anomalies
from app.rag import RAGRetriever
from app.llm import (
    generate_explanation,
    chat_with_transaction
)


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Banking Anomaly Assistant",
    page_icon="🏦",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
TRANSACTIONS_PATH = BASE_DIR / "data" / "transactions.json"


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #777;
        font-size: 16px;
        margin-bottom: 20px;
    }

    .transaction-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        background-color: #fafafa;
        margin-bottom: 20px;
    }

    .rule-card {
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-bottom: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# LOAD TRANSACTIONS
# ==========================================================

def load_transactions():

    with open(
        TRANSACTIONS_PATH,
        "r"
    ) as file:

        return json.load(file)


transactions = load_transactions()


# ==========================================================
# SESSION STATE
# ==========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "selected_transaction" not in st.session_state:

    st.session_state.selected_transaction = None


if "transactions" not in st.session_state:

    st.session_state.transactions = transactions


if "anomalies" not in st.session_state:

    st.session_state.anomalies = []


if "rag_context" not in st.session_state:

    st.session_state.rag_context = []


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">🏦 Banking Anomaly Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered conversational transaction explanation system'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("💳 Transaction input")

    st.caption(
        "Choose a sample or enter any transaction dynamically. "
        "Use synthetic or anonymized data only."
    )

    input_mode = st.radio(
        "Input mode",
        ["Sample transaction", "Enter transaction", "Paste JSON"],
        label_visibility="collapsed"
    )

    selected_transaction = None
    analyze_transaction = False

    if input_mode == "Sample transaction":

        transaction_ids = [
            transaction["transaction_id"]
            for transaction in st.session_state.transactions
        ]

        selected_id = st.selectbox(
            "Select a transaction",
            transaction_ids
        )

        selected_transaction = next(
            transaction
            for transaction in st.session_state.transactions
            if transaction["transaction_id"] == selected_id
        )
        analyze_transaction = st.button(
            "🔄 Analyze transaction",
            use_container_width=True
        )

    elif input_mode == "Enter transaction":

        with st.form("transaction_form"):

            transaction_id = st.text_input(
                "Transaction ID",
                value="TXN_CUSTOM"
            )
            amount = st.number_input(
                "Amount (₹)",
                min_value=0.0,
                value=10000.0,
                step=500.0
            )
            customer_average = st.number_input(
                "Customer average amount (₹)",
                min_value=0.0,
                value=12000.0,
                step=500.0
            )
            location = st.text_input(
                "Current location",
                value="Mumbai"
            )
            usual_location = st.text_input(
                "Usual location",
                value="Mumbai"
            )
            transaction_time = st.text_input(
                "Transaction time (HH:MM)",
                value="14:30"
            )
            transactions_last_hour = st.number_input(
                "Transactions in last hour",
                min_value=0,
                value=1,
                step=1
            )
            enter_transaction = st.form_submit_button(
                "Use this transaction",
                use_container_width=True
            )

        if enter_transaction:
            selected_transaction = {
                "transaction_id": transaction_id.strip(),
                "amount": amount,
                "customer_average": customer_average,
                "location": location.strip(),
                "usual_location": usual_location.strip(),
                "time": transaction_time.strip(),
                "transactions_last_hour": transactions_last_hour
            }
            analyze_transaction = True

    else:

        transaction_json = st.text_area(
            "Transaction JSON",
            value=json.dumps(
                st.session_state.transactions[0],
                indent=2
            ),
            height=220
        )

        if st.button(
            "Use JSON transaction",
            use_container_width=True
        ):
            try:
                selected_transaction = json.loads(transaction_json)
                analyze_transaction = True
            except json.JSONDecodeError as error:
                st.error(f"Invalid JSON: {error.msg}.")

    if selected_transaction is not None and analyze_transaction:
        required_fields = [
            "transaction_id",
            "amount",
            "customer_average",
            "location",
            "usual_location",
            "time"
        ]
        missing_fields = [
            field
            for field in required_fields
            if field not in selected_transaction
        ]

        if missing_fields:
            st.error(
                "Missing required fields: "
                + ", ".join(missing_fields)
            )
        else:
            try:
                selected_transaction["amount"] = float(
                    selected_transaction["amount"]
                )
                selected_transaction["customer_average"] = float(
                    selected_transaction["customer_average"]
                )
                selected_transaction["transactions_last_hour"] = int(
                    selected_transaction.get("transactions_last_hour", 0)
                )
                anomalies = detect_anomalies(selected_transaction)
                rag = RAGRetriever()
                anomaly_query = " ".join(
                    anomaly["type"] for anomaly in anomalies
                ) or "normal transaction"
                rag_context = rag.retrieve(anomaly_query, top_k=3)

                st.session_state.selected_transaction = selected_transaction
                st.session_state.anomalies = anomalies
                st.session_state.rag_context = rag_context
                st.session_state.messages = []

                if selected_transaction not in st.session_state.transactions:
                    st.session_state.transactions.append(selected_transaction)

                try:
                    explanation = generate_explanation(
                        selected_transaction,
                        anomalies,
                        rag_context
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": explanation}
                    )
                except Exception as error:
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                "I couldn't generate the initial "
                                "explanation.\n\n"
                                f"Error: {error}"
                            )
                        }
                    )
                st.rerun()
            except (TypeError, ValueError) as error:
                st.error(f"Invalid transaction values: {error}")

# ==========================================================
# CHECK TRANSACTION LOADED
# ==========================================================

transaction = (
    st.session_state.selected_transaction
)


if transaction is None:

    st.info(
        "👈 Select a transaction from the sidebar "
        "and click **Load Transaction** to begin."
    )

    st.markdown(
        """
        ### 💬 You will be able to ask questions like:

        - **Why was this transaction flagged?**
        - **Which rule was triggered?**
        - **Why is the amount unusual?**
        - **Why is the location suspicious?**
        - **Why was the transaction made at this time?**
        - **What should I do next?**
        - **Is this definitely fraud?**
        """
    )

    st.stop()


# ==========================================================
# TRANSACTION CARD
# ==========================================================

st.subheader("Transaction Under Review")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Transaction",
        transaction["transaction_id"]
    )


with col2:

    st.metric(
        "Amount",
        f"₹{transaction['amount']:,}"
    )


with col3:

    st.metric(
        "Location",
        transaction["location"]
    )


with col4:

    st.metric(
        "Time",
        transaction["time"]
    )


# ==========================================================
# ANOMALIES
# ==========================================================

anomalies = st.session_state.anomalies


if anomalies:

    st.markdown("### ⚠️ Detected Indicators")

    anomaly_text = "  •  ".join(
        anomaly["type"]
        for anomaly in anomalies
    )

    st.warning(
        anomaly_text
    )

else:

    st.success(
        "No anomaly indicators were detected."
    )


st.divider()


# ==========================================================
# CHAT HEADER
# ==========================================================

st.subheader("💬 Ask About This Transaction")


# ==========================================================
# CHAT HISTORY
# ==========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ==========================================================
# SUGGESTED QUESTIONS
# ==========================================================

st.markdown("#### Suggested questions")

col1, col2, col3 = st.columns(3)


suggested_question = None


with col1:

    if st.button(
        "Why was it flagged?",
        use_container_width=True
    ):

        suggested_question = (
            "Why was this transaction flagged?"
        )


with col2:

    if st.button(
        "Which rules triggered?",
        use_container_width=True
    ):

        suggested_question = (
            "Which rules were triggered for this transaction?"
        )


with col3:

    if st.button(
        "What should I do?",
        use_container_width=True
    ):

        suggested_question = (
            "What review action should be taken?"
        )


# ==========================================================
# CHAT INPUT
# ==========================================================

user_question = st.chat_input(
    "Ask a question about this transaction..."
)


if suggested_question:

    user_question = suggested_question


# ==========================================================
# PROCESS QUESTION
# ==========================================================

if user_question:

    # ----------------------------------------------
    # Show user message
    # ----------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )


    with st.chat_message("user"):

        st.markdown(
            user_question
        )


    # ----------------------------------------------
    # Get answer
    # ----------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing the transaction..."
        ):

            try:

                answer = chat_with_transaction(
                    transaction,
                    anomalies,
                    st.session_state.rag_context,
                    user_question,
                    st.session_state.messages
                )

                st.markdown(
                    answer
                )


                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                error_message = (
                    "I couldn't generate an answer.\n\n"
                    f"Error: {str(e)}"
                )

                st.error(
                    error_message
                )


# ==========================================================
# KNOWLEDGE SOURCES
# ==========================================================

st.divider()

with st.expander(
    "📚 View RAG Knowledge Used"
):

    st.caption(
        "These are the knowledge-base sections "
        "retrieved for this transaction."
    )


    for i, context in enumerate(
        st.session_state.rag_context,
        start=1
    ):

        st.markdown(
            f"**Retrieved Source {i}**"
        )

        st.write(
            context
        )

        st.divider()


# ==========================================================
# FOOTER
# ==========================================================

st.caption(
    "Prototype / Demo — anomaly indicators do not "
    "by themselves confirm fraud."
)