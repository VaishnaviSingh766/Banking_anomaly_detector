import json

from app.anomaly_detector import detect_anomalies
from app.rag import RAGRetriever
from app.llm import generate_explanation


def load_transactions():
    with open("data/transactions.json", "r") as file:
        return json.load(file)


def main():

    print("=" * 60)
    print("BANKING TRANSACTION ANOMALY EXPLAINER")
    print("=" * 60)

    # --------------------------------
    # 1. Load transaction
    # --------------------------------

    transactions = load_transactions()

    transaction = transactions[0]

    print("\nTRANSACTION")
    print("-" * 60)

    print(f"Transaction ID: {transaction['transaction_id']}")
    print(f"Amount: ₹{transaction['amount']}")
    print(f"Location: {transaction['location']}")
    print(f"Time: {transaction['time']}")

    # --------------------------------
    # 2. Detect anomalies
    # --------------------------------

    anomalies = detect_anomalies(transaction)

    print("\nDETECTED ANOMALIES")
    print("-" * 60)

    if not anomalies:
        print("No anomalies detected.")

    else:
        for anomaly in anomalies:
            print(
                f"⚠ {anomaly['type']} "
                f"({anomaly['rule_id']})"
            )

    # --------------------------------
    # 3. RAG retrieval
    # --------------------------------

    rag = RAGRetriever()

    # Convert detected anomalies into a search query
    anomaly_query = " ".join(
        anomaly["type"]
        for anomaly in anomalies
    )

    print("\nRAG QUERY")
    print("-" * 60)

    print(anomaly_query)

    # Search the PDF knowledge stored in ChromaDB
    rag_context = rag.retrieve(
        anomaly_query,
        top_k=3
    )

    # --------------------------------
    # 4. Display RAG output
    # --------------------------------

    print("\nRAG OUTPUT")
    print("-" * 60)

    for context in rag_context:
        print(context)
        print("-" * 40)

    # --------------------------------
    # 5. Generate explanation
    # --------------------------------

    explanation = generate_explanation(
        transaction,
        anomalies,
        rag_context
    )

    print("\nFINAL EXPLANATION")
    print("=" * 60)

    print(explanation)


if __name__ == "__main__":
    main()