from datetime import datetime


def detect_anomalies(transaction):

    anomalies = []

    amount = transaction["amount"]
    customer_average = transaction["customer_average"]

    location = transaction["location"]
    usual_location = transaction["usual_location"]

    transaction_time = transaction["time"]

    # -------------------------
    # 1. Unusual Amount
    # -------------------------

    if amount > customer_average * 5:

        anomalies.append({
            "type": "Unusual Amount",
            "rule_id": "AMOUNT_001"
        })

    # -------------------------
    # 2. Unusual Location
    # -------------------------

    if location.lower() != usual_location.lower():

        anomalies.append({
            "type": "Unusual Location",
            "rule_id": "LOCATION_001"
        })

    # -------------------------
    # 3. Unusual Time
    # -------------------------

    hour = int(
        transaction_time.split(":")[0]
    )

    if hour >= 0 and hour < 5:

        anomalies.append({
            "type": "Unusual Time",
            "rule_id": "TIME_001"
        })

    # -------------------------
    # 4. High Transaction Frequency
    # -------------------------

    frequency = transaction.get(
        "transactions_last_hour",
        0
    )

    if frequency >= 5:

        anomalies.append({
            "type": "High Transaction Frequency",
            "rule_id": "FREQUENCY_001"
        })

    return anomalies