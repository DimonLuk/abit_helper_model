import json
import pandas as pd
from joblib import load


def map_data_from_query(event):
    data = {
        "district": event["queryStringParameters"]["district"],
        "specialty": event["queryStringParameters"]["specialty"],
        "points": event["queryStringParameters"]["points"],
        "kind": event["queryStringParameters"]["kind"],
        "university": event["queryStringParameters"]["university"],
    }
    data["specialty"] = int(data["specialty"][:3])
    data["points"] = float(data["points"])
    return data


def get_positive_prediction(data, estimator):
    df = pd.DataFrame(
        [[data["district"], data["university"], data["specialty"], data["points"]]]
    )
    return estimator.predict_proba(df.values)[0][1]


def lambda_handler(event, context):
    budget = load("budget_after_greed_search.joblib")
    contract = load("contract_after_greed_search.joblib")

    data = map_data_from_query(event)
    if data["kind"] == "budget":
        estimator = budget
    else:
        estimator = contract

    prediction = get_positive_prediction(data, estimator)

    return {
        "statusCode": 200,
        "body": json.dumps({"prediction": prediction}),
        "headers": {
            "Access-Control-Allow-Headers": (
                "Content-Type,X-Amz-Date,Authorization,"
                "X-Api-Key,X-Amz-Security-Token,"
                "Access-Control-Allow-Origin"
            ),
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS,HEAD",
        },
    }
