"""
DynamoDB Automation Results Store
Uses 'digitranva-conversations' table (email HASH + timestamp RANGE).
We store automation results under a fixed timestamp key "automation_result".
"""

import os, json, time, boto3

RESULTS_TABLE  = os.environ.get("RESULTS_TABLE", "digitranva-conversations")
REGION         = os.environ.get("AWS_REGION", "us-east-1")
RESULT_TS_KEY  = "automation_result"   # fixed range key for automation results
TTL_SECONDS    = 86400                 # 24 hours


def _table():
    return boto3.resource("dynamodb", region_name=REGION).Table(RESULTS_TABLE)


def store_result(email: str, result_store: dict) -> None:
    try:
        _table().put_item(Item={
            "email":      email,
            "timestamp":  RESULT_TS_KEY,
            "data":       json.dumps(result_store, default=str),
            "updated_at": int(time.time()),
            "ttl":        int(time.time()) + TTL_SECONDS,
        })
    except Exception as e:
        print(f"Result store error: {e}")


def get_result(email: str) -> dict:
    try:
        resp = _table().get_item(Key={"email": email, "timestamp": RESULT_TS_KEY})
        item = resp.get("Item")
        if not item:
            return {"status": "idle", "message": "No test run yet."}
        return json.loads(item.get("data", "{}"))
    except Exception as e:
        print(f"Result get error: {e}")
        return {"status": "error", "message": str(e)}
