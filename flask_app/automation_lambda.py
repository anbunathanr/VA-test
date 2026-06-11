"""
Automation Lambda Handler
This is the entry point for the separate Playwright automation Lambda.
Triggered asynchronously by the main Flask Lambda.

Deployed as a container image (needs Chromium inside the image).
⚠️  No AWS credentials in this file — uses Lambda execution role.
"""

import json
import os
import sys

# Add app directory to path
sys.path.insert(0, "/var/task")

import dynamo_results
from test_framework.runner import run_test_suite


def lambda_handler(event, context):
    """
    AWS Lambda handler for Playwright automation.

    Triggered with payload:
        { "full_name": "...", "email": "...", "mobile": "..." }
    """
    print(f"Automation Lambda triggered: {json.dumps(event)}")

    full_name = event.get("full_name", "")
    email     = event.get("email",     "")
    mobile    = event.get("mobile",    "")

    if not email:
        print("ERROR: No email in payload")
        return {"statusCode": 400, "body": "Missing email"}

    # result_store dict — synced to DynamoDB every 2s by run_test_suite
    result_store = {
        "status":       "running",
        "progress":     0,
        "current_test": "Starting...",
        "message":      "Automation in progress...",
        "results":      [],
        "summary":      None,
    }

    # Override run_test_suite to sync to DynamoDB at each update
    _orig_store_update = dict.update

    class _SyncingDict(dict):
        """Dict subclass that syncs to DynamoDB on every update."""
        def update(self, *args, **kwargs):
            _orig_store_update(self, *args, **kwargs)
            try:
                dynamo_results.store_result(email, dict(self))
            except Exception as e:
                print(f"DynamoDB sync error: {e}")

    syncing_store = _SyncingDict(result_store)

    try:
        run_test_suite(full_name, email, mobile, syncing_store)
        print(f"Automation complete for {email}: {syncing_store.get('overall_status')}")
        return {"statusCode": 200, "body": "Automation complete"}

    except Exception as e:
        print(f"Automation error: {e}")
        syncing_store.update({
            "status":  "failed",
            "message": f"Automation error: {e}",
        })
        return {"statusCode": 500, "body": str(e)}
