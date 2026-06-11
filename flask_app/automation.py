"""
Automation Entry Point
Launches the test suite in a background thread so Flask stays responsive.
"""

import threading
from test_framework.runner import run_test_suite


def run_automation_async(full_name: str, email: str, mobile: str) -> dict:
    """
    Start the Playwright test suite in a daemon thread.

    Args:
        full_name (str): User's full name.
        email     (str): User's email address.
        mobile    (str): User's mobile number.

    Returns:
        dict: result_store — shared dict updated live by the background thread.
    """
    result_store = {
        "status":       "running",
        "progress":     0,
        "current_test": "Starting...",
        "message":      "Automation in progress...",
        "results":      [],
        "summary":      None,
        "report":       None,
    }

    thread = threading.Thread(
        target=run_test_suite,
        args=(full_name, email, mobile, result_store),
        daemon=True
    )
    thread.start()
    return result_store
