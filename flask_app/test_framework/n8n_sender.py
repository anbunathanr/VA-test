"""
n8n Report Sender
Posts the PDF report (base64 encoded) and metadata to an n8n webhook.
n8n workflow receives it, decodes the PDF, and emails it to the user.
"""

import os
import base64
import json
import urllib.request
import urllib.error
from datetime import datetime


def _load_env(key: str, default: str = "") -> str:
    """Load a value from .env file or environment variable."""
    val = os.environ.get(key, "")
    if val:
        return val
    # Search parent .env files
    for path in [
        os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
        os.path.join(os.path.dirname(__file__), "..", ".env"),
        ".env",
    ]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"{key}="):
                        return line.split("=", 1)[1].strip()
    return default


def send_report_via_n8n(
    recipient_email: str,
    recipient_name:  str,
    pdf_path:        str,
    report:          dict,
) -> tuple:
    """
    Send the test report PDF to the user via n8n webhook.

    The webhook payload contains everything n8n needs:
      - recipient email & name
      - PDF as base64 string
      - summary stats for the email body
      - overall test status

    Args:
        recipient_email (str): User's email address.
        recipient_name  (str): User's full name.
        pdf_path        (str): Path to the generated PDF file.
        report          (dict): Full report dict from TestReport.to_dict()

    Returns:
        tuple: (bool success, str message)
    """
    webhook_url = _load_env("N8N_WEBHOOK_URL")

    if not webhook_url:
        return (False, "N8N_WEBHOOK_URL not set in .env")

    if not os.path.exists(pdf_path):
        return (False, f"PDF not found: {pdf_path}")

    # ── Read & encode PDF ─────────────────────────────────────────────────────
    with open(pdf_path, "rb") as f:
        pdf_bytes  = f.read()
    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_filename = os.path.basename(pdf_path)

    summary = report.get("summary", {})
    overall = report.get("overall_status", "UNKNOWN")
    user    = report.get("user", {})

    # ── Build payload ─────────────────────────────────────────────────────────
    payload = {
        # Recipient
        "recipient_email": recipient_email,
        "recipient_name":  recipient_name,

        # Email subject & status
        "overall_status":  overall,
        "subject": (
            f"[DigiTranVA] Test Report — {overall} — "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
        ),

        # Summary for email body
        "summary": {
            "total":           summary.get("total",           0),
            "passed":          summary.get("passed",          0),
            "failed":          summary.get("failed",          0),
            "errors":          summary.get("errors",          0),
            "skipped":         summary.get("skipped",         0),
            "pass_rate":       summary.get("pass_rate",       0),
            "avg_accuracy":    summary.get("avg_accuracy",    0),
            "avg_response_ms": summary.get("avg_response_ms", 0),
        },

        # Tester info
        "tester_name":   user.get("full_name", recipient_name),
        "tester_email":  user.get("email",     recipient_email),
        "tester_mobile": user.get("mobile",    ""),
        "generated_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        # PDF attachment
        "pdf_filename": pdf_filename,
        "pdf_base64":   pdf_base64,
        "pdf_size_kb":  round(len(pdf_bytes) / 1024, 1),
    }

    # ── POST to n8n webhook ───────────────────────────────────────────────────
    payload_bytes = json.dumps(payload).encode("utf-8")

    print(f"📤 Sending report to n8n webhook...")
    print(f"   URL      : {webhook_url}")
    print(f"   To       : {recipient_email}")
    print(f"   PDF size : {payload['pdf_size_kb']} KB")

    try:
        req = urllib.request.Request(
            webhook_url,
            data=payload_bytes,
            headers={
                "Content-Type": "application/json",
                "Accept":       "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            status_code   = resp.getcode()
            response_body = resp.read().decode("utf-8", errors="replace")

        print(f"✅ n8n webhook responded: HTTP {status_code}")
        print(f"   Response: {response_body[:200]}")

        if status_code in (200, 201):
            return (True, f"Report sent to {recipient_email} via n8n")
        else:
            return (False, f"n8n returned HTTP {status_code}: {response_body[:100]}")

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        msg  = f"n8n HTTP {e.code}: {body[:200]}"
        print(f"❌ {msg}")
        return (False, msg)

    except urllib.error.URLError as e:
        msg = f"n8n connection failed: {e.reason}"
        print(f"❌ {msg}")
        return (False, msg)

    except Exception as e:
        msg = f"n8n send error: {e}"
        print(f"❌ {msg}")
        return (False, msg)
