"""
n8n OTP Sender
Sends OTP email via n8n webhook — from ceo@digitransolutions.in to the user.

The n8n workflow at N8N_OTP_WEBHOOK_URL should:
  1. Receive the JSON payload (POST)
  2. Send an email via the configured SMTP / Google Workspace node
  3. Return {"success": true} on success

Environment variables (set in Lambda / .env):
  N8N_OTP_WEBHOOK_URL  — e.g. https://n8n.digitransolutions.in/webhook/send-otp
"""

import os
import json
import urllib.request
import urllib.error

# Separate webhook for OTP (different from the report webhook)
N8N_OTP_WEBHOOK = os.environ.get(
    "N8N_OTP_WEBHOOK_URL",
    "https://n8n.digitransolutions.in/webhook/send-otp"
)

SENDER_EMAIL = "ceo@digitransolutions.in"
SENDER_NAME  = "DigiTran Solutions"


def send_otp_via_n8n(recipient_email: str, otp: str, recipient_name: str = "User") -> tuple:
    """
    POST OTP details to the n8n webhook so n8n sends the email.

    Payload sent to n8n:
        {
            "to_email":   "user@example.com",
            "to_name":    "John Doe",
            "from_email": "ceo@digitransolutions.in",
            "from_name":  "DigiTran Solutions",
            "subject":    "Your OTP for AI Testing Automation Platform",
            "otp":        "123456",
            "message":    "... HTML email body ..."
        }

    Returns:
        (True,  "OTP sent via n8n to user@example.com")  on success
        (False, "<error description>")                   on failure
    """
    subject = "Your OTP for AI Testing Automation Platform"

    html_body = f"""
<div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;padding:32px;
            border:1px solid #e0e0e0;border-radius:12px;background:#fff;">
  <div style="text-align:center;margin-bottom:24px;">
    <h2 style="color:#1a73e8;margin:0;">DigiTran Solutions</h2>
    <p style="color:#555;margin:4px 0 0;">AI Testing Automation Platform</p>
  </div>

  <p style="color:#333;">Hello <strong>{recipient_name}</strong>,</p>
  <p style="color:#333;">Use the OTP below to complete your sign-in.
     It is valid for <strong>15 minutes</strong>.</p>

  <div style="text-align:center;margin:32px 0;">
    <div style="display:inline-block;background:#f0f7ff;border:2px dashed #1a73e8;
                border-radius:10px;padding:18px 40px;">
      <span style="font-size:36px;font-weight:bold;letter-spacing:10px;color:#1a73e8;">
        {otp}
      </span>
    </div>
  </div>

  <p style="color:#888;font-size:13px;">
    ⚠️ Do not share this code with anyone.<br>
    If you did not request this OTP, please ignore this email.
  </p>

  <hr style="border:none;border-top:1px solid #eee;margin:24px 0;">
  <p style="color:#aaa;font-size:12px;text-align:center;">
    Sent from <a href="mailto:{SENDER_EMAIL}" style="color:#1a73e8;">{SENDER_EMAIL}</a>
  </p>
</div>
"""

    text_body = (
        f"Hello {recipient_name},\n\n"
        f"Your OTP for DigiTran Solutions AI Testing Automation Platform:\n\n"
        f"  {otp}\n\n"
        f"Valid for 15 minutes. Do not share this code.\n\n"
        f"DigiTran Solutions\n{SENDER_EMAIL}"
    )

    payload = {
        "to_email":   recipient_email,
        "to_name":    recipient_name,
        "from_email": SENDER_EMAIL,
        "from_name":  SENDER_NAME,
        "subject":    subject,
        "otp":        otp,
        "html":       html_body,
        "text":       text_body,
    }

    # Also print to console/CloudWatch as fallback visibility
    print(f"[OTP] Sending to {recipient_email} via n8n  OTP={otp}")

    try:
        data    = json.dumps(payload).encode("utf-8")
        req     = urllib.request.Request(
            N8N_OTP_WEBHOOK,
            data    = data,
            headers = {
                "Content-Type":   "application/json",
                "Content-Length": str(len(data)),
            },
            method = "POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.getcode()
            body   = resp.read().decode("utf-8", errors="replace")
            print(f"[n8n OTP] HTTP {status}  body={body[:200]}")

            if status in (200, 201):
                return True, f"OTP sent to {recipient_email} via n8n"
            else:
                return False, f"n8n returned HTTP {status}"

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        msg  = f"n8n HTTP {e.code}: {body[:200]}"
        print(f"[n8n OTP] ERROR {msg}")
        return False, msg

    except Exception as e:
        msg = f"n8n request failed: {e}"
        print(f"[n8n OTP] ERROR {msg}")
        return False, msg
