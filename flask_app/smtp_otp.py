"""
SMTP OTP Fallback
Sends OTP email directly via SMTP when n8n is unavailable.
Uses Google Workspace SMTP from ceo@digitransolutions.in.

Credentials come from environment variables ONLY — never hardcoded.
  SMTP_HOST  (default: smtp.gmail.com)
  SMTP_PORT  (default: 587)
  SMTP_USER  (default: ceo@digitransolutions.in)
  SMTP_PASS  — App Password from Google Workspace (required)
"""

import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText

SMTP_HOST   = os.environ.get("SMTP_HOST",  "smtp.gmail.com")
SMTP_PORT   = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER   = os.environ.get("SMTP_USER",  "ceo@digitransolutions.in")
SMTP_PASS   = os.environ.get("SMTP_PASS",  "")
SENDER_NAME = "DigiTran Solutions"


def send_otp_via_smtp(recipient_email: str, otp: str, recipient_name: str = "User") -> tuple:
    """
    Send OTP email via SMTP (Google Workspace App Password).

    Returns:
        (True,  "OTP sent via SMTP to user@example.com")  on success
        (False, "<error description>")                    on failure
    """
    if not SMTP_PASS:
        return False, "SMTP_PASS not configured"

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
    Do not share this code with anyone.<br>
    If you did not request this, please ignore this email.
  </p>
  <hr style="border:none;border-top:1px solid #eee;margin:24px 0;">
  <p style="color:#aaa;font-size:12px;text-align:center;">
    Sent from <a href="mailto:{SMTP_USER}" style="color:#1a73e8;">{SMTP_USER}</a>
  </p>
</div>
"""

    text_body = (
        f"Hello {recipient_name},\n\n"
        f"Your OTP for DigiTran Solutions AI Testing Automation Platform:\n\n"
        f"  {otp}\n\n"
        f"Valid for 15 minutes. Do not share this code.\n\n"
        f"DigiTran Solutions\n{SMTP_USER}"
    )

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"{SENDER_NAME} <{SMTP_USER}>"
        msg["To"]      = recipient_email

        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, recipient_email, msg.as_string())

        print(f"[SMTP] OTP sent to {recipient_email}")
        return True, f"OTP sent via SMTP to {recipient_email}"

    except Exception as e:
        print(f"[SMTP] Failed: {e}")
        return False, f"SMTP failed: {e}"
