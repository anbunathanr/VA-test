"""
Email Sender
Sends the PDF test report to the user via SMTP (Google Workspace).
Uses the same SMTP credentials as the OTP system.
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from email.mime.base      import MIMEBase
from email                import encoders
from datetime             import datetime


# ── SMTP config (same as OTP system — Google Workspace) ──────────────────────
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465
SMTP_USER = "ceo@digitransolutions.in"


def _load_smtp_pass() -> str:
    """Load SMTP password from .env file or environment variable."""
    # Try environment variable first
    pwd = os.environ.get("SMTP_PASS", "")
    if pwd:
        return pwd

    # Try reading from parent .env file
    env_paths = [
        os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
        os.path.join(os.path.dirname(__file__), "..", ".env"),
        ".env",
    ]
    for env_path in env_paths:
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("SMTP_PASS="):
                        return line.split("=", 1)[1].strip()
    return ""


def send_report_email(
    recipient_email: str,
    recipient_name:  str,
    pdf_path:        str,
    report:          dict,
) -> tuple:
    """
    Send the PDF test report to the user via email.

    Args:
        recipient_email (str):  User's email address.
        recipient_name  (str):  User's full name.
        pdf_path        (str):  Absolute or relative path to the PDF file.
        report          (dict): Report summary dict for inline email content.

    Returns:
        tuple: (bool: success, str: message)
    """
    smtp_pass = _load_smtp_pass()
    if not smtp_pass or smtp_pass in ("YOUR_APP_PASSWORD", "abcdefghijklmnop", "wxyzabcdefghijkl"):
        msg = "SMTP password not configured. Please set SMTP_PASS in .env"
        print(f"⚠️  {msg}")
        return (False, msg)

    if not os.path.exists(pdf_path):
        return (False, f"PDF file not found: {pdf_path}")

    summary = report.get("summary", {})
    overall = report.get("overall_status", "UNKNOWN")
    ov_icon = {"PASSED": "✅", "FAILED": "❌", "PARTIAL": "⚠️"}.get(overall, "")
    ov_color = {"PASSED": "#2e7d32", "FAILED": "#c62828", "PARTIAL": "#e65100"}.get(overall, "#555")
    ov_bg    = {"PASSED": "#e6f4ea", "FAILED": "#fdecea", "PARTIAL": "#fff3e0"}.get(overall, "#f5f5f5")

    # ── Build email HTML body ─────────────────────────────────────────────────
    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:auto;
                border:1px solid #e0e0e0;border-radius:8px;overflow:hidden;">

      <div style="background:#1a73e8;padding:24px;text-align:center;">
        <h2 style="color:#fff;margin:0;font-size:18px;">🤖 DigiTranVA — Test Report</h2>
        <p style="color:rgba(255,255,255,0.85);margin:6px 0 0;font-size:12px;">
          AI Testing Automation Platform
        </p>
      </div>

      <div style="padding:28px 30px;">
        <p style="font-size:15px;color:#333;">Hello <strong>{recipient_name}</strong>,</p>
        <p style="font-size:14px;color:#555;margin-top:8px;">
          Your automated test run for the <strong>DigitranVA Voice AI Assistant</strong>
          has completed. Please find the full PDF report attached to this email.
        </p>

        <!-- Overall status box -->
        <div style="background:{ov_bg};border:2px solid {ov_color};border-radius:8px;
                    padding:16px;margin:20px 0;text-align:center;">
          <div style="font-size:20px;font-weight:800;color:{ov_color};">
            {ov_icon} AUTOMATION {overall}
          </div>
          <div style="font-size:13px;color:#555;margin-top:6px;">
            {summary.get('passed',0)} of {summary.get('total',0)} tests passed &nbsp;·&nbsp;
            Pass rate: {summary.get('pass_rate',0)}%
          </div>
        </div>

        <!-- Summary table -->
        <table style="width:100%;border-collapse:collapse;margin:16px 0;">
          <tr>
            <td style="padding:10px;background:#f8faff;border-radius:6px;text-align:center;">
              <div style="font-size:22px;font-weight:800;color:#1a73e8;">
                {summary.get('total',0)}
              </div>
              <div style="font-size:11px;color:#888;">Total Tests</div>
            </td>
            <td style="width:8px;"></td>
            <td style="padding:10px;background:#e6f4ea;border-radius:6px;text-align:center;">
              <div style="font-size:22px;font-weight:800;color:#2e7d32;">
                {summary.get('passed',0)}
              </div>
              <div style="font-size:11px;color:#888;">Passed</div>
            </td>
            <td style="width:8px;"></td>
            <td style="padding:10px;background:#fdecea;border-radius:6px;text-align:center;">
              <div style="font-size:22px;font-weight:800;color:#c62828;">
                {summary.get('failed',0) + summary.get('errors',0)}
              </div>
              <div style="font-size:11px;color:#888;">Failed/Errors</div>
            </td>
            <td style="width:8px;"></td>
            <td style="padding:10px;background:#f3e8ff;border-radius:6px;text-align:center;">
              <div style="font-size:22px;font-weight:800;color:#7c4dff;">
                {summary.get('avg_accuracy',0)}%
              </div>
              <div style="font-size:11px;color:#888;">Avg Accuracy</div>
            </td>
            <td style="width:8px;"></td>
            <td style="padding:10px;background:#e0f7fa;border-radius:6px;text-align:center;">
              <div style="font-size:22px;font-weight:800;color:#00838f;">
                {summary.get('avg_response_ms',0)}ms
              </div>
              <div style="font-size:11px;color:#888;">Avg Response</div>
            </td>
          </tr>
        </table>

        <p style="font-size:13px;color:#888;margin-top:20px;">
          The full detailed report with per-test results, response accuracy scores,
          and matched keywords is attached as a PDF.
        </p>
        <p style="font-size:12px;color:#aaa;margin-top:8px;">
          Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
      </div>

      <div style="background:#f5f5f5;padding:14px;text-align:center;">
        <p style="font-size:11px;color:#aaa;margin:0;">
          © {datetime.now().year} DigiTran Solutions — AI Testing Automation Platform
        </p>
      </div>
    </div>
    """

    # ── Build MIME message ────────────────────────────────────────────────────
    msg = MIMEMultipart("mixed")
    msg["From"]    = f'"DigiTran Solutions" <{SMTP_USER}>'
    msg["To"]      = recipient_email
    msg["Subject"] = f"[DigiTranVA] Test Report — {overall} — {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    # HTML body
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    # PDF attachment
    pdf_filename = os.path.basename(pdf_path)
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()

    part = MIMEBase("application", "pdf")
    part.set_payload(pdf_data)
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f'attachment; filename="{pdf_filename}"')
    msg.attach(part)

    # ── Send via SMTP ─────────────────────────────────────────────────────────
    try:
        print(f"📧 Sending report email to {recipient_email}...")
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.login(SMTP_USER, smtp_pass)
            server.sendmail(SMTP_USER, recipient_email, msg.as_bytes())

        print(f"✅ Report email sent to {recipient_email}")
        return (True, f"Report emailed to {recipient_email}")

    except smtplib.SMTPAuthenticationError:
        err = "SMTP authentication failed — check App Password in .env"
        print(f"❌ {err}")
        return (False, err)

    except smtplib.SMTPException as e:
        err = f"SMTP error: {e}"
        print(f"❌ {err}")
        return (False, err)

    except Exception as e:
        err = f"Email send failed: {e}"
        print(f"❌ {err}")
        return (False, err)
