"""
Authentication Routes — Stateless Token-Based Auth
===================================================
Because Lambda sits behind API Gateway + CloudFront, session cookies are
unreliable (domain mismatch).  We use signed URL tokens instead:

  • OTP token  (?t=...)   — carries email + OTP to verify page
  • Auth token (?auth=...) — carries user data to dashboard and all protected pages

Tokens are HMAC-SHA256 signed with FLASK_SECRET_KEY.
User data is also written to DynamoDB Sessions table as a backup.
"""

import os, re, base64, hashlib, hmac, json, time, uuid
import dynamo_results
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from otp_service import OTPService
from dynamo_otp  import store_otp, validate_otp
from n8n_otp     import send_otp_via_n8n

IS_LAMBDA = bool(os.environ.get("AWS_EXECUTION_ENV") or os.environ.get("LAMBDA_TASK_ROOT"))
SECRET    = os.environ.get("FLASK_SECRET_KEY", "digitranva-local-dev-only").encode()

auth_bp = Blueprint("auth", __name__)


# ── Token helpers ─────────────────────────────────────────────────────────────

def _make_token(payload: dict) -> str:
    """HMAC-sign a JSON payload → URL-safe token string."""
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    sig     = hmac.new(SECRET, encoded.encode(), hashlib.sha256).hexdigest()[:24]
    return f"{encoded}.{sig}"

def _read_token(token: str) -> dict | None:
    """Verify and decode a token. Returns dict or None."""
    try:
        parts = token.rsplit(".", 1)
        if len(parts) != 2:
            return None
        encoded, sig = parts
        expected = hmac.new(SECRET, encoded.encode(), hashlib.sha256).hexdigest()[:24]
        if not hmac.compare_digest(sig, expected):
            return None
        pad  = encoded + "=" * (4 - len(encoded) % 4)
        return json.loads(base64.urlsafe_b64decode(pad).decode())
    except Exception:
        return None

def _auth_token(user_data: dict) -> str:
    """Create a signed auth token carrying user data + expiry."""
    payload = {
        "u":   user_data,
        "exp": int(time.time()) + 3600,   # 1 hour
    }
    return _make_token(payload)

def _verify_auth(token: str) -> dict | None:
    """Return user_data if auth token is valid and not expired."""
    data = _read_token(token)
    if not data:
        return None
    if data.get("exp", 0) < int(time.time()):
        return None
    return data.get("u")

def _otp_token(email: str, otp: str) -> str:
    return _make_token({"e": email, "o": otp})

def _decode_otp_token(token: str):
    data = _read_token(token)
    if not data:
        return None, None
    return data.get("e"), data.get("o")


# ── Validators ────────────────────────────────────────────────────────────────

def is_valid_email(email):
    return bool(re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email.strip()))

def is_valid_mobile(mobile):
    return bool(re.match(r"^\d{7,15}$", re.sub(r"[\s\-\+]", "", mobile)))


# ── Routes ────────────────────────────────────────────────────────────────────

@auth_bp.route("/", methods=["GET"])
def index():
    # If already authenticated via URL token, go to dashboard
    auth = request.args.get("auth", "")
    if auth and _verify_auth(auth):
        return redirect(url_for("auth.dashboard", auth=auth))
    return render_template("index.html")


@auth_bp.route("/send-otp", methods=["POST"])
def send_otp():
    full_name = request.form.get("full_name", "").strip()
    email     = request.form.get("email",     "").strip()
    mobile    = request.form.get("mobile",    "").strip()

    errors = []
    if not full_name or len(full_name) < 2:
        errors.append("Full name must be at least 2 characters.")
    if not email or not is_valid_email(email):
        errors.append("Please enter a valid email address.")
    if not mobile or not is_valid_mobile(mobile):
        errors.append("Please enter a valid mobile number (7–15 digits).")

    if errors:
        for e in errors:
            flash(e, "error")
        return redirect(url_for("auth.index"))

    otp       = OTPService.generate_otp()
    user_data = {"full_name": full_name, "email": email, "mobile": mobile}
    store_otp(email, otp, user_data)

    # Log to CloudWatch always
    OTPService.display_otp_console(email, otp, full_name)

    # Try n8n → SMTP fallback
    sent, msg = send_otp_via_n8n(email, otp, full_name)
    if not sent:
        print(f"[WARN] n8n failed: {msg} — trying SMTP")
        from smtp_otp import send_otp_via_smtp
        sent, msg = send_otp_via_smtp(email, otp, full_name)

    if sent:
        flash(f"OTP sent to {email} — check your inbox.", "success")
    else:
        flash("OTP generated (demo mode — code shown below).", "info")

    t = _otp_token(email, otp)
    return redirect(url_for("auth.verify", t=t))


@auth_bp.route("/verify", methods=["GET"])
def verify():
    token = request.args.get("t", "")
    email, demo_otp = _decode_otp_token(token)
    if not email:
        flash("Please fill the form first.", "error")
        return redirect(url_for("auth.index"))
    return render_template("verify.html", email=email, token=token, demo_otp=demo_otp)


@auth_bp.route("/verify-otp", methods=["POST"])
def verify_otp():
    entered_otp = request.form.get("otp",         "").strip()
    token       = request.form.get("email_token", "").strip()

    email, _ = _decode_otp_token(token)
    if not email:
        flash("Session expired. Please sign in again.", "error")
        return redirect(url_for("auth.index"))

    if not entered_otp:
        flash("Please enter the OTP.", "error")
        return redirect(url_for("auth.verify", t=token))

    valid, user_data = validate_otp(email, entered_otp)

    if valid and user_data:
        auth_tok = _auth_token(user_data)
        # Also set cookie session for local runs
        if not IS_LAMBDA:
            session['authenticated'] = True
            session['current_user']  = user_data
            session.modified = True
        flash(f"Welcome, {user_data['full_name']}! You are now logged in.", "success")
        return redirect(url_for("auth.dashboard", auth=auth_tok))
    else:
        flash("Invalid OTP. Please try again.", "error")
        return redirect(url_for("auth.verify", t=token))


@auth_bp.route("/dashboard", methods=["GET"])
def dashboard():
    auth = request.args.get("auth", "")
    user = _verify_auth(auth)
    # Local fallback: accept Flask cookie session
    if not user and not IS_LAMBDA:
        user = session.get("current_user") if session.get("authenticated") else None
    if not user:
        flash("Please sign in first.", "error")
        return redirect(url_for("auth.index"))
    return render_template("dashboard.html", user=user, auth_token=auth)


@auth_bp.route("/run-test", methods=["POST"])
def run_test():
    auth = request.form.get("auth_token", "") or request.args.get("auth", "")
    user = _verify_auth(auth)
    # Local fallback: accept Flask cookie session
    if not user and not IS_LAMBDA:
        user = session.get("current_user") if session.get("authenticated") else None
    if not user:
        return jsonify({"status": "error", "message": "Not authenticated."}), 401

    full_name = user.get("full_name", "")
    email     = user.get("email",     "")
    mobile    = user.get("mobile",    "")

    if IS_LAMBDA:
        import boto3
        lc = boto3.client("lambda", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        automation_fn = os.environ.get("AUTOMATION_LAMBDA", "")
        if not automation_fn:
            return jsonify({
                "status":  "error",
                "message": "Playwright automation requires a local environment with a browser. "
                           "Open the app locally (python flask_app/app.py) and click Test there."
            }), 503
        try:
            lc.invoke(
                FunctionName   = automation_fn,
                InvocationType = "Event",
                Payload        = json.dumps({"full_name": full_name, "email": email, "mobile": mobile}),
            )
            dynamo_results.store_result(email, {
                "status": "running", "progress": 0,
                "current_test": "Starting automation Lambda...",
                "message": "Automation started.", "results": [],
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        from automation import run_automation_async
        import threading, time as _t
        rs = run_automation_async(full_name, email, mobile)
        dynamo_results.store_result(email, rs)
        def _sync():
            while rs.get("status") == "running":
                dynamo_results.store_result(email, rs); _t.sleep(2)
            dynamo_results.store_result(email, rs)
        threading.Thread(target=_sync, daemon=True).start()

    return jsonify({"status": "running", "message": "Automation started."})


@auth_bp.route("/test-status", methods=["GET"])
def test_status():
    auth  = request.args.get("auth", "")
    user  = _verify_auth(auth)
    if not user and not IS_LAMBDA:
        user = session.get("current_user") if session.get("authenticated") else None
    if not user:
        return jsonify({"status": "error", "message": "Not authenticated."}), 401
    email  = user.get("email", "")
    result = dynamo_results.get_result(email)
    return jsonify(result)


@auth_bp.route("/logout")
def logout():
    flash("You have been logged out.", "success")
    return redirect(url_for("auth.index"))
