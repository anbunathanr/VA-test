"""
Flask Application — Serverless Entry Point
Runs on AWS Lambda via Mangum, or locally via Flask dev server.
⚠️  AWS credentials must NEVER appear in this file.
    Use IAM roles on Lambda and environment variables for secrets.
"""

import os
import json
import base64
from flask import Flask, after_this_request, request as flask_request
from auth import auth_bp

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

# Secret key from environment variable — never hardcode
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "digitranva-local-dev-only")

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_NAME"]     = "digitranva_session"

# ── CORS — allow CloudFront origin to call API Gateway directly ───────────────
ALLOWED_ORIGINS = [
    "https://d2qrljay7xtfr5.cloudfront.net",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
]

@app.after_request
def add_cors(response):
    origin = flask_request.headers.get("Origin", "")
    if origin in ALLOWED_ORIGINS or origin.endswith(".cloudfront.net"):
        response.headers["Access-Control-Allow-Origin"]      = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"]     = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"]     = "Content-Type, Authorization"
    return response

@app.route("/<path:path>", methods=["OPTIONS"])
@app.route("/", methods=["OPTIONS"])
def options_handler(path=""):
    from flask import Response
    r = Response()
    origin = flask_request.headers.get("Origin", "")
    if origin in ALLOWED_ORIGINS or origin.endswith(".cloudfront.net"):
        r.headers["Access-Control-Allow-Origin"]  = origin
        r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return r, 204

# ── Session backend: DynamoDB in Lambda, cookie locally ───────────────────────
IS_LAMBDA = bool(os.environ.get("AWS_EXECUTION_ENV") or
                 os.environ.get("LAMBDA_TASK_ROOT"))

if IS_LAMBDA:
    from dynamo_session import DynamoSessionInterface
    app.session_interface = DynamoSessionInterface()

app.register_blueprint(auth_bp)


# ── Lambda WSGI handler (no external adapter needed) ─────────────────────────
def handler(event, context):
    """
    AWS Lambda handler — converts API Gateway v2 payload format 2.0
    into a WSGI environ and calls the Flask app.
    ⚠️  No credentials in this file — Lambda execution role provides access.
    """
    from io import BytesIO
    from urllib.parse import urlencode

    # ── Build WSGI environ from API Gateway event ─────────────────────────────
    method      = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    path        = event.get("rawPath", "/")
    qs          = event.get("rawQueryString", "")

    # Strip API Gateway stage prefix (e.g. /prod/foo → /foo)
    stage = event.get("requestContext", {}).get("stage", "")
    if stage and path.startswith("/" + stage):
        path = path[len("/" + stage):] or "/"
    headers     = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    body_raw    = event.get("body", "") or ""
    is_b64      = event.get("isBase64Encoded", False)
    body_bytes  = base64.b64decode(body_raw) if is_b64 else body_raw.encode("utf-8")

    environ = {
        "REQUEST_METHOD":    method,
        "PATH_INFO":         path,
        "QUERY_STRING":      qs,
        "CONTENT_TYPE":      headers.get("content-type", ""),
        "CONTENT_LENGTH":    str(len(body_bytes)),
        "SERVER_NAME":       headers.get("host", "localhost"),
        "SERVER_PORT":       "443",
        "SERVER_PROTOCOL":   "HTTP/1.1",
        "wsgi.version":      (1, 0),
        "wsgi.url_scheme":   "https",
        "wsgi.input":        BytesIO(body_bytes),
        "wsgi.errors":       __import__("sys").stderr,
        "wsgi.multithread":  False,
        "wsgi.multiprocess": False,
        "wsgi.run_once":     False,
    }

    # Add HTTP_ headers
    for key, val in headers.items():
        wsgi_key = "HTTP_" + key.upper().replace("-", "_")
        environ[wsgi_key] = val

    # ── Collect response via WSGI ─────────────────────────────────────────────
    response_started = {}
    response_body    = []

    def start_response(status, response_headers, exc_info=None):
        response_started["status"]  = status
        response_started["headers"] = dict(response_headers)

    result = app(environ, start_response)
    for chunk in result:
        response_body.append(chunk)

    body_out   = b"".join(response_body)
    status_int = int(response_started["status"].split(" ", 1)[0])

    # Detect binary content
    ct         = response_started["headers"].get("Content-Type", "")
    is_binary  = not (ct.startswith("text/") or
                      "json" in ct or "xml" in ct or "javascript" in ct)

    if is_binary:
        body_str   = base64.b64encode(body_out).decode("utf-8")
        encode_b64 = True
    else:
        body_str   = body_out.decode("utf-8", errors="replace")
        encode_b64 = False

    return {
        "statusCode":         status_int,
        "headers":            response_started["headers"],
        "body":               body_str,
        "isBase64Encoded":    encode_b64,
    }


# ── Local development server ───────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    print("=" * 60)
    print("  DigiTranVA — AI Testing Automation Platform")
    print(f"  Running at: http://localhost:{port}")
    print("  ⚠️  Running locally — using cookie sessions")
    print("=" * 60)
    # use_reloader=False prevents Flask from killing background threads (Playwright)
    app.run(debug=debug, host="0.0.0.0", port=port, use_reloader=False)
