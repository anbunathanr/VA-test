"""
DynamoDB Session Backend
Replaces Flask's cookie-based session with DynamoDB-backed server-side sessions.

Key fix: always save the session even if not explicitly modified,
so the cookie + DynamoDB token survive Lambda redirects.
"""

import os
import json
import uuid
import time
import boto3
from flask.sessions import SessionInterface, SessionMixin
from werkzeug.datastructures import CallbackDict


SESSIONS_TABLE      = os.environ.get("DYNAMO_SESSIONS_TABLE", "Sessions")
SESSION_TTL_SECONDS = 3600   # 1 hour
REGION              = os.environ.get("AWS_REGION", "us-east-1")


class DynamoSession(CallbackDict, SessionMixin):
    def __init__(self, initial=None, sid=None, new=False):
        def on_update(self):
            self.modified = True
        CallbackDict.__init__(self, initial or {}, on_update)
        self.sid      = sid
        self.new      = new
        self.modified = False


class DynamoSessionInterface(SessionInterface):
    """Server-side session backed by DynamoDB 'Sessions' table (hash key: token)."""

    def _get_table(self):
        return boto3.resource("dynamodb", region_name=REGION).Table(SESSIONS_TABLE)

    def open_session(self, app, request):
        cookie_name = app.config.get("SESSION_COOKIE_NAME", "digitranva_session")
        sid = request.cookies.get(cookie_name)
        if not sid:
            return DynamoSession(sid=str(uuid.uuid4()), new=True)
        try:
            resp = self._get_table().get_item(Key={"token": sid})
            item = resp.get("Item")
            if not item:
                return DynamoSession(sid=str(uuid.uuid4()), new=True)
            if item.get("ttl", 0) < int(time.time()):
                return DynamoSession(sid=str(uuid.uuid4()), new=True)
            data = json.loads(item.get("data", "{}"))
            return DynamoSession(data, sid=sid, new=False)
        except Exception as e:
            print(f"Session open error: {e}")
            return DynamoSession(sid=str(uuid.uuid4()), new=True)

    def save_session(self, app, session, response):
        cookie_name = app.config.get("SESSION_COOKIE_NAME", "digitranva_session")
        domain      = self.get_cookie_domain(app)

        # Delete empty modified sessions
        if not session and session.modified:
            try:
                self._get_table().delete_item(Key={"token": session.sid})
            except Exception:
                pass
            response.delete_cookie(cookie_name, domain=domain)
            return

        # Always save if session has any data OR is new with data
        if not session and not session.modified:
            return

        ttl = int(time.time()) + SESSION_TTL_SECONDS
        try:
            self._get_table().put_item(Item={
                "token": session.sid,
                "data":  json.dumps(dict(session)),
                "ttl":   ttl,
            })
        except Exception as e:
            print(f"Session save error: {e}")
            # Fall back: still set the cookie so at minimum the SID is preserved
            pass

        # Set cookie — must be SameSite=None for cross-origin CloudFront→API Gateway
        response.set_cookie(
            cookie_name,
            session.sid,
            httponly  = True,
            samesite  = "None",   # required for CloudFront → API Gateway cross-origin
            secure    = True,
            max_age   = SESSION_TTL_SECONDS,
            path      = "/",
        )
