"""
DynamoDB OTP Store
Replaces in-memory Flask session OTP storage with DynamoDB.
Reuses the existing 'OTP' DynamoDB table.
"""

import os
import time
import boto3
from boto3.dynamodb.conditions import Key

OTP_TABLE     = os.environ.get("OTP_TABLE", "OTP")
OTP_TTL       = 900   # 15 minutes
REGION        = os.environ.get("AWS_REGION", "us-east-1")


def _table():
    return boto3.resource("dynamodb", region_name=REGION).Table(OTP_TABLE)


def store_otp(email: str, otp: str, user_data: dict) -> bool:
    """Store OTP and user data in DynamoDB with TTL."""
    try:
        _table().put_item(Item={
            "otpId":      email,          # hash key — use email as the unique ID
            "email":      email,          # range key
            "otp":        otp,
            "user_data":  user_data,
            "ttl":        int(time.time()) + OTP_TTL,
            "created_at": int(time.time()),
        })
        return True
    except Exception as e:
        print(f"OTP store error: {e}")
        return False


def get_otp(email: str) -> dict | None:
    """Retrieve OTP record for an email. Returns None if expired or not found."""
    try:
        resp = _table().get_item(Key={"otpId": email, "email": email})
        item = resp.get("Item")
        if not item:
            return None
        if item.get("ttl", 0) < int(time.time()):
            delete_otp(email)
            return None
        return item
    except Exception as e:
        print(f"OTP get error: {e}")
        return None


def delete_otp(email: str) -> None:
    """Delete OTP after successful verification."""
    try:
        _table().delete_item(Key={"otpId": email, "email": email})
    except Exception as e:
        print(f"OTP delete error: {e}")


def validate_otp(email: str, entered_otp: str) -> tuple:
    """
    Validate OTP. Returns (bool valid, dict|None user_data).
    Deletes OTP from DynamoDB on success.
    """
    if not entered_otp or not email:
        return False, None

    record = get_otp(email)
    if not record:
        return False, None

    if str(record.get("otp", "")).strip() == str(entered_otp).strip():
        user_data = record.get("user_data", {})
        delete_otp(email)
        return True, user_data

    return False, None
