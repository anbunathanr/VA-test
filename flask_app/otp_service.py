"""
OTP Service Module
Handles OTP generation, validation, and console display.
No SMTP or external dependencies — perfect for development/testing.
"""

import random
import string


class OTPService:
    """Service class for OTP operations."""

    @staticmethod
    def generate_otp():
        """
        Generate a random 6-digit OTP.

        Returns:
            str: A 6-digit numeric OTP string.
        """
        return ''.join(random.choices(string.digits, k=6))

    @staticmethod
    def validate_otp(session_otp, entered_otp):
        """
        Compare the session OTP with the user-entered OTP.

        Args:
            session_otp (str | None): OTP stored in the server session.
            entered_otp (str | None): OTP entered by the user.

        Returns:
            bool: True if both match, False otherwise.
        """
        # Handle None or empty values
        if not session_otp or not entered_otp:
            return False

        return str(session_otp).strip() == str(entered_otp).strip()

    @staticmethod
    def display_otp_console(recipient_email, otp, recipient_name="User"):
        """
        Print a formatted OTP to the terminal/console.
        Uses Unicode box-drawing characters for a clean display.

        Args:
            recipient_email (str): The recipient's email address.
            otp (str): The 6-digit OTP to display.
            recipient_name (str): The recipient's full name. Defaults to "User".

        Returns:
            tuple: (True, "OTP displayed in console (check terminal)")
        """
        border = "=" * 60

        # Build the OTP decorative box
        otp_box = (
            "  ╔═══════════╗\n"
            f"  ║  {otp}  ║\n"
            "  ╚═══════════╝"
        )

        message = f"""
{border}
📧 OTP EMAIL SIMULATION (Check Console)
{border}
From:    ceo@digitransolutions.in
To:      {recipient_email}
Subject: 🛡 Your OTP for AI Testing Automation Platform
{border}

Hello {recipient_name}!

Your One-Time Password (OTP) is:

{otp_box}

⚠️  This OTP is valid for 15 minutes.
⚠️  Do not share this code with anyone.

DigiTran Solutions - AI Testing Automation Platform
{border}
"""
        print(message)
        return (True, "OTP displayed in console (check terminal)")
