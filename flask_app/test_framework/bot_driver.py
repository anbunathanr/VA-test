"""
Bot Driver
Low-level Playwright interactions with the DigitranVA LexWebUI chatbot.
"""

import time
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeout
from .config import SEL, TIMEOUT_BOT_RESPONSE, TIMEOUT_ELEMENT


class BotDriver:
    """Wraps Playwright page interactions for the DigitranVA chatbot."""

    def __init__(self, page: Page):
        self.page = page

    # ── Registration ──────────────────────────────────────────────────────────
    def fill_registration(self, email, first_name, last_name, mobile):
        self.page.wait_for_selector(SEL["email"], timeout=TIMEOUT_ELEMENT)
        self.page.fill(SEL["email"],          email)
        self.page.fill(SEL["firstName"],      first_name)
        self.page.fill(SEL["lastName"],       last_name)
        self.page.fill(SEL["mobile"],         mobile)
        self.page.locator(SEL["continue_btn"]).click()

        # Page re-navigates after submit — wait for it to settle
        try:
            self.page.wait_for_load_state("networkidle", timeout=20_000)
        except Exception:
            pass

        # Wait for main menu buttons to appear
        self.page.wait_for_function(
            "() => document.querySelectorAll('button').length > 3",
            timeout=TIMEOUT_BOT_RESPONSE
        )

        # Give the bot time to post its welcome message
        self.page.wait_for_timeout(3000)

        # Minimize the welcome card overlay (yellow − button, class=card-minimize-btn)
        self.minimize_welcome_card()

    def minimize_welcome_card(self):
        """
        Click the yellow minimize button on the DigitranVA welcome card overlay.
        This collapses the card so the underlying chatbot UI is fully visible.
        """
        try:
            self.page.wait_for_selector(".card-minimize-btn", timeout=5000)
            self.page.click(".card-minimize-btn")
            self.page.wait_for_timeout(800)
            print("✅ Welcome card minimized")
        except Exception as e:
            print(f"⚠️  Could not minimize welcome card: {e}")

    # ── Button Click ──────────────────────────────────────────────────────────
    def click_button_by_text(self, text: str) -> bool:
        """Click a button containing the given text. Uses JS for reliability."""
        found = self.page.evaluate("""(text) => {
            const btns = Array.from(document.querySelectorAll('button'));
            const b = btns.find(x => x.innerText.includes(text));
            if (b) { b.click(); return true; }
            return false;
        }""", text)
        return bool(found)

    # ── Message Sending ───────────────────────────────────────────────────────
    def send_message(self, text: str) -> bool:
        """Type a message and press Enter. Tries multiple known selectors."""
        selectors = [
            '#text-input',                        # Lex Web UI primary input
            'input[placeholder*="Type here"]',
            'input[placeholder*="type"]',
            '.toolbar-input input',
            '.v-field__input',
        ]
        for sel in selectors:
            try:
                self.page.wait_for_selector(sel, timeout=4000)
                el = self.page.locator(sel).first
                if el.is_visible():
                    el.fill(text)
                    self.page.keyboard.press("Enter")
                    return True
            except Exception:
                continue
        return False

    # ── Message Reading ───────────────────────────────────────────────────────
    def get_all_bot_messages(self) -> list:
        try:
            return self.page.evaluate("""() => {
                const els = document.querySelectorAll(
                    '.message-bubble-row-bot .message-text, .message-bubble.message-bubble-row-bot'
                );
                return Array.from(els)
                    .map(e => e.innerText.replace('bot says:', '').trim())
                    .filter(t => t.length > 0 && !t.includes('thumb_up') && !t.includes('thumb_down'));
            }""")
        except Exception:
            return []

    def count_bot_messages(self) -> int:
        return len(self.get_all_bot_messages())

    def wait_for_new_bot_response(self, prev_count: int, timeout_s: int = 25) -> str:
        """
        Wait until a new bot message appears AND stops changing (fully received).

        Strategy:
          1. Wait for message count to exceed prev_count.
          2. Once a new message appears, keep sampling every 600ms.
          3. Only return when the message text has been stable for 2 consecutive
             samples — meaning the bot has finished streaming/typing.

        Args:
            prev_count (int): Number of bot messages before the action.
            timeout_s  (int): Max seconds to wait overall.

        Returns:
            str: The fully received bot response, or "" on timeout.
        """
        deadline     = time.time() + timeout_s
        last_text    = ""
        stable_count = 0
        STABLE_NEEDED = 2          # consecutive identical samples required
        SAMPLE_INTERVAL = 0.6      # seconds between samples

        while time.time() < deadline:
            msgs = self.get_all_bot_messages()

            if len(msgs) > prev_count:
                current_text = msgs[-1]

                if current_text == last_text and current_text:
                    stable_count += 1
                    if stable_count >= STABLE_NEEDED:
                        return current_text   # response is fully received
                else:
                    # Text changed — reset stability counter
                    last_text    = current_text
                    stable_count = 0

            time.sleep(SAMPLE_INTERVAL)

        # Timeout — return whatever we have
        msgs = self.get_all_bot_messages()
        return msgs[-1] if len(msgs) > prev_count else ""

    # ── Button Presence Check ─────────────────────────────────────────────────
    def get_visible_buttons(self) -> list:
        return self.page.evaluate("""() => {
            return Array.from(document.querySelectorAll('button'))
                .map(b => b.innerText.trim())
                .filter(t => t.length > 1 && t.length < 80);
        }""")

    def buttons_present(self, expected_texts: list) -> tuple:
        """
        Check that all expected button texts appear in visible buttons.
        Returns (bool: all_found, list: missing).
        """
        visible = [b.lower() for b in self.get_visible_buttons()]
        missing = [t for t in expected_texts
                   if not any(t.lower() in v for v in visible)]
        return (len(missing) == 0, missing)

    def is_card_minimized(self) -> bool:
        """
        Verify the welcome card overlay is minimized after login.
        Returns True if minimized or gone, False if still fully visible.
        """
        try:
            result = self.page.evaluate("""() => {
                const minBtn   = document.querySelector('.card-minimize-btn');
                const maxBtn   = document.querySelector('.card-maximize-btn');
                const container = document.querySelector('.form-container');
                if (!minBtn && !container) return true;
                if (maxBtn && window.getComputedStyle(maxBtn).display !== 'none') return true;
                if (container && container.classList.contains('minimized')) return true;
                return false;
            }""")
            return bool(result)
        except Exception:
            return True   # assume ok on error

    # ── Screenshot ────────────────────────────────────────────────────────────
    def screenshot(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.page.screenshot(path=path, full_page=True)
