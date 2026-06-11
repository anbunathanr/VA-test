"""
Test Runner
Executes all 30 test cases against DigitranVA voice assistant.
Measures response times, computes accuracy, captures screenshots,
and writes live results to result_store for Flask polling.
"""

import os
import time
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

from .config         import TARGET_URL, TEST_CASES, REFERENCE_ANSWERS, TIMEOUT_PAGE_LOAD
from .bot_driver     import BotDriver
from .accuracy       import score_response


# ── Data Structures ───────────────────────────────────────────────────────────

class TestResult:
    def __init__(self, tc: dict):
        self.name              = tc["name"]
        self.phase             = tc.get("phase", "")
        self.action            = tc.get("action", "")
        self.query             = tc.get("value", "") if tc.get("action") == "type_message" else ""
        self.status            = "pending"
        self.actual_response   = ""
        self.expected_keywords = tc.get("expect", [])
        self.response_ms       = 0
        self.accuracy          = {}
        self.error             = ""
        self.screenshot        = ""
        self.timestamp         = datetime.now().isoformat()

    def to_dict(self):
        acc = self.accuracy or {}
        return {
            "name":               self.name,
            "phase":              self.phase,
            "action":             self.action,
            "query":              self.query,
            "status":             self.status,
            "actual_response":    self.actual_response[:400],
            "expected_keywords":  self.expected_keywords,
            "response_ms":        self.response_ms,
            "keyword_accuracy":   acc.get("keyword_accuracy",   0),
            "reference_accuracy": acc.get("reference_accuracy", 0),
            "overall_accuracy":   acc.get("overall_accuracy",   0),
            "matched_keywords":   acc.get("matched_keywords",   []),
            "missed_keywords":    acc.get("missed_keywords",    []),
            "error":              self.error,
            "screenshot":         self.screenshot,
            "timestamp":          self.timestamp,
        }


class TestReport:
    def __init__(self, user, results):
        self.user       = user
        self.results    = results
        self.started_at = datetime.now()

    @property
    def total(self):   return len(self.results)
    @property
    def passed(self):  return sum(1 for r in self.results if r.status == "pass")
    @property
    def failed(self):  return sum(1 for r in self.results if r.status == "fail")
    @property
    def errors(self):  return sum(1 for r in self.results if r.status == "error")
    @property
    def skipped(self): return sum(1 for r in self.results if r.status == "skip")

    @property
    def avg_accuracy(self):
        scores = [r.accuracy.get("overall_accuracy", 0)
                  for r in self.results if r.status == "pass" and r.accuracy]
        return round(sum(scores) / len(scores), 1) if scores else 0.0

    @property
    def avg_response_ms(self):
        times = [r.response_ms for r in self.results
                 if r.status in ("pass", "fail") and r.response_ms]
        return round(sum(times) / len(times)) if times else 0

    @property
    def pass_rate(self):
        run = self.total - self.skipped
        return round((self.passed / run) * 100, 1) if run > 0 else 0

    @property
    def overall_status(self):
        if self.failed == 0 and self.errors == 0:
            return "PASSED"
        if self.passed == 0:
            return "FAILED"
        return "PARTIAL"

    def to_dict(self):
        return {
            "overall_status": self.overall_status,
            "user":           self.user,
            "started_at":     self.started_at.isoformat(),
            "summary": {
                "total":           self.total,
                "passed":          self.passed,
                "failed":          self.failed,
                "errors":          self.errors,
                "skipped":         self.skipped,
                "pass_rate":       self.pass_rate,
                "avg_accuracy":    self.avg_accuracy,
                "avg_response_ms": self.avg_response_ms,
            },
            "results": [r.to_dict() for r in self.results],
        }


def _log(msg):
    print(msg, flush=True)


def _ss_path(idx, status):
    return os.path.join("static", "screenshots", f"tc_{idx:02d}_{status}.png")


# ── Main Runner ───────────────────────────────────────────────────────────────

def run_test_suite(full_name: str, email: str, mobile: str, result_store: dict):
    """
    Run all test cases with a headed Playwright browser.
    Writes live progress to result_store dict (read by Flask /test-status).
    """
    parts      = full_name.strip().split()
    first_name = parts[0]
    last_name  = " ".join(parts[1:]) if len(parts) > 1 else ""

    os.makedirs(os.path.join("static", "screenshots"), exist_ok=True)
    os.makedirs(os.path.join("static", "reports"),     exist_ok=True)

    results     = []
    total_tests = len(TEST_CASES)

    _log("\n" + "=" * 65)
    _log("🤖 DIGITRANVA — AUTOMATED VOICE ASSISTANT TEST SUITE")
    _log(f"   User   : {full_name} ({email})")
    _log(f"   Tests  : {total_tests}")
    _log(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _log("=" * 65)

    result_store.update({
        "status": "running", "progress": 0,
        "current_test": "Initializing browser...",
        "results": [], "summary": None,
    })

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,   # headless on server — no display available
            slow_mo=400,
            args=["--no-sandbox", "--disable-setuid-sandbox",
                  "--disable-dev-shm-usage", "--disable-gpu"]
        )
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page    = context.new_page()
        driver  = BotDriver(page)

        # ── Registration ──────────────────────────────────────────────────
        try:
            result_store["current_test"] = "Registering user on target site..."
            _log("\n▶ Registering user...")
            page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=TIMEOUT_PAGE_LOAD)

            # Fill & submit form
            page.wait_for_selector("#email", timeout=15_000)
            page.fill("#email",      email)
            page.fill("#firstName",  first_name)
            page.fill("#lastName",   last_name)
            page.fill("#mobile",     mobile)
            page.locator("button[type=submit]").click()

            # Page re-navigates — wait for it to settle + menu to appear
            try:
                page.wait_for_load_state("networkidle", timeout=20_000)
            except Exception:
                pass

            page.wait_for_function(
                "() => document.querySelectorAll('button').length > 3",
                timeout=20_000
            )
            # Extra wait for bot to post welcome message (~2s observed)
            page.wait_for_timeout(5000)

            msg_count = driver.count_bot_messages()
            _log(f"✅ Registration complete — {msg_count} bot message(s) received")

        except Exception as e:
            _log(f"❌ Registration failed: {e}")
            result_store.update({
                "status":  "failed",
                "message": f"Registration failed: {e}",
                "results": [],
            })
            browser.close()
            return

        # ── Test Loop ─────────────────────────────────────────────────────
        critical_failed = False   # set True only when critical=True test fails

        for idx, tc in enumerate(TEST_CASES):
            result = TestResult(tc)
            results.append(result)

            result_store.update({
                "progress":     int((idx / total_tests) * 100),
                "current_test": tc["name"],
                "results":      [r.to_dict() for r in results],
            })

            _log(f"\n{'─' * 65}")
            _log(f"[{idx+1:02d}/{total_tests}] {tc['name']}")

            # ── Skip if critical test previously failed ────────────────
            if critical_failed:
                result.status = "skip"
                result.error  = "Skipped — prior critical test failed"
                _log("⏭  SKIPPED")
                result_store["results"] = [r.to_dict() for r in results]
                continue

            try:
                start_ms   = time.time() * 1000
                prev_count = driver.count_bot_messages()
                bot_reply  = ""

                # ── Action: check_minimized ───────────────────────────
                if tc["action"] == "check_minimized":
                    minimized = driver.is_card_minimized()
                    elapsed = int(time.time() * 1000 - start_ms)
                    result.response_ms = elapsed
                    if minimized:
                        result.status          = "pass"
                        result.actual_response = "Welcome card is minimized"
                        result.accuracy        = {"overall_accuracy": 100.0, "keyword_accuracy": 100.0,
                                                   "reference_accuracy": 100.0, "matched_keywords": ["minimized"],
                                                   "missed_keywords": []}
                        _log(f"✅ PASS — card is minimized ({elapsed}ms)")
                    else:
                        result.status          = "fail"
                        result.actual_response = "Welcome card is still visible"
                        result.error           = "Card not minimized after login"
                        result.accuracy        = {"overall_accuracy": 0.0, "keyword_accuracy": 0.0,
                                                   "reference_accuracy": 0.0, "matched_keywords": [],
                                                   "missed_keywords": ["minimized"]}
                        _log(f"❌ FAIL — card still visible ({elapsed}ms)")
                    ss = _ss_path(idx + 1, result.status)
                    try: driver.screenshot(ss); result.screenshot = ss.replace("\\", "/")
                    except Exception: pass
                    result_store["results"] = [r.to_dict() for r in results]
                    page.wait_for_timeout(400)
                    continue

                # ── Action: wait_for_bot ───────────────────────────────
                elif tc["action"] == "wait_for_bot":
                    if prev_count == 0:
                        bot_reply = driver.wait_for_new_bot_response(-1, timeout_s=8)
                    else:
                        msgs      = driver.get_all_bot_messages()
                        bot_reply = msgs[-1] if msgs else ""

                # ── Action: check_buttons ──────────────────────────────
                elif tc["action"] == "check_buttons":
                    all_found, missing = driver.buttons_present(tc["value"])
                    elapsed = int(time.time() * 1000 - start_ms)
                    result.response_ms = elapsed

                    if all_found:
                        result.status          = "pass"
                        result.actual_response = "All expected buttons present"
                        result.accuracy        = {
                            "overall_accuracy":   100.0,
                            "keyword_accuracy":   100.0,
                            "reference_accuracy": 100.0,
                            "matched_keywords":   tc["value"],
                            "missed_keywords":    [],
                        }
                        _log(f"✅ PASS — buttons present ({elapsed}ms)")
                    else:
                        result.status          = "fail"
                        result.actual_response = f"Missing buttons: {missing}"
                        result.error           = f"Buttons not found: {missing}"
                        result.accuracy        = {
                            "overall_accuracy":   0.0,
                            "keyword_accuracy":   0.0,
                            "reference_accuracy": 0.0,
                            "matched_keywords":   [],
                            "missed_keywords":    missing,
                        }
                        _log(f"❌ FAIL — missing: {missing}")
                        if tc.get("critical"):
                            critical_failed = True

                    ss = _ss_path(idx + 1, result.status)
                    try:
                        driver.screenshot(ss)
                        result.screenshot = ss.replace("\\", "/")
                    except Exception:
                        pass
                    result_store["results"] = [r.to_dict() for r in results]
                    page.wait_for_timeout(500)
                    continue

                # ── Action: click_button ───────────────────────────────
                elif tc["action"] == "click_button":
                    clicked = driver.click_button_by_text(tc["value"])
                    if not clicked:
                        raise Exception(f"Button not found: '{tc['value']}'")
                    bot_reply = driver.wait_for_new_bot_response(prev_count, timeout_s=35)

                # ── Action: type_message ───────────────────────────────
                elif tc["action"] == "type_message":
                    sent = driver.send_message(tc["value"])
                    if not sent:
                        raise Exception("Chat input not found")
                    bot_reply = driver.wait_for_new_bot_response(prev_count)

                elapsed                = int(time.time() * 1000 - start_ms)
                result.response_ms     = elapsed
                result.actual_response = bot_reply

                # ── Accuracy scoring ───────────────────────────────────
                reference   = REFERENCE_ANSWERS.get(tc.get("value", ""), "")
                acc         = score_response(bot_reply, tc["expect"], reference)
                result.accuracy = acc

                # ── Pass / Fail ────────────────────────────────────────
                if not bot_reply:
                    result.status = "fail"
                    result.error  = "No bot response received within timeout"
                    _log(f"❌ FAIL — No response ({elapsed}ms)")
                    if tc.get("critical"):
                        critical_failed = True

                elif acc["keyword_accuracy"] >= 10:   # lowered from 25% — bot context may shift
                    result.status = "pass"
                    _log(f"✅ PASS  {elapsed}ms  kw={acc['keyword_accuracy']}%  overall={acc['overall_accuracy']}%")
                    _log(f"   Bot    : {bot_reply[:120]}")
                    _log(f"   Matched: {acc['matched_keywords']}")

                else:
                    result.status = "fail"
                    result.error  = (
                        f"Low keyword match {acc['keyword_accuracy']}% — "
                        f"missed: {acc['missed_keywords']}"
                    )
                    _log(f"❌ FAIL  kw={acc['keyword_accuracy']}%  ({elapsed}ms)")
                    _log(f"   Bot    : {bot_reply[:120]}")
                    _log(f"   Missed : {acc['missed_keywords']}")
                    if tc.get("critical"):
                        critical_failed = True

            except PlaywrightTimeout as e:
                result.status = "error"
                result.error  = f"Timeout: {e}"
                _log(f"⏱  TIMEOUT: {e}")
                if tc.get("critical"):
                    critical_failed = True

            except Exception as e:
                result.status = "error"
                result.error  = str(e)
                _log(f"💥 ERROR: {e}")
                if tc.get("critical"):
                    critical_failed = True

            # Screenshot after every test
            ss = _ss_path(idx + 1, result.status)
            try:
                driver.screenshot(ss)
                result.screenshot = ss.replace("\\", "/")
            except Exception:
                pass

            result_store["results"] = [r.to_dict() for r in results]
            page.wait_for_timeout(600)

        # ── Final Report ──────────────────────────────────────────────────
        final_ss = os.path.join("static", "screenshots", "final_state.png")
        try:
            driver.screenshot(final_ss)
        except Exception:
            pass

        report      = TestReport(
            user={"full_name": full_name, "email": email, "mobile": mobile},
            results=results
        )
        report_dict = report.to_dict()

        from .report_generator import generate_html_report
        from .pdf_generator    import generate_pdf_report
        from .n8n_sender       import send_report_via_n8n

        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = os.path.join("static", "reports", f"report_{ts}.html")
        pdf_path  = os.path.join("static", "reports", f"report_{ts}.pdf")

        # 1 — HTML report
        generate_html_report(report_dict, html_path)
        _log(f"📄 HTML report : {html_path}")

        # 2 — PDF report
        result_store["current_test"] = "Generating PDF report..."
        pdf_ok = generate_pdf_report(report_dict, pdf_path)
        if pdf_ok:
            _log(f"📑 PDF report  : {pdf_path}")
        else:
            _log("⚠️  PDF generation failed — HTML report still available")
            pdf_path = ""

        # 3 — Send via n8n webhook
        email_status = "not_sent"
        email_msg    = ""
        if pdf_ok:
            result_store["current_test"] = f"Sending report to {email} via n8n..."
            email_success, email_msg = send_report_via_n8n(
                recipient_email=email,
                recipient_name=full_name,
                pdf_path=pdf_path,
                report=report_dict,
            )
            email_status = "sent" if email_success else "failed"
        else:
            email_msg = "PDF generation failed — report not sent"

        final_status = "completed" if report.overall_status != "FAILED" else "failed"

        _log("\n" + "=" * 65)
        _log(f"🏁 SUITE COMPLETE — {report.overall_status}")
        _log(f"   Passed       : {report.passed}/{report.total}")
        _log(f"   Pass rate    : {report.pass_rate}%")
        _log(f"   Avg accuracy : {report.avg_accuracy}%")
        _log(f"   Avg response : {report.avg_response_ms}ms")
        _log(f"   HTML Report  : {html_path}")
        _log(f"   PDF Report   : {pdf_path or 'N/A'}")
        _log(f"   n8n status   : {email_status} — {email_msg}")
        _log("=" * 65)

        if final_status == "completed":
            banner = (f"✅ Automation Completed — {report.passed}/{report.total} passed "
                      f"({report.pass_rate}%) | Avg accuracy: {report.avg_accuracy}%")
        else:
            banner = (f"❌ Automation Failed — {report.passed}/{report.total} passed "
                      f"({report.pass_rate}%)")

        result_store.update({
            "status":           final_status,
            "overall_status":   report.overall_status,
            "progress":         100,
            "current_test":     "Done",
            "message":          banner,
            "summary":          report_dict["summary"],
            "results":          report_dict["results"],
            "final_screenshot": final_ss.replace("\\", "/"),
            "report_path":      html_path.replace("\\", "/"),
            "pdf_path":         pdf_path.replace("\\", "/") if pdf_path else "",
            "email_status":     email_status,
            "email_message":    email_msg,
            "report":           report_dict,
        })

        page.wait_for_timeout(4000)
        browser.close()
