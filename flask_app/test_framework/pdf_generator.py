"""
PDF Report Generator
Converts the HTML test report to PDF using xhtml2pdf.
"""

import os
from datetime import datetime


def generate_pdf_report(report: dict, output_path: str) -> str:
    """
    Generate a PDF test report from the report dict.

    Args:
        report      (dict): Report dict from TestReport.to_dict()
        output_path (str):  File path to write the PDF (e.g. static/reports/report.pdf)

    Returns:
        str: Path to the generated PDF, or "" on failure.
    """
    try:
        from xhtml2pdf import pisa
    except ImportError:
        print("❌ xhtml2pdf not installed. Run: pip install xhtml2pdf")
        return ""

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    html = _build_pdf_html(report)

    try:
        with open(output_path, "wb") as f:
            result = pisa.CreatePDF(html, dest=f, encoding="utf-8")

        if result.err:
            print(f"❌ PDF generation error: {result.err}")
            return ""

        print(f"✅ PDF report generated: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ PDF generation exception: {e}")
        return ""


def _status_icon(status: str) -> str:
    return {"pass": "PASS", "fail": "FAIL", "error": "ERROR", "skip": "SKIP"}.get(status, status.upper())


def _status_color(status: str) -> str:
    return {"pass": "#2e7d32", "fail": "#c62828", "error": "#b71c1c", "skip": "#777"}.get(status, "#333")


def _status_bg(status: str) -> str:
    return {"pass": "#e6f4ea", "fail": "#fdecea", "error": "#ffebee", "skip": "#f5f5f5"}.get(status, "#fff")


def _build_pdf_html(report: dict) -> str:
    summary = report.get("summary", {})
    results = report.get("results", [])
    user    = report.get("user", {})
    overall = report.get("overall_status", "UNKNOWN")

    ov_color = {"PASSED": "#2e7d32", "FAILED": "#c62828", "PARTIAL": "#e65100"}.get(overall, "#555")
    ov_bg    = {"PASSED": "#e6f4ea", "FAILED": "#fdecea", "PARTIAL": "#fff3e0"}.get(overall, "#f5f5f5")
    ov_icon  = {"PASSED": "✅ AUTOMATION COMPLETED",
                "FAILED": "❌ AUTOMATION FAILED",
                "PARTIAL": "⚠️ AUTOMATION PARTIAL"}.get(overall, overall)

    # Group by phase
    phases = {}
    for r in results:
        ph = r.get("phase", "General")
        phases.setdefault(ph, []).append(r)

    phase_html = ""
    for phase_name, phase_results in phases.items():
        rows = ""
        for i, r in enumerate(phase_results, 1):
            status_color = _status_color(r["status"])
            status_bg    = _status_bg(r["status"])
            query        = r.get("query") or "—"
            actual       = (r.get("actual_response") or r.get("error") or "—")[:120]
            acc          = r.get("overall_accuracy", 0)
            matched      = ", ".join(r.get("matched_keywords", [])[:4]) or "—"
            rows += f"""
            <tr>
              <td style="padding:6px 5px;font-size:9px;color:#888;">{i}</td>
              <td style="padding:6px 5px;font-size:9px;font-weight:bold;">{r['name']}</td>
              <td style="padding:6px 5px;font-size:9px;color:#555;font-style:italic;">{query}</td>
              <td style="padding:6px 5px;text-align:center;">
                <span style="background:{status_bg};color:{status_color};padding:2px 6px;
                             border-radius:3px;font-size:8px;font-weight:bold;">
                  {_status_icon(r['status'])}
                </span>
              </td>
              <td style="padding:6px 5px;font-size:9px;color:#555;">{r.get('response_ms',0)}ms</td>
              <td style="padding:6px 5px;font-size:9px;color:#444;">{actual}</td>
              <td style="padding:6px 5px;font-size:9px;color:{ov_color};">{acc}%</td>
              <td style="padding:6px 5px;font-size:9px;color:#2e7d32;">{matched}</td>
            </tr>"""

        phase_html += f"""
        <div style="margin-bottom:20px;">
          <h3 style="font-size:11px;color:#1a73e8;margin:0 0 8px;padding-bottom:4px;
                     border-bottom:2px solid #e8f0fe;">{phase_name}</h3>
          <table style="width:100%;border-collapse:collapse;border:1px solid #e0e0e0;">
            <thead>
              <tr style="background:#f5f7ff;">
                <th style="padding:6px 5px;font-size:8px;color:#888;text-align:left;">#</th>
                <th style="padding:6px 5px;font-size:8px;color:#888;text-align:left;">Test Case</th>
                <th style="padding:6px 5px;font-size:8px;color:#888;text-align:left;">Query</th>
                <th style="padding:6px 5px;font-size:8px;color:#888;text-align:left;">Status</th>
                <th style="padding:6px 5px;font-size:8px;color:#888;text-align:left;">Time</th>
                <th style="padding:6px 5px;font-size:8px;color:#888;text-align:left;">Bot Response</th>
                <th style="padding:6px 5px;font-size:8px;color:#888;text-align:left;">Accuracy</th>
                <th style="padding:6px 5px;font-size:8px;color:#888;text-align:left;">Matched</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8"/>
  <style>
    @page {{ size: A4 landscape; margin: 15mm; }}
    body {{ font-family: Helvetica, Arial, sans-serif; font-size: 10px; color: #222; margin: 0; }}
    * {{ box-sizing: border-box; }}
  </style>
</head>
<body>

  <!-- Header -->
  <div style="background:#1a73e8;color:#fff;padding:14px 18px;border-radius:6px;margin-bottom:14px;">
    <div style="font-size:15px;font-weight:bold;">DigiTranVA — Voice AI Automated Test Report</div>
    <div style="font-size:9px;margin-top:4px;opacity:0.85;">
      Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp;
      Tester: {user.get('full_name','N/A')} ({user.get('email','N/A')}) &nbsp;|&nbsp;
      Mobile: {user.get('mobile','N/A')}
    </div>
  </div>

  <!-- Overall Status -->
  <div style="background:{ov_bg};border:2px solid {ov_color};border-radius:6px;
              padding:12px;margin-bottom:14px;text-align:center;">
    <div style="font-size:14px;font-weight:bold;color:{ov_color};">{ov_icon}</div>
    <div style="font-size:9px;color:#555;margin-top:4px;">
      {summary.get('passed',0)} of {summary.get('total',0)} tests passed &nbsp;·&nbsp;
      Pass rate: {summary.get('pass_rate',0)}% &nbsp;·&nbsp;
      Avg accuracy: {summary.get('avg_accuracy',0)}% &nbsp;·&nbsp;
      Avg response time: {summary.get('avg_response_ms',0)}ms
    </div>
  </div>

  <!-- Summary Cards -->
  <table style="width:100%;margin-bottom:16px;border-collapse:separate;border-spacing:6px;">
    <tr>
      <td style="background:#e8f0fe;border-radius:5px;padding:8px;text-align:center;width:14%;">
        <div style="font-size:16px;font-weight:bold;color:#1a73e8;">{summary.get('total',0)}</div>
        <div style="font-size:8px;color:#888;">Total</div>
      </td>
      <td style="background:#e6f4ea;border-radius:5px;padding:8px;text-align:center;width:14%;">
        <div style="font-size:16px;font-weight:bold;color:#2e7d32;">{summary.get('passed',0)}</div>
        <div style="font-size:8px;color:#888;">Passed</div>
      </td>
      <td style="background:#fdecea;border-radius:5px;padding:8px;text-align:center;width:14%;">
        <div style="font-size:16px;font-weight:bold;color:#c62828;">{summary.get('failed',0)}</div>
        <div style="font-size:8px;color:#888;">Failed</div>
      </td>
      <td style="background:#ffebee;border-radius:5px;padding:8px;text-align:center;width:14%;">
        <div style="font-size:16px;font-weight:bold;color:#b71c1c;">{summary.get('errors',0)}</div>
        <div style="font-size:8px;color:#888;">Errors</div>
      </td>
      <td style="background:#f5f5f5;border-radius:5px;padding:8px;text-align:center;width:14%;">
        <div style="font-size:16px;font-weight:bold;color:#777;">{summary.get('skipped',0)}</div>
        <div style="font-size:8px;color:#888;">Skipped</div>
      </td>
      <td style="background:#f3e8ff;border-radius:5px;padding:8px;text-align:center;width:14%;">
        <div style="font-size:16px;font-weight:bold;color:#7c4dff;">{summary.get('avg_accuracy',0)}%</div>
        <div style="font-size:8px;color:#888;">Avg Accuracy</div>
      </td>
      <td style="background:#e0f7fa;border-radius:5px;padding:8px;text-align:center;width:14%;">
        <div style="font-size:16px;font-weight:bold;color:#00838f;">{summary.get('avg_response_ms',0)}ms</div>
        <div style="font-size:8px;color:#888;">Avg Response</div>
      </td>
    </tr>
  </table>

  <!-- Phase Results -->
  {phase_html}

  <!-- Footer -->
  <div style="text-align:center;font-size:8px;color:#aaa;margin-top:10px;
              border-top:1px solid #eee;padding-top:8px;">
    DigiTran Solutions — AI Testing Automation Platform &nbsp;|&nbsp;
    Report generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  </div>

</body>
</html>"""
