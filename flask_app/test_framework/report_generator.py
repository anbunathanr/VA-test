"""
HTML Report Generator
Produces a rich, standalone HTML test report with accuracy scoring.
"""

import os
from datetime import datetime


def _accuracy_bar(score: float) -> str:
    color = "#2e7d32" if score >= 70 else "#e65100" if score >= 40 else "#c62828"
    return (
        f'<div style="display:flex;align-items:center;gap:6px;">'
        f'<div style="flex:1;background:#eee;border-radius:4px;height:8px;overflow:hidden;">'
        f'<div style="width:{score}%;height:100%;background:{color};border-radius:4px;"></div>'
        f'</div><span style="font-size:11px;color:{color};font-weight:700;">{score}%</span>'
        f'</div>'
    )


def _badge(status: str) -> str:
    cfg = {
        "pass":  ("#2e7d32", "#e6f4ea", "✅ PASS"),
        "fail":  ("#c62828", "#fdecea", "❌ FAIL"),
        "error": ("#b71c1c", "#ffebee", "💥 ERROR"),
        "skip":  ("#555",    "#f5f5f5", "⏭ SKIP"),
    }.get(status, ("#555", "#f5f5f5", status.upper()))
    return (f'<span style="background:{cfg[1]};color:{cfg[0]};padding:3px 9px;'
            f'border-radius:12px;font-size:11px;font-weight:700;">{cfg[2]}</span>')


def generate_html_report(report: dict, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    summary = report.get("summary", {})
    results = report.get("results", [])
    user    = report.get("user", {})
    overall = report.get("overall_status", "UNKNOWN")

    ov_color = {"PASSED": "#2e7d32", "FAILED": "#c62828", "PARTIAL": "#e65100"}.get(overall, "#555")
    ov_bg    = {"PASSED": "#e6f4ea", "FAILED": "#fdecea", "PARTIAL": "#fff3e0"}.get(overall, "#f5f5f5")
    ov_icon  = {"PASSED": "✅", "FAILED": "❌", "PARTIAL": "⚠️"}.get(overall, "")

    # Group results by phase
    phases = {}
    for r in results:
        ph = r.get("phase", "General")
        phases.setdefault(ph, []).append(r)

    phase_html = ""
    for phase_name, phase_results in phases.items():
        rows = ""
        for i, r in enumerate(phase_results, 1):
            query   = r.get("query") or r.get("action", "")
            actual  = r.get("actual_response", "")[:150]
            matched = ", ".join(r.get("matched_keywords", [])[:5])
            missed  = ", ".join(r.get("missed_keywords",  [])[:5])
            ss_cell = (f'<a href="/{r["screenshot"]}" target="_blank" '
                       f'style="color:#1a73e8;font-size:11px;">📸</a>'
                       if r.get("screenshot") else "—")
            err_html = (f'<span style="color:#c62828;font-size:11px;">'
                        f'{r["error"][:80]}</span>' if r.get("error") else "")

            rows += f"""
            <tr style="border-bottom:1px solid #f3f3f3;">
              <td style="padding:9px 8px;font-size:12px;color:#888;">{i}</td>
              <td style="padding:9px 8px;font-size:12px;font-weight:500;">{r['name']}</td>
              <td style="padding:9px 8px;font-size:12px;color:#555;font-style:italic;">
                {query if query else '—'}
              </td>
              <td style="padding:9px 8px;">{_badge(r['status'])}</td>
              <td style="padding:9px 8px;font-size:12px;color:#555;">{r.get('response_ms',0)} ms</td>
              <td style="padding:9px 8px;font-size:11px;color:#444;max-width:180px;">{actual}</td>
              <td style="padding:9px 8px;min-width:100px;">{_accuracy_bar(r.get('overall_accuracy',0))}</td>
              <td style="padding:9px 8px;font-size:11px;color:#2e7d32;">{matched or '—'}</td>
              <td style="padding:9px 8px;font-size:11px;color:#c62828;">{missed or '—'}{err_html}</td>
              <td style="padding:9px 8px;">{ss_cell}</td>
            </tr>"""

        phase_html += f"""
        <div class="section">
          <h3>{phase_name}</h3>
          <table>
            <thead>
              <tr>
                <th>#</th><th>Test Case</th><th>Query</th><th>Status</th>
                <th>Time</th><th>Actual Response</th><th>Accuracy</th>
                <th>Matched</th><th>Missed / Error</th><th>SS</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>Voice AI Test Report — DigiTranVA</title>
  <style>
    body{{font-family:'Segoe UI',Arial,sans-serif;background:#f5f7fb;margin:0;padding:24px;color:#222;}}
    .container{{max-width:1200px;margin:auto;}}
    .header{{background:linear-gradient(135deg,#1a73e8,#7c4dff);color:#fff;border-radius:12px;padding:28px 32px;margin-bottom:20px;}}
    .header h1{{margin:0 0 6px;font-size:22px;}}
    .header p{{margin:0;font-size:13px;opacity:0.85;}}
    .overall{{border-radius:12px;padding:18px 24px;margin-bottom:20px;
              background:{ov_bg};border:2px solid {ov_color};text-align:center;}}
    .overall h2{{margin:0;font-size:26px;font-weight:800;color:{ov_color};}}
    .overall p{{margin:5px 0 0;font-size:13px;color:#555;}}
    .cards{{display:grid;grid-template-columns:repeat(7,1fr);gap:12px;margin-bottom:20px;}}
    .card{{background:#fff;border-radius:10px;padding:14px;text-align:center;
           box-shadow:0 2px 8px rgba(0,0,0,0.07);}}
    .card .num{{font-size:24px;font-weight:800;}}
    .card .lbl{{font-size:11px;color:#888;margin-top:3px;}}
    .section{{background:#fff;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.06);
              padding:22px;margin-bottom:20px;}}
    .section h3{{margin:0 0 14px;font-size:15px;color:#333;border-bottom:2px solid #f0f0f0;padding-bottom:8px;}}
    table{{width:100%;border-collapse:collapse;}}
    th{{text-align:left;padding:8px;font-size:11px;color:#888;text-transform:uppercase;
        border-bottom:2px solid #f0f0f0;white-space:nowrap;}}
    tr:hover{{background:#fafbff;}}
    .footer{{text-align:center;font-size:12px;color:#aaa;margin-top:16px;}}
  </style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>🤖 DigiTranVA — Voice AI Automated Test Report</h1>
    <p>AI Testing Automation Platform &nbsp;|&nbsp;
       Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp;
       Tester: {user.get('full_name','N/A')} ({user.get('email','N/A')})</p>
  </div>

  <div class="overall">
    <h2>{ov_icon} &nbsp; AUTOMATION {overall}</h2>
    <p>{summary.get('passed',0)} of {summary.get('total',0)} tests passed &nbsp;·&nbsp;
       Pass rate: {summary.get('pass_rate',0)}% &nbsp;·&nbsp;
       Avg accuracy: {summary.get('avg_accuracy',0)}% &nbsp;·&nbsp;
       Avg response: {summary.get('avg_response_ms',0)}ms</p>
  </div>

  <div class="cards">
    <div class="card"><div class="num" style="color:#1a73e8;">{summary.get('total',0)}</div><div class="lbl">Total</div></div>
    <div class="card"><div class="num" style="color:#2e7d32;">{summary.get('passed',0)}</div><div class="lbl">Passed</div></div>
    <div class="card"><div class="num" style="color:#c62828;">{summary.get('failed',0)}</div><div class="lbl">Failed</div></div>
    <div class="card"><div class="num" style="color:#b71c1c;">{summary.get('errors',0)}</div><div class="lbl">Errors</div></div>
    <div class="card"><div class="num" style="color:#888;">{summary.get('skipped',0)}</div><div class="lbl">Skipped</div></div>
    <div class="card"><div class="num" style="color:#7c4dff;">{summary.get('avg_accuracy',0)}%</div><div class="lbl">Avg Accuracy</div></div>
    <div class="card"><div class="num" style="color:#0097a7;">{summary.get('avg_response_ms',0)}ms</div><div class="lbl">Avg Response</div></div>
  </div>

  {phase_html}

  <div class="footer">
    DigiTran Solutions — AI Testing Automation Platform &nbsp;|&nbsp;
    Report generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  </div>
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
