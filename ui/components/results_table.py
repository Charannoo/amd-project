"""Ranked results table HTML — premium design."""


def render_results_table(rows: list[dict], max_rows: int = 12) -> str:
    if not rows:
        return '<p style="color:#5a6380;text-align:center;padding:20px">No results.</p>'

    html = """
<div style="
  background:linear-gradient(180deg, #0d1120 0%, #0a0a14 100%);
  border-radius:14px; border:1px solid #1e2548;
  overflow:hidden; box-shadow:0 4px 24px rgba(0,0,0,0.3);
">
<table style="width:100%;font-size:11px;border-collapse:collapse;color:#eaedf5;font-family:'Inter',sans-serif">
  <tr style="background:#131830">
    <th style="text-align:left;padding:10px 14px;color:#5a6380;font-weight:600;font-size:10px;text-transform:uppercase;letter-spacing:1px">#</th>
    <th style="text-align:left;padding:10px 14px;color:#5a6380;font-weight:600;font-size:10px;text-transform:uppercase;letter-spacing:1px">SMILES</th>
    <th style="text-align:left;padding:10px 14px;color:#5a6380;font-weight:600;font-size:10px;text-transform:uppercase;letter-spacing:1px">Score</th>
  </tr>"""

    for i, row in enumerate(rows[:max_rows]):
        smi = row.get("smiles") or row.get("final_smiles") or ""
        sc = row.get("similarity_score") or row.get("best_binding_score") or ""
        bg = "transparent" if i % 2 == 0 else "rgba(19,24,48,0.3)"
        html += f"""
  <tr style="border-bottom:1px solid #1e2548;background:{bg}">
    <td style="padding:8px 14px;color:#5a6380;font-weight:500">{i+1}</td>
    <td style="padding:8px 14px;word-break:break-all;font-family:'JetBrains Mono',monospace;font-size:10px;color:#8a92b0">{smi[:72]}</td>
    <td style="padding:8px 14px;color:#7c6ff7;font-weight:600;font-family:'JetBrains Mono',monospace">{sc}</td>
  </tr>"""

    return html + "</table></div>"


def render_candidate_comparison(pharma_duel_result: dict, max_rows: int = 12) -> str:
    if not pharma_duel_result:
        return '<p style="color:#5a6380;text-align:center;padding:20px">No candidates to compare.</p>'

    rows = []
    for status, items in (
        ("Survived", pharma_duel_result.get("survivors", [])),
        ("Eliminated", pharma_duel_result.get("eliminated", [])),
    ):
        for item in items:
            rounds = item.get("battle_log", [])
            final_defense = rounds[-1].get("defense", {}) if rounds else {}
            rows.append(
                {
                    "status": status,
                    "smiles": item.get("final_smiles") or item.get("initial_smiles") or "",
                    "binding": item.get("best_binding_score"),
                    "rounds": item.get("rounds_survived", 0),
                    "tox": final_defense.get("toxicity", {}).get("passed"),
                    "res": final_defense.get("resistance", {}).get("passed"),
                    "syn": final_defense.get("synthesis", {}).get("passed"),
                    "reason": "; ".join(item.get("elimination_reason") or []),
                }
            )

    if not rows:
        return '<p style="color:#5a6380;text-align:center;padding:20px">No candidates to compare.</p>'

    html = """
<div style="
  background:linear-gradient(180deg, #0d1120 0%, #0a0a14 100%);
  border-radius:14px; border:1px solid #1e2548;
  overflow:hidden; box-shadow:0 4px 24px rgba(0,0,0,0.3);
  overflow-x:auto;
">
<table style="width:100%;font-size:11px;border-collapse:collapse;color:#eaedf5;font-family:'Inter',sans-serif;min-width:700px">
  <tr style="background:#131830">
    <th style="text-align:left;padding:10px 12px;color:#5a6380;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:1px">Status</th>
    <th style="text-align:left;padding:10px 12px;color:#5a6380;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:1px">SMILES</th>
    <th style="text-align:center;padding:10px 8px;color:#5a6380;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:1px">Binding</th>
    <th style="text-align:center;padding:10px 8px;color:#5a6380;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:1px">Rounds</th>
    <th style="text-align:center;padding:10px 8px;color:#5a6380;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:1px">Tox</th>
    <th style="text-align:center;padding:10px 8px;color:#5a6380;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:1px">Res</th>
    <th style="text-align:center;padding:10px 8px;color:#5a6380;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:1px">Syn</th>
    <th style="text-align:left;padding:10px 12px;color:#5a6380;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:1px">Reason</th>
  </tr>
"""
    for i, row in enumerate(rows[:max_rows]):
        is_survived = row["status"] == "Survived"
        status_color = "#34d399" if is_survived else "#f43f5e"
        status_bg = "#34d39910" if is_survived else "#f43f5e10"
        binding = row["binding"]
        binding_text = f"{binding:.2f}" if isinstance(binding, (int, float)) else "N/A"
        bg = "transparent" if i % 2 == 0 else "rgba(19,24,48,0.3)"

        def _check_badge(val):
            if val is True:
                return '<span style="background:#34d39920;color:#34d399;padding:2px 6px;border-radius:4px;font-size:9px;font-weight:600">PASS</span>'
            elif val is False:
                return '<span style="background:#f43f5e20;color:#f43f5e;padding:2px 6px;border-radius:4px;font-size:9px;font-weight:600">FAIL</span>'
            return '<span style="color:#3a4060;font-size:9px">N/A</span>'

        html += f"""
  <tr style="border-bottom:1px solid #1e2548;background:{bg}">
    <td style="padding:8px 12px">
      <span style="background:{status_bg};color:{status_color};padding:3px 8px;border-radius:20px;font-size:10px;font-weight:600">{row['status']}</span>
    </td>
    <td style="padding:8px 12px;word-break:break-all;font-family:'JetBrains Mono',monospace;font-size:10px;color:#8a92b0;max-width:250px">{row['smiles'][:72]}</td>
    <td style="padding:8px;text-align:center;color:#7c6ff7;font-weight:600;font-family:'JetBrains Mono',monospace">{binding_text}</td>
    <td style="padding:8px;text-align:center;color:#4d8fff;font-weight:500">{row['rounds']}</td>
    <td style="padding:8px;text-align:center">{_check_badge(row['tox'])}</td>
    <td style="padding:8px;text-align:center">{_check_badge(row['res'])}</td>
    <td style="padding:8px;text-align:center">{_check_badge(row['syn'])}</td>
    <td style="padding:8px 12px;color:#5a6380;font-size:10px;max-width:200px">{row['reason'][:100]}</td>
  </tr>
"""
    return html + "</table></div>"
