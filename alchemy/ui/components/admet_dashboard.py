# ui/components/admet_dashboard.py
"""ADMET property dashboard — premium radial + bar visualization."""


def _ring_svg(value: float, max_val: float, label: str, color: str, size: int = 56) -> str:
    """Small SVG ring gauge for a single property."""
    pct = min(100, (value / max_val * 100)) if max_val else 0
    r = (size - 8) / 2
    circ = 2 * 3.14159 * r
    offset = circ * (1 - pct / 100)

    return f"""
<div style="text-align:center;min-width:{size+8}px">
  <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
    <circle cx="{size/2}" cy="{size/2}" r="{r}"
      fill="none" stroke="#131830" stroke-width="4"/>
    <circle cx="{size/2}" cy="{size/2}" r="{r}"
      fill="none" stroke="{color}" stroke-width="4"
      stroke-dasharray="{circ}" stroke-dashoffset="{offset}"
      stroke-linecap="round"
      transform="rotate(-90 {size/2} {size/2})"
      style="transition:stroke-dashoffset 0.6s ease"/>
  </svg>
  <div style="color:#eaedf5;font-size:10px;font-weight:600;margin-top:2px;font-family:'JetBrains Mono',monospace">
    {value}
  </div>
  <div style="color:#5a6380;font-size:8px;text-transform:uppercase;letter-spacing:0.5px">{label}</div>
</div>"""


def render_admet_dashboard(admet_result: dict) -> str:
    if not admet_result or "error" in admet_result:
        return ""

    props = admet_result.get("properties", {})
    pass_fail = admet_result.get("pass_fail", {})
    lipinski = admet_result.get("lipinski", {})
    score = float(admet_result.get("admet_score", 0) or 0)

    # Score tier
    if score >= 75:
        score_color = "#34d399"
        score_label = "Excellent"
        score_glow = "rgba(52,211,153,0.15)"
    elif score >= 55:
        score_color = "#fbbf24"
        score_label = "Moderate"
        score_glow = "rgba(251,191,36,0.15)"
    else:
        score_color = "#f43f5e"
        score_label = "Poor"
        score_glow = "rgba(244,63,94,0.15)"

    # Ring gauges for key metrics
    rings_html = ""
    ring_metrics = [
        ("molecular_weight", "MW", 500, "#4d8fff"),
        ("logp", "LogP", 5, "#7c6ff7"),
        ("qed", "QED", 1, "#34d399"),
        ("sa_score", "SA", 6, "#fbbf24"),
    ]
    for key, label, max_v, color in ring_metrics:
        val = props.get(key)
        if val is not None:
            rings_html += _ring_svg(val, max_v, label, color)

    # Detailed property bars
    properties_html = ""
    prop_display = [
        ("molecular_weight", "Molecular Weight", "Da", 500),
        ("logp", "LogP", "", 5),
        ("hbd", "H-Bond Donors", "", 5),
        ("hba", "H-Bond Acceptors", "", 10),
        ("tpsa", "Polar Surface Area", "Å²", 140),
        ("qed", "Drug-likeness (QED)", "", 1),
        ("sa_score", "Synthetic Accessibility", "", 6),
        ("rotatable_bonds", "Rotatable Bonds", "", 10),
    ]

    for key, label, unit, threshold in prop_display:
        val = props.get(key)
        if val is None:
            continue
        is_pass = pass_fail.get(f"{key}_pass", True)
        try:
            pct = min(100, (float(val) / float(threshold) * 100)) if threshold else 50
        except Exception:
            pct = 50
        bar_color = "#34d399" if is_pass else "#f43f5e"
        icon = "✓" if is_pass else "✗"

        properties_html += f"""
<div style="margin-bottom:10px">
  <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:4px">
    <span style="color:#8a92b0;font-weight:400">{label}</span>
    <span style="color:{bar_color};font-weight:500;font-family:'JetBrains Mono',monospace">{val}{unit} {icon}</span>
  </div>
  <div style="height:4px;background:#131830;border-radius:3px;overflow:hidden">
    <div style="
      height:100%;width:{pct}%;
      background:linear-gradient(90deg, {bar_color}cc, {bar_color});
      border-radius:3px;
      transition:width 0.5s ease;
    "></div>
  </div>
</div>"""

    # Lipinski badge
    lipinski_pass = lipinski.get("lipinski_pass", False)
    viol = lipinski.get("violations", [])
    if lipinski_pass:
        lipinski_html = '<span style="background:#34d39915;color:#34d399;padding:4px 12px;border-radius:20px;font-size:11px;font-weight:500">✅ Lipinski Ro5 PASS</span>'
    else:
        viol_txt = ", ".join(viol) if viol else "Unknown"
        lipinski_html = f'<span style="background:#f43f5e15;color:#f43f5e;padding:4px 12px;border-radius:20px;font-size:11px;font-weight:500">❌ Lipinski FAIL — {viol_txt}</span>'

    formula = props.get("formula", "")
    heavy = props.get("heavy_atoms", "")

    return f"""
<div style="
  background:linear-gradient(180deg, #0d1120 0%, #0a0a14 100%);
  border-radius:14px;
  border:1px solid #1e2548;
  padding:20px;
  font-family:'Inter',sans-serif;
  box-shadow:0 4px 24px rgba(0,0,0,0.3);
">
  <!-- Header + Score -->
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:18px">
    <div>
      <div style="color:#eaedf5;font-size:14px;font-weight:600;letter-spacing:0.3px">ADMET Profile</div>
      <div style="color:#5a6380;font-size:11px;margin-top:3px">
        {formula} · {heavy} heavy atoms
      </div>
    </div>
    <div style="
      text-align:center;
      background:{score_glow};
      padding:10px 18px;
      border-radius:12px;
      border:1px solid {score_color}30;
    ">
      <div style="font-size:28px;font-weight:300;color:{score_color};line-height:1;font-family:'JetBrains Mono',monospace">
        {score:.0f}<span style="font-size:14px">%</span>
      </div>
      <div style="font-size:9px;color:{score_color};text-transform:uppercase;letter-spacing:1px;margin-top:2px">{score_label}</div>
    </div>
  </div>

  <!-- Ring Gauges -->
  <div style="
    display:flex; justify-content:space-around; align-items:center;
    background:#0a0a14; border-radius:10px; padding:14px 8px;
    margin-bottom:18px; border:1px solid #131830;
  ">
    {rings_html}
  </div>

  <!-- Lipinski -->
  <div style="margin-bottom:16px">{lipinski_html}</div>

  <!-- Property Bars -->
  <div style="
    background:#0a0a14;border-radius:10px;padding:14px;
    border:1px solid #131830;
  ">
    {properties_html}
  </div>
</div>"""
