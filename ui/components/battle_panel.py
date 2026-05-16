# ui/components/battle_panel.py
"""PharmaDuel battle log — premium visualization."""


def render_battle_panel(pharma_duel_result: dict) -> str:
    if not pharma_duel_result:
        return ""

    survivors = pharma_duel_result.get("survivors", [])
    eliminated = pharma_duel_result.get("eliminated", [])
    battle_results = pharma_duel_result.get("battle_results", [])
    total = len(battle_results)
    survival_rate = pharma_duel_result.get("survival_rate", 0)

    # Header stats
    rate_color = "#34d399" if survival_rate >= 0.5 else "#fbbf24" if survival_rate >= 0.3 else "#f43f5e"

    html = f"""
<div style="
  background:linear-gradient(180deg, #0d1120 0%, #0a0a14 100%);
  border-radius:14px;
  border:1px solid #1e2548;
  padding:20px;
  font-family:'Inter',sans-serif;
  box-shadow:0 4px 24px rgba(0,0,0,0.3);
">
  <!-- Header -->
  <div style="
    display:flex; align-items:center; gap:14px;
    margin-bottom:16px; padding-bottom:14px;
    border-bottom:1px solid #1e2548;
  ">
    <div style="
      width:40px; height:40px; border-radius:10px;
      background:linear-gradient(135deg, #f43f5e20, #7c6ff720);
      display:flex; align-items:center; justify-content:center;
      font-size:20px;
    ">⚔️</div>
    <div style="flex:1">
      <div style="color:#eaedf5;font-weight:600;font-size:14px;letter-spacing:0.3px">PharmaDuel Battle Log</div>
      <div style="color:#5a6380;font-size:11px;margin-top:2px">Adversarial multi-agent drug refinement</div>
    </div>
    <div style="display:flex;gap:16px;align-items:center">
      <div style="text-align:center">
        <div style="color:#34d399;font-size:20px;font-weight:600">{len(survivors)}</div>
        <div style="color:#5a6380;font-size:9px;text-transform:uppercase;letter-spacing:1px">Survived</div>
      </div>
      <div style="width:1px;height:30px;background:#1e2548"></div>
      <div style="text-align:center">
        <div style="color:#f43f5e;font-size:20px;font-weight:600">{len(eliminated)}</div>
        <div style="color:#5a6380;font-size:9px;text-transform:uppercase;letter-spacing:1px">Eliminated</div>
      </div>
      <div style="width:1px;height:30px;background:#1e2548"></div>
      <div style="text-align:center">
        <div style="color:{rate_color};font-size:20px;font-weight:600">{survival_rate*100:.0f}%</div>
        <div style="color:#5a6380;font-size:9px;text-transform:uppercase;letter-spacing:1px">Rate</div>
      </div>
    </div>
  </div>
"""

    for i, battle in enumerate(battle_results):
        survived = battle.get("survived", False)
        bs = battle.get("best_binding_score")
        bs_str = f"{bs:.2f}" if isinstance(bs, (int, float)) else "N/A"
        smi = battle.get("final_smiles") or battle.get("initial_smiles") or ""

        # Card colors
        if survived:
            border_color = "#1a3a2a"
            bg_color = "rgba(52,211,153,0.03)"
            status_badge = '<span style="background:#34d39920;color:#34d399;padding:2px 8px;border-radius:20px;font-size:10px;font-weight:600">✅ SURVIVED</span>'
        else:
            border_color = "#3a1a2a"
            bg_color = "rgba(244,63,94,0.03)"
            status_badge = '<span style="background:#f43f5e20;color:#f43f5e;padding:2px 8px;border-radius:20px;font-size:10px;font-weight:600">❌ ELIMINATED</span>'

        html += f"""
  <!-- Candidate {i+1} -->
  <div style="
    border:1px solid {border_color};
    border-radius:10px;
    padding:14px;
    margin-bottom:10px;
    background:{bg_color};
    transition:all 150ms ease;
  ">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
      <div style="display:flex;align-items:center;gap:10px">
        <span style="
          color:#5a6380;font-size:10px;font-weight:600;
          background:#131830;padding:3px 8px;border-radius:6px;
        ">#{i+1}</span>
        {status_badge}
      </div>
      <div style="display:flex;align-items:center;gap:6px">
        <span style="color:#5a6380;font-size:10px">Binding</span>
        <span style="
          color:{'#34d399' if survived else '#8a92b0'};
          font-size:13px;font-weight:600;
          font-family:'JetBrains Mono',monospace;
        ">{bs_str}</span>
      </div>
    </div>
    <div style="
      color:#5a6380;font-size:10px;word-break:break-all;
      font-family:'JetBrains Mono',monospace;
      background:#0a0a14;padding:6px 10px;border-radius:6px;
      margin-bottom:10px;
    ">{(smi or '')[:90]}{'...' if len(smi)>90 else ''}</div>
"""

        # Round details
        for round_data in battle.get("battle_log", []):
            round_num = round_data["round"]
            defense = round_data.get("defense", {})
            passed = defense.get("passed", 0)
            total_def = defense.get("total", 3)
            objections = defense.get("objections", [])
            att = round_data.get("binding_score")
            att_str = f"{att:.1f}" if isinstance(att, (int, float)) else "?"
            round_survived = round_data.get("survived", False)

            # Defense dots
            dots = ""
            for agent_name, agent_key in [("TOX", "toxicity"), ("RES", "resistance"), ("SYN", "synthesis")]:
                agent_passed = defense.get(agent_key, {}).get("passed", True)
                dot_color = "#34d399" if agent_passed else "#f43f5e"
                dots += f'<span style="display:inline-flex;align-items:center;gap:3px;margin-right:10px"><span style="width:6px;height:6px;border-radius:50%;background:{dot_color};display:inline-block"></span><span style="color:#5a6380;font-size:9px">{agent_name}</span></span>'

            html += f"""
    <div style="
      margin-top:6px;padding:8px 10px;
      background:#0a0a14;border-radius:8px;
      border-left:2px solid {'#34d399' if round_survived else '#f43f5e'};
      font-size:10px;
    ">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px">
        <div style="display:flex;align-items:center;gap:10px">
          <span style="color:#7c6ff7;font-weight:600">R{round_num}</span>
          <span style="color:#4d8fff">⚡ {att_str}</span>
          <span style="color:{'#34d399' if passed == total_def else '#f43f5e'};font-weight:500">{passed}/{total_def}</span>
        </div>
        <div>{dots}</div>
      </div>"""

            for obj in objections:
                issue = (obj.get("issue") or "")[:100]
                severity = obj.get("severity", "medium")
                sev_color = "#f43f5e" if severity == "high" else "#fbbf24"
                html += f'<div style="color:{sev_color};margin-top:4px;padding-left:16px;font-size:10px">⚠ {issue}</div>'
            html += "</div>"

        html += "</div>"

    html += "</div>"
    return html
