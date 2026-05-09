# ui/components/agent_ticker.py
"""Live agent status ticker with animated pipeline visualization."""


def render_agent_ticker(messages: list[str]) -> str:
    if not messages:
        return ""

    items_html = ""
    for i, msg in enumerate(messages[-15:]):
        opacity = max(0.4, 1 - (len(messages[-15:]) - 1 - i) * 0.06)
        # Pick icon based on keyword
        icon = "⏳"
        color = "#8a92b0"
        if "complete" in msg.lower() or "done" in msg.lower():
            icon = "✅"
            color = "#34d399"
        elif "searching" in msg.lower() or "fetching" in msg.lower():
            icon = "🔍"
            color = "#4d8fff"
        elif "generating" in msg.lower() or "computing" in msg.lower():
            icon = "⚡"
            color = "#fbbf24"
        elif "battle" in msg.lower() or "duel" in msg.lower():
            icon = "⚔️"
            color = "#f43f5e"
        elif "target" in msg.lower() or "protein" in msg.lower():
            icon = "🧬"
            color = "#7c6ff7"
        elif "literature" in msg.lower() or "pubmed" in msg.lower():
            icon = "📚"
            color = "#00d4aa"
        elif "report" in msg.lower() or "hypothesis" in msg.lower():
            icon = "📝"
            color = "#a78bfa"

        items_html += f"""
<div style="
  display:flex; align-items:center; gap:10px;
  padding:7px 12px; margin-bottom:3px;
  background:rgba(13,17,32,{0.6 + i*0.02});
  border-radius:8px; opacity:{opacity};
  border-left:2px solid {color};
  animation:fadeIn 0.3s ease-out;
">
  <span style="font-size:13px;min-width:20px;text-align:center">{icon}</span>
  <span style="color:{color};font-size:11px;font-family:'Inter',sans-serif;font-weight:400">{msg}</span>
  <span style="margin-left:auto;color:#3a4060;font-size:9px">#{i+1}</span>
</div>"""

    progress_pct = min(100, (len(messages) / 8) * 100)

    return f"""
<div style="
  background:linear-gradient(180deg, #0d1120 0%, #0a0a14 100%);
  border-radius:12px;
  border:1px solid #1e2548;
  padding:16px;
  max-height:280px;
  overflow-y:auto;
  box-shadow:0 4px 24px rgba(0,0,0,0.3);
">
  <div style="
    display:flex; align-items:center; justify-content:space-between;
    margin-bottom:12px; padding-bottom:10px;
    border-bottom:1px solid #1e2548;
  ">
    <div style="display:flex;align-items:center;gap:8px">
      <div style="
        width:8px; height:8px; border-radius:50%;
        background:#34d399;
        box-shadow:0 0 8px rgba(52,211,153,0.5);
        animation:pulseGlow 2s ease-in-out infinite;
      "></div>
      <span style="color:#eaedf5;font-size:12px;font-weight:600;letter-spacing:0.5px;text-transform:uppercase">
        Agent Pipeline
      </span>
    </div>
    <span style="color:#5a6380;font-size:10px">{len(messages)} steps</span>
  </div>
  <div style="
    height:3px; background:#131830; border-radius:2px;
    margin-bottom:12px; overflow:hidden;
  ">
    <div style="
      height:100%; width:{progress_pct}%;
      background:linear-gradient(90deg, #534AB7, #7c6ff7, #4d8fff);
      border-radius:2px;
      transition:width 0.5s ease;
    "></div>
  </div>
  {items_html}
</div>
<style>
@keyframes fadeIn {{
  from {{ opacity:0; transform:translateX(-8px); }}
  to {{ opacity:1; transform:translateX(0); }}
}}
@keyframes pulseGlow {{
  0%, 100% {{ box-shadow:0 0 8px rgba(52,211,153,0.3); }}
  50% {{ box-shadow:0 0 14px rgba(52,211,153,0.6); }}
}}
</style>"""
