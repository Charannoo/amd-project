# ui/components/molecule_viewer.py
"""3Dmol.js viewer for Gradio HTML — premium glassmorphic design."""


def smiles_to_sdf(smiles: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)
        return Chem.MolToMolBlock(mol)
    except Exception:
        return ""


def create_3d_viewer_html(smiles: str, width: str = "100%", height: int = 380) -> str:
    if not smiles:
        return """
<div style="
  background:linear-gradient(180deg, #0d1120 0%, #0a0a14 100%);
  border-radius:14px; border:1px solid #1e2548;
  padding:60px 20px; text-align:center;
">
  <div style="font-size:32px;margin-bottom:12px">🧪</div>
  <div style="color:#5a6380;font-size:13px">No molecule to display</div>
  <div style="color:#3a4060;font-size:11px;margin-top:4px">Run a pipeline or load a SMILES</div>
</div>"""

    sdf_content = smiles_to_sdf(smiles)
    if not sdf_content:
        return f"""
<div style="
  background:linear-gradient(180deg, #0d1120 0%, #0a0a14 100%);
  border-radius:14px; border:1px solid #1e2548;
  padding:30px 20px; text-align:center;
">
  <div style="font-size:24px;margin-bottom:8px">⚠️</div>
  <div style="color:#fbbf24;font-size:12px">Could not generate 3D conformation</div>
  <div style="color:#5a6380;font-size:10px;margin-top:6px;word-break:break-all;
    font-family:'JetBrains Mono',monospace">{smiles[:80]}</div>
</div>"""

    sdf_escaped = sdf_content.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    vid = abs(hash(smiles)) % 99999

    display_smiles = smiles[:70] + ('...' if len(smiles) > 70 else '')

    return f"""
<div style="
  background:linear-gradient(180deg, #0d1120 0%, #0a0a14 100%);
  border-radius:14px;
  border:1px solid #1e2548;
  overflow:hidden;
  box-shadow:0 4px 24px rgba(0,0,0,0.3);
">
  <!-- Toolbar -->
  <div style="
    padding:10px 16px;
    background:linear-gradient(90deg, #131830 0%, #0d1120 100%);
    display:flex; justify-content:space-between; align-items:center;
    border-bottom:1px solid #1e2548;
  ">
    <div style="display:flex;align-items:center;gap:8px">
      <div style="
        width:8px;height:8px;border-radius:50%;
        background:#7c6ff7;
        box-shadow:0 0 6px rgba(124,111,247,0.4);
      "></div>
      <span style="
        color:#8a92b0;font-size:10px;
        font-family:'JetBrains Mono',monospace;
        max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
      ">{display_smiles}</span>
    </div>
    <div style="display:flex;gap:4px">
      <button type="button"
        onclick="window.__v{vid}.setStyle({{}},{{stick:{{colorscheme:'Jmol',radius:0.15}},sphere:{{colorscheme:'Jmol',radius:0.25}}}}); window.__v{vid}.render()"
        style="
          font-size:10px;padding:4px 10px;
          background:#1e2548;color:#8a92b0;
          border:1px solid #2a3060;border-radius:6px;cursor:pointer;
          font-family:'Inter',sans-serif;font-weight:500;
          transition:all 150ms ease;
        "
        onmouseover="this.style.background='#2a3060';this.style.color='#eaedf5'"
        onmouseout="this.style.background='#1e2548';this.style.color='#8a92b0'"
      >Ball & Stick</button>
      <button type="button"
        onclick="window.__v{vid}.setStyle({{}},{{sphere:{{colorscheme:'Jmol',radius:0.6}}}}); window.__v{vid}.render()"
        style="
          font-size:10px;padding:4px 10px;
          background:#1e2548;color:#8a92b0;
          border:1px solid #2a3060;border-radius:6px;cursor:pointer;
          font-family:'Inter',sans-serif;font-weight:500;
          transition:all 150ms ease;
        "
        onmouseover="this.style.background='#2a3060';this.style.color='#eaedf5'"
        onmouseout="this.style.background='#1e2548';this.style.color='#8a92b0'"
      >Space Fill</button>
      <button type="button"
        onclick="window.__v{vid}.setStyle({{}},{{stick:{{colorscheme:'Jmol',radius:0.12}}}}); window.__v{vid}.render()"
        style="
          font-size:10px;padding:4px 10px;
          background:#1e2548;color:#8a92b0;
          border:1px solid #2a3060;border-radius:6px;cursor:pointer;
          font-family:'Inter',sans-serif;font-weight:500;
          transition:all 150ms ease;
        "
        onmouseover="this.style.background='#2a3060';this.style.color='#eaedf5'"
        onmouseout="this.style.background='#1e2548';this.style.color='#8a92b0'"
      >Wireframe</button>
      <button type="button"
        onclick="window.__v{vid}.spin(window.__v{vid}._spinning ? false : 'y'); window.__v{vid}._spinning = !window.__v{vid}._spinning;"
        style="
          font-size:10px;padding:4px 10px;
          background:#534AB720;color:#7c6ff7;
          border:1px solid #534AB750;border-radius:6px;cursor:pointer;
          font-family:'Inter',sans-serif;font-weight:500;
          transition:all 150ms ease;
        "
        onmouseover="this.style.background='#534AB740'"
        onmouseout="this.style.background='#534AB720'"
      >🔄 Spin</button>
    </div>
  </div>
  <!-- Viewer -->
  <div id="viewer-{vid}" style="width:{width};height:{height}px;background:#0a0a14"></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
<script>
(function() {{
  const container = document.getElementById('viewer-{vid}');
  const viewer = $3Dmol.createViewer(container, {{
    backgroundColor: '0x0a0a14',
    antialias: true
  }});
  window.__v{vid} = viewer;
  viewer._spinning = false;
  const sdfData = `{sdf_escaped}`;
  viewer.addModel(sdfData, 'sdf');
  viewer.setStyle({{}}, {{
    stick: {{colorscheme: 'Jmol', radius: 0.15}},
    sphere: {{colorscheme: 'Jmol', radius: 0.25}}
  }});
  viewer.zoomTo();
  viewer.render();
}})();
</script>
"""
