# ui/app.py
"""
ALCHEMY — Main Gradio Application (Premium UI)
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gradio as gr
import torch

from models.esm2_loader import load_esm2
from modes.disease_mode import DiseaseMode
from modes.molecule_chat_mode import MoleculeChatMode
from modes.pandemic_mode import PandemicMode
from modes.repurposing_mode import RepurposingMode
from modes.superbug_mode import SuperbugMode
from ui.export_utils import write_result_exports
from ui.components.admet_dashboard import render_admet_dashboard
from ui.components.molecule_viewer import create_3d_viewer_html
from ui.components.battle_panel import render_battle_panel
from ui.components.results_table import render_candidate_comparison
from ui.components.agent_ticker import render_agent_ticker

ESM2_MODEL = None
ESM2_ALPHABET = None
disease_mode = None
pandemic_mode = None
superbug_mode = None
repurposing_mode = None


def get_esm2():
    global ESM2_MODEL, ESM2_ALPHABET
    if ESM2_MODEL is None or ESM2_ALPHABET is None:
        print("Initializing ALCHEMY ESM2...")
        ESM2_MODEL, ESM2_ALPHABET = load_esm2()
        if torch.cuda.is_available():
            print(f"GPU Memory after ESM2 load: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return ESM2_MODEL, ESM2_ALPHABET


def get_disease_mode():
    global disease_mode
    if disease_mode is None:
        model, alphabet = get_esm2()
        disease_mode = DiseaseMode(model, alphabet)
    return disease_mode


def get_pandemic_mode():
    global pandemic_mode
    if pandemic_mode is None:
        model, alphabet = get_esm2()
        pandemic_mode = PandemicMode(model, alphabet)
    return pandemic_mode


def get_superbug_mode():
    global superbug_mode
    if superbug_mode is None:
        model, alphabet = get_esm2()
        superbug_mode = SuperbugMode(model, alphabet)
    return superbug_mode


def get_repurposing_mode():
    global repurposing_mode
    if repurposing_mode is None:
        model, alphabet = get_esm2()
        repurposing_mode = RepurposingMode(model, alphabet)
    return repurposing_mode


def top_survivor_smiles(survivors: list[dict]) -> str:
    survivor = top_survivor(survivors)
    if not survivor:
        return ""
    return survivor.get("final_smiles") or survivor.get("initial_smiles") or ""


def top_survivor(survivors: list[dict]) -> dict:
    for survivor in survivors:
        smiles = survivor.get("final_smiles") or survivor.get("initial_smiles")
        if smiles:
            return survivor
    return {}


def demo_index_warning(result: dict) -> str:
    candidates = result.get("pipeline_steps", {}).get("repurposing", {}).get("candidates", [])
    if any(c.get("demo_index") for c in candidates):
        return " | Demo index: repurposing uses deterministic fallback embeddings"
    return ""


def _make_progress_cb(log: list, gr_progress=None):
    """Return a callback that logs messages and updates gr.Progress."""
    def cb(message: str, pct: float):
        log.append(message)
        if gr_progress is not None:
            gr_progress(pct / 100, desc=message)
    return cb


# ---------------------------------------------------------------------------
# Disease Mode
# ---------------------------------------------------------------------------

def run_disease_mode(disease_name: str, progress=gr.Progress()):
    log = []
    cb = _make_progress_cb(log, progress)
    result = get_disease_mode().run(disease_name, progress_callback=cb)
    ticker = render_agent_ticker(log)

    if result.get("error"):
        msg = result["error"]
        return msg, f"<p>{msg}</p>", "", "", "", "No report.", [], ticker

    survivors = result.get("survivors", [])
    pharma_duel = result.get("pipeline_steps", {}).get("pharma_duel", {})
    battle_html = render_battle_panel(pharma_duel)
    comparison_html = render_candidate_comparison(pharma_duel)

    displayed_survivor = top_survivor(survivors)
    top_smiles = top_survivor_smiles(survivors)
    if top_smiles:
        mol_html = create_3d_viewer_html(top_smiles)
        admet_html = render_admet_dashboard(displayed_survivor.get("admet", {}))
    else:
        mol_html = create_3d_viewer_html("")
        admet_html = ""

    report = result.get("report", "No report generated")
    summary = (
        f"\u2705 {len(survivors)} battle-hardened candidates | "
        f"\U0001f9ec Target: {result.get('primary_target', 'Unknown')}"
        f"{demo_index_warning(result)}"
    )
    downloads = write_result_exports(result, f"disease-{disease_name}")
    return summary, battle_html, comparison_html, mol_html, admet_html, report, downloads, ticker


# ---------------------------------------------------------------------------
# Molecule Chat Mode
# ---------------------------------------------------------------------------

def load_molecule_for_chat(smiles_input: str, molecule_chat_state):
    molecule_chat_state = molecule_chat_state or MoleculeChatMode()
    result = molecule_chat_state.load_molecule(smiles_input)
    if "error" in result:
        return result["error"], "", "", molecule_chat_state
    mol_html = create_3d_viewer_html(result["smiles"])
    admet_html = render_admet_dashboard(result["admet"])
    return f"Loaded: {result['smiles'][:50]}...", mol_html, admet_html, molecule_chat_state


def chat_with_molecule(user_message: str, chat_history: list, molecule_chat_state):
    molecule_chat_state = molecule_chat_state or MoleculeChatMode()
    if chat_history is None:
        chat_history = []
    result = molecule_chat_state.chat(user_message)
    if "error" in result:
        chat_history.append([user_message, result["error"]])
        return chat_history, "", "", molecule_chat_state

    response_text = (
        f"**{result['interpretation']}**\n\n{result['explanation']}\n\n"
        f"ADMET Score: {result['admet'].get('admet_score', 'N/A')}%"
    )
    if result.get("improved"):
        response_text += " \u2705 Improved!"

    chat_history.append([user_message, response_text])
    mol_html = create_3d_viewer_html(result["modified_smiles"])
    admet_html = render_admet_dashboard(result["admet"])
    return chat_history, mol_html, admet_html, molecule_chat_state


# ---------------------------------------------------------------------------
# Pandemic Mode
# ---------------------------------------------------------------------------

def run_pandemic_mode(genome_sequence: str, progress=gr.Progress()):
    log = []
    cb = _make_progress_cb(log, progress)
    result = get_pandemic_mode().run(genome_sequence, progress_callback=cb)
    ticker = render_agent_ticker(log)

    if result.get("error"):
        e = result["error"]
        return e, f"<p>{e}</p>", "", "", "", [], ticker

    survivors = result.get("survivors", [])
    pharma_duel = result.get("pharma_duel", {})
    battle_html = render_battle_panel(pharma_duel)
    comparison_html = render_candidate_comparison(pharma_duel)
    top_smiles = top_survivor_smiles(survivors)
    mol_html = create_3d_viewer_html(top_smiles) if top_smiles else ""
    report = result.get("report", "")
    nprot = len(result.get("proteins_found", []))
    summary = f"\U0001f9ab {nprot} viral proteins identified | {len(survivors)} antiviral candidates"
    downloads = write_result_exports(result, "pandemic-mode")
    return summary, battle_html, comparison_html, mol_html, report, downloads, ticker


# ---------------------------------------------------------------------------
# Superbug Mode
# ---------------------------------------------------------------------------

def run_superbug_mode(bacteria: str, resistance_mechanism: str, progress=gr.Progress()):
    log = []
    cb = _make_progress_cb(log, progress)
    result = get_superbug_mode().run(bacteria, resistance_mechanism, progress_callback=cb)
    ticker = render_agent_ticker(log)

    if result.get("error"):
        e = result["error"]
        return e, f"<p>{e}</p>", "", "", "", [], ticker

    survivors = result.get("survivors", [])
    pharma_duel = result.get("pharma_duel", {})
    battle_html = render_battle_panel(pharma_duel)
    comparison_html = render_candidate_comparison(pharma_duel)
    top_smiles = top_survivor_smiles(survivors)
    mol_html = create_3d_viewer_html(top_smiles) if top_smiles else ""
    report = result.get("report", "")
    summary = f"\U0001f9ab {len(survivors)} novel antibiotic scaffolds surviving resistance testing"
    downloads = write_result_exports(result, f"superbug-{bacteria}")
    return summary, battle_html, comparison_html, mol_html, report, downloads, ticker


# ---------------------------------------------------------------------------
# Repurposing Mode
# ---------------------------------------------------------------------------

def run_repurposing_mode(query: str, query_type: str, progress=gr.Progress()):
    log = []
    cb = _make_progress_cb(log, progress)
    result = get_repurposing_mode().run(query, query_type, progress_callback=cb)
    ticker = render_agent_ticker(log)

    if result.get("error"):
        e = result["error"]
        return e, f"<p>{e}</p>", "", "", [], ticker

    survivors = result.get("survivors", [])
    pharma_duel = result.get("pharma_duel", {})
    battle_html = render_battle_panel(pharma_duel)
    comparison_html = render_candidate_comparison(pharma_duel)
    report = result.get("report", "")
    summary = f"\U0001f48a {len(survivors)} repurposing candidates identified"
    downloads = write_result_exports(result, f"repurposing-{query}")
    return summary, battle_html, comparison_html, report, downloads, ticker


# ---------------------------------------------------------------------------
# UI Layout — Premium Design
# ---------------------------------------------------------------------------

_CSS_PATH = Path(__file__).parent / "assets" / "custom.css"
_custom_css = ""
if _CSS_PATH.is_file():
    _custom_css = _CSS_PATH.read_text(encoding="utf-8")

HEADER_HTML = """
<div style="
  text-align:center;
  padding:32px 20px 24px;
  background:linear-gradient(180deg, rgba(124,111,247,0.06) 0%, transparent 100%);
  border-bottom:1px solid #1e2548;
  margin-bottom:24px;
  position:relative;
  overflow:hidden;
">
  <!-- Decorative orbs -->
  <div style="
    position:absolute; top:-40px; left:20%; width:200px; height:200px;
    background:radial-gradient(circle, rgba(124,111,247,0.08) 0%, transparent 70%);
    border-radius:50%; pointer-events:none;
  "></div>
  <div style="
    position:absolute; top:-20px; right:15%; width:160px; height:160px;
    background:radial-gradient(circle, rgba(0,212,170,0.06) 0%, transparent 70%);
    border-radius:50%; pointer-events:none;
  "></div>

  <!-- Logo -->
  <div style="
    display:inline-flex; align-items:center; gap:6px;
    margin-bottom:8px;
  ">
    <div style="
      font-size:36px; font-weight:200; color:#eaedf5;
      letter-spacing:12px;
      font-family:'Inter',sans-serif;
      text-shadow:0 0 40px rgba(124,111,247,0.3);
    ">ALCHEMY</div>
  </div>

  <!-- Tagline -->
  <div style="
    display:flex; align-items:center; justify-content:center; gap:8px;
    margin-bottom:10px;
  ">
    <div style="width:40px;height:1px;background:linear-gradient(90deg,transparent,#534AB7)"></div>
    <div style="
      color:#7c6ff7; font-size:10px;
      letter-spacing:4px; text-transform:uppercase;
      font-weight:500;
    ">Agent-Based Ligand &amp; Chemistry Engine</div>
    <div style="width:40px;height:1px;background:linear-gradient(90deg,#534AB7,transparent)"></div>
  </div>

  <!-- Tech stack pills -->
  <div style="display:flex;justify-content:center;gap:6px;flex-wrap:wrap">
    <span style="
      background:#131830;color:#8a92b0;
      padding:3px 10px;border-radius:20px;
      font-size:10px;border:1px solid #1e2548;
      font-family:'JetBrains Mono',monospace;
    ">AMD MI300X</span>
    <span style="
      background:#131830;color:#8a92b0;
      padding:3px 10px;border-radius:20px;
      font-size:10px;border:1px solid #1e2548;
      font-family:'JetBrains Mono',monospace;
    ">ROCm 6.1</span>
    <span style="
      background:#131830;color:#8a92b0;
      padding:3px 10px;border-radius:20px;
      font-size:10px;border:1px solid #1e2548;
      font-family:'JetBrains Mono',monospace;
    ">Qwen2.5-72B</span>
    <span style="
      background:#131830;color:#8a92b0;
      padding:3px 10px;border-radius:20px;
      font-size:10px;border:1px solid #1e2548;
      font-family:'JetBrains Mono',monospace;
    ">ESM2-3B</span>
    <span style="
      background:#131830;color:#8a92b0;
      padding:3px 10px;border-radius:20px;
      font-size:10px;border:1px solid #1e2548;
      font-family:'JetBrains Mono',monospace;
    ">MolT5</span>
    <span style="
      background:#131830;color:#8a92b0;
      padding:3px 10px;border-radius:20px;
      font-size:10px;border:1px solid #1e2548;
      font-family:'JetBrains Mono',monospace;
    ">PharmaDuel</span>
  </div>
</div>
"""

EXAMPLE_DISEASES = [
    ["Chagas disease"],
    ["Drug-resistant tuberculosis"],
    ["Leishmaniasis"],
    ["Sleeping sickness"],
    ["Alzheimer's disease"],
]

EXAMPLE_MOLECULES = [
    ["CC(=O)Oc1ccccc1C(=O)O"],
    ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],
    ["CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C"],
]

with gr.Blocks(
    css=_custom_css,
    title="ALCHEMY — Drug Discovery OS",
    theme=gr.themes.Base(),
) as demo:
    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # ---- Disease Mode ----
        with gr.Tab("\U0001f9ec Disease Mode"):
            gr.Markdown("### Enter any disease name → get battle-hardened drug candidates")
            with gr.Row():
                disease_input = gr.Textbox(
                    label="Disease Name",
                    placeholder="e.g. Chagas disease, Leishmaniasis, Alzheimer's...",
                    scale=4,
                )
                disease_btn = gr.Button("\u2697\ufe0f Run ALCHEMY Pipeline", variant="primary", scale=1)
            gr.Examples(EXAMPLE_DISEASES, inputs=disease_input)
            disease_summary = gr.Textbox(label="Summary", interactive=False)
            disease_ticker = gr.HTML(label="Agent Pipeline")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    disease_battle = gr.HTML(label="PharmaDuel Battle Log")
                with gr.Column(scale=1):
                    disease_3d = gr.HTML(label="3D Molecule Viewer")

            disease_comparison = gr.HTML(label="Candidate Comparison Table")
            disease_admet = gr.HTML(label="ADMET Property Profile")
            disease_report = gr.Textbox(label="AI Hypothesis Report", lines=8, interactive=False)
            disease_downloads = gr.File(label="Download Results", file_count="multiple")
            disease_btn.click(
                run_disease_mode,
                [disease_input],
                [
                    disease_summary,
                    disease_battle,
                    disease_comparison,
                    disease_3d,
                    disease_admet,
                    disease_report,
                    disease_downloads,
                    disease_ticker,
                ],
            )

        # ---- Molecule Chat ----
        with gr.Tab("\U0001f4ac Molecule Chat"):
            gr.Markdown("### Load a molecule → chat to redesign it → watch ADMET update live")
            molecule_chat_state = gr.State(None)
            with gr.Row():
                mol_input = gr.Textbox(
                    label="SMILES or Drug Name",
                    placeholder="e.g. Aspirin or CC(=O)Oc1ccccc1C(=O)O",
                    scale=4,
                )
                mol_load_btn = gr.Button("\U0001f9ea Load Molecule", variant="primary", scale=1)
            gr.Examples(EXAMPLE_MOLECULES, inputs=mol_input)
            mol_status = gr.Textbox(label="Status", interactive=False)
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    mol_3d = gr.HTML(label="3D Structure")
                with gr.Column(scale=1):
                    mol_admet = gr.HTML(label="ADMET Dashboard")
            chatbot = gr.Chatbot(label="Chat with your molecule", height=320)
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="e.g. 'make it more water-soluble' or 'reduce toxicity'",
                    scale=5,
                    show_label=False,
                )
                chat_btn = gr.Button("Send \u2192", variant="primary", scale=1)
            mol_load_btn.click(
                load_molecule_for_chat,
                [mol_input, molecule_chat_state],
                [mol_status, mol_3d, mol_admet, molecule_chat_state],
            )
            chat_btn.click(
                chat_with_molecule,
                [chat_input, chatbot, molecule_chat_state],
                [chatbot, mol_3d, mol_admet, molecule_chat_state],
            )

        # ---- Pandemic Mode ----
        with gr.Tab("\U0001f9ab Pandemic Mode"):
            gr.Markdown("### Paste a viral genome → discover antiviral candidates")
            genome_input = gr.Textbox(
                label="Viral Genome (FASTA or raw nucleotide sequence)",
                lines=4,
                placeholder="ATGAAACCCGGG... or paste FASTA with header",
            )
            pandemic_btn = gr.Button("\U0001f52c Identify Antiviral Targets", variant="primary")
            pandemic_summary = gr.Textbox(label="Summary", interactive=False)
            pandemic_ticker = gr.HTML(label="Agent Pipeline")
            with gr.Row(equal_height=True):
                pandemic_battle = gr.HTML(label="PharmaDuel Battle")
                pandemic_3d = gr.HTML(label="Top Candidate 3D")
            pandemic_comparison = gr.HTML(label="Candidate Comparison")
            pandemic_report = gr.Textbox(label="Antiviral Strategy Report", lines=8, interactive=False)
            pandemic_downloads = gr.File(label="Download Results", file_count="multiple")
            pandemic_btn.click(
                run_pandemic_mode,
                [genome_input],
                [
                    pandemic_summary,
                    pandemic_battle,
                    pandemic_comparison,
                    pandemic_3d,
                    pandemic_report,
                    pandemic_downloads,
                    pandemic_ticker,
                ],
            )

        # ---- Superbug Mode ----
        with gr.Tab("\U0001f9ab Superbug Mode"):
            gr.Markdown("### Design novel antibiotics to overcome resistance mechanisms")
            with gr.Row():
                bacteria_input = gr.Textbox(
                    label="Bacteria",
                    placeholder="e.g. MRSA, Pseudomonas aeruginosa",
                    scale=2,
                )
                resistance_input = gr.Textbox(
                    label="Resistance Mechanism",
                    placeholder="e.g. beta-lactamase, efflux pump",
                    scale=2,
                )
            superbug_btn = gr.Button("\u2694\ufe0f Design Antibiotic", variant="primary")
            superbug_summary = gr.Textbox(label="Summary", interactive=False)
            superbug_ticker = gr.HTML(label="Agent Pipeline")
            with gr.Row(equal_height=True):
                superbug_battle = gr.HTML(label="PharmaDuel Battle")
                superbug_3d = gr.HTML(label="Top Scaffold 3D")
            superbug_comparison = gr.HTML(label="Candidate Comparison")
            superbug_report = gr.Textbox(label="Antibiotic Strategy Report", lines=8, interactive=False)
            superbug_downloads = gr.File(label="Download Results", file_count="multiple")
            superbug_btn.click(
                run_superbug_mode,
                [bacteria_input, resistance_input],
                [
                    superbug_summary,
                    superbug_battle,
                    superbug_comparison,
                    superbug_3d,
                    superbug_report,
                    superbug_downloads,
                    superbug_ticker,
                ],
            )

        # ---- Repurposing Mode ----
        with gr.Tab("\U0001f48a Repurposing Mode"):
            gr.Markdown("### Find new uses for existing drugs — or existing drugs for a disease")
            with gr.Row():
                repurpose_input = gr.Textbox(
                    label="Drug Name or Disease Name",
                    placeholder="e.g. Metformin or NASH",
                    scale=4,
                )
                repurpose_type = gr.Radio(
                    ["Disease \u2192 Drug", "Drug \u2192 Disease"],
                    value="Disease \u2192 Drug",
                    label="Search Direction",
                    scale=2,
                )
            repurpose_btn = gr.Button("\U0001f504 Search Repurposing Candidates", variant="primary")
            repurpose_summary = gr.Textbox(label="Summary", interactive=False)
            repurpose_ticker = gr.HTML(label="Agent Pipeline")
            repurpose_battle = gr.HTML(label="PharmaDuel Validation")
            repurpose_comparison = gr.HTML(label="Candidate Comparison")
            repurpose_report = gr.Textbox(label="Repurposing Report", lines=8, interactive=False)
            repurpose_downloads = gr.File(label="Download Results", file_count="multiple")
            repurpose_btn.click(
                run_repurposing_mode,
                [repurpose_input, repurpose_type],
                [
                    repurpose_summary,
                    repurpose_battle,
                    repurpose_comparison,
                    repurpose_report,
                    repurpose_downloads,
                    repurpose_ticker,
                ],
            )

    # Footer
    gr.HTML("""
    <div style="
      text-align:center; padding:20px; margin-top:24px;
      border-top:1px solid #1e2548;
      color:#3a4060; font-size:10px;
    ">
      <div style="margin-bottom:4px">
        ALCHEMY v1.0 · MIT License · Research / Hackathon Use Only — Not for Clinical Decisions
      </div>
      <div>
        Built for AMD Developer Hackathon · Powered by MI300X · ROCm 6.1
      </div>
    </div>
    """)

if __name__ == "__main__":
    from config.settings import GRADIO_PORT, GRADIO_SHARE

    demo.launch(server_port=GRADIO_PORT, share=GRADIO_SHARE)
