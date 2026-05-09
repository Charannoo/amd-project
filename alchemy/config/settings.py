# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

# AMD ROCm
ROCM_DEVICE = "cuda:0"  # ROCm exposes as cuda
HSA_GFX_VERSION = "9.4.2"

# Model names
QWEN_MODEL = "Qwen/Qwen2.5-72B-Instruct"
ESM2_MODEL = "facebook/esm2_t36_3B_UR50D"
MOLT5_MODEL = "laituan245/molt5-large"
BIOBERT_MODEL = "dmis-lab/biobert-v1.1"

# vLLM server
VLLM_HOST = os.getenv("VLLM_HOST", "http://localhost:8000")
VLLM_API_KEY = "token-alchemy"  # dummy key for vLLM

# FAISS
DRUG_INDEX_PATH = os.getenv("DRUG_INDEX_PATH", "data/drug_index.faiss")
DRUG_METADATA_PATH = os.getenv("DRUG_METADATA_PATH", "data/drug_metadata.json")
PUBMED_INDEX_PATH = os.getenv("PUBMED_INDEX_PATH", "data/pubmed_index.faiss")
PUBMED_METADATA_PATH = os.getenv("PUBMED_METADATA_PATH", "data/pubmed_metadata.json")
# NCBI Entrez: set a real email for bulk PubMed fetches (https://www.ncbi.nlm.nih.gov/books/NBK25497/)
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL", "")
ENTREZ_TOOL = os.getenv("ENTREZ_TOOL", "alchemy-drug-discovery")
PUBMED_FETCH_BATCH = int(os.getenv("PUBMED_FETCH_BATCH", "150"))
FAISS_TOP_K = 20

# PharmaDuel
PHARMA_DUEL_ROUNDS = 3
PHARMA_DUEL_CANDIDATES = 3  # top N candidates to battle
HERG_IC50_THRESHOLD_UM = 1.0  # µM — below this = cardiotoxic FAIL
SA_SCORE_THRESHOLD = 4.0  # above this = too hard to synthesize FAIL
BINDING_DROP_THRESHOLD = 0.40  # >40% binding drop on mutation = resistance FAIL
ADMET_PASS_THRESHOLD = 0.6  # 60% of properties must pass

# Optional AutoDock Vina / smina — set receptor PDBQT + box for structure-informed scores
# Score returned to PharmaDuel is higher = better (roughly −Δaffinity kcal/mol when smina runs).
BINDING_USE_SMINA = os.getenv("BINDING_USE_SMINA", "").lower() in ("1", "true", "yes")
SMINA_BINARY = os.getenv("SMINA_BINARY", "smina")
VINA_RECEPTOR_PDBQT = os.getenv("VINA_RECEPTOR_PDBQT", "")
# Comma-separated: cx,cy,cz and sx,sy,sz (Å)
VINA_BOX_CENTER = os.getenv("VINA_BOX_CENTER", "")
VINA_BOX_SIZE = os.getenv("VINA_BOX_SIZE", "22,22,22")
SMINA_EXHAUSTIVENESS = int(os.getenv("SMINA_EXHAUSTIVENESS", "8"))
SMINA_TIMEOUT_SEC = int(os.getenv("SMINA_TIMEOUT_SEC", "120"))

# Offline pitch demo: load cached pipeline JSON (see data/demo_results/)
ALCHEMY_OFFLINE_DEMO = os.getenv("ALCHEMY_OFFLINE_DEMO", "").lower() in ("1", "true", "yes")

# APIs (all free, no auth except PubMed)
UNIPROT_API = "https://rest.uniprot.org/uniprotkb/search"
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
PUBMED_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBCHEM_API = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
OPENTARGETS_API = "https://api.platform.opentargets.org/api/v4/graphql"
ASKCOS_API = "https://askcos.mit.edu/api/v2"  # fallback synthesis check

# FastAPI
API_HOST = "0.0.0.0"
API_PORT = 8001

# Gradio
GRADIO_PORT = 7860
GRADIO_SHARE = True  # set True for HuggingFace Space
