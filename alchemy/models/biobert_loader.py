"""BioBERT for literature RAG — optional; LiteratureAgent uses PubMed + Qwen by default."""
from transformers import AutoModel, AutoTokenizer

from config.settings import BIOBERT_MODEL


def load_biobert():
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
    model = AutoModel.from_pretrained(BIOBERT_MODEL)
    return tokenizer, model
