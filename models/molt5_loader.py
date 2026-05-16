"""Lazy MolT5 loader — prefer MoleculeAgent which loads on first use."""
from transformers import T5ForConditionalGeneration, T5Tokenizer

from config.settings import MOLT5_MODEL


def load_molt5():
    tokenizer = T5Tokenizer.from_pretrained(MOLT5_MODEL)
    model = T5ForConditionalGeneration.from_pretrained(MOLT5_MODEL)
    return tokenizer, model
