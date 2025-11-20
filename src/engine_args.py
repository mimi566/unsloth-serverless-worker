# src/engine_args.py (Unsloth version)
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("engine_args")

# Environment-driven config for Unsloth
MODEL_NAME = os.getenv("MODEL_NAME", "Sourabh66/Llama-2-17B-Fine-Tune-Blog")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", None)
HF_TOKEN = os.getenv("HF_TOKEN", None)
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", 32768))
CHUNK_MAX_WORDS = int(os.getenv("CHUNK_MAX_WORDS", 2500))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", 50))

def get_engine_args():
    """
    Returns a simple dict of Unsloth engine config.
    """
    args = {
        "model_name": MODEL_NAME,
        "cache_dir": MODEL_CACHE_DIR,
        "hf_token": HF_TOKEN,
        "max_seq_length": MAX_SEQ_LENGTH,
        "chunk_max_words": CHUNK_MAX_WORDS,
        "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
    }
    logger.info(f"Using Unsloth engine args: {args}")
    return args
