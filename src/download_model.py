# src/download_model.py
import os
import json
import logging
from huggingface_hub import snapshot_download
import time

BASE_DIR = "/" 
TOKENIZER_PATTERNS = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
MODEL_PATTERNS = ["*.safetensors", "*.bin", "*.pt", "config.json", "generation_config.json", "special_tokens_map.json", "*.jinja"]

logging.basicConfig(level=logging.INFO)

# Timer decorator
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"{func.__name__} completed in {end - start:.2f}s")
        return result
    return wrapper

def setup_env():
    """Set environment for testing."""
    if os.getenv("TESTING_DOWNLOAD") == "1":
        global BASE_DIR
        BASE_DIR = "tmp"
        os.makedirs(BASE_DIR, exist_ok=True)
        os.environ.update({
            "HF_HOME": f"{BASE_DIR}/hf_cache",
            "MODEL_NAME": "openchat/openchat-3.5-0106",
            "HF_HUB_ENABLE_HF_TRANSFER": "1"
        })

@timer_decorator
def download(name, revision, type, cache_dir):
    """Download model or tokenizer from Hugging Face Hub."""
    if type == "model":
        patterns = MODEL_PATTERNS
    elif type == "tokenizer":
        patterns = TOKENIZER_PATTERNS
    else:
        raise ValueError(f"Invalid type: {type}")

    try:
        path = snapshot_download(
            repo_id=name,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=patterns,
            local_files_only=False
        )
        logging.info(f"Downloaded {type} from {name} to {path}")
        return path
    except Exception as e:
        raise ValueError(f"Failed to download {type} from {name}: {e}")

if __name__ == "__main__":
    setup_env()
    cache_dir = os.getenv("HF_HOME", "/tmp/hf_cache")
    os.makedirs(cache_dir, exist_ok=True)

    model_name = os.getenv("MODEL_NAME")
    model_revision = os.getenv("MODEL_REVISION") or None
    tokenizer_name = os.getenv("TOKENIZER_NAME") or model_name
    tokenizer_revision = os.getenv("TOKENIZER_REVISION") or model_revision

    # Download model (includes weights + config + templates)
    model_path = download(model_name, model_revision, "model", cache_dir)

    # Download tokenizer
    tokenizer_path = download(tokenizer_name, tokenizer_revision, "tokenizer", cache_dir)

    # Save metadata for FastLanguageModel
    metadata = {
        "MODEL_NAME": model_path,
        "MODEL_REVISION": model_revision,
        "TOKENIZER_NAME": tokenizer_path,
        "TOKENIZER_REVISION": tokenizer_revision
    }

    with open(f"{BASE_DIR}/local_model_args.json", "w") as f:
        json.dump({k: v for k, v in metadata.items() if v not in (None, "")}, f)

    logging.info(f"Saved local model metadata: {metadata}")
