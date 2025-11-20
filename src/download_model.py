# src/download_model.py
import os
import json
import logging
import glob
from huggingface_hub import snapshot_download
from utils import timer_decorator

BASE_DIR = "/" 
TOKENIZER_PATTERNS = ["tokenizer.json", "tokenizer*"]
MODEL_PATTERNS = ["*.safetensors", "*.bin", "*.pt"]

logging.basicConfig(level=logging.INFO)

def setup_env():
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
            allow_patterns=patterns
        )
        logging.info(f"Downloaded {type} from {name} to {path}")
        return path
    except Exception as e:
        raise ValueError(f"Failed to download {type} from {name}: {e}")

if __name__ == "__main__":
    setup_env()
    cache_dir = os.getenv("HF_HOME")
    model_name = os.getenv("MODEL_NAME")
    model_revision = os.getenv("MODEL_REVISION") or None
    tokenizer_name = os.getenv("TOKENIZER_NAME") or model_name
    tokenizer_revision = os.getenv("TOKENIZER_REVISION") or model_revision

    model_path = download(model_name, model_revision, "model", cache_dir)
    tokenizer_path = download(tokenizer_name, tokenizer_revision, "tokenizer", cache_dir)

    metadata = {
        "MODEL_NAME": model_path,
        "MODEL_REVISION": model_revision,
        "TOKENIZER_NAME": tokenizer_path,
        "TOKENIZER_REVISION": tokenizer_revision
    }

    with open(f"{BASE_DIR}/local_model_args.json", "w") as f:
        json.dump({k: v for k, v in metadata.items() if v not in (None, "")}, f)

    logging.info(f"Saved local model metadata: {metadata}")
