# src/engine.py
import os
import logging
import json
import asyncio
from typing import AsyncGenerator, Optional, Dict, Any, List
import time

from unsloth import FastLanguageModel
from transformers import TextStreamer, AutoTokenizer

from utils import DummyRequest, JobInput, BatchSize, create_error_response
from tokenizer import TokenizerWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("engine")

# Environment-driven config (with sensible defaults)
MODEL_NAME = os.getenv("MODEL_NAME", "Sourabh66/Llama-2-17B-Fine-Tune-Blog")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "32768"))
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", None)
HF_TOKEN = os.getenv("HF_TOKEN", None)
CHUNK_MAX_WORDS = int(os.getenv("CHUNK_MAX_WORDS", "2500"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "50"))


def split_text_into_chunks(text: str, max_words: int = CHUNK_MAX_WORDS, overlap_words: int = CHUNK_OVERLAP_WORDS):
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        chunk = words[i:i + max_words]
        chunks.append(" ".join(chunk))
        i += max_words - overlap_words
    return chunks


class UnslothEngine:
    """
    Simple Unsloth-based engine that loads FastLanguageModel once
    and exposes an async generator `generate(job_input)` that yields dicts.
    """

    def __init__(self):
        start = time.time()
        logger.info(f"Initializing UnslothEngine with model={MODEL_NAME} cache_dir={MODEL_CACHE_DIR}")
        # Load model + tokenizer via Unsloth helper
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_NAME,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=True,
                cache_dir=MODEL_CACHE_DIR if MODEL_CACHE_DIR and os.path.exists(MODEL_CACHE_DIR) else None,
                use_auth_token=HF_TOKEN if HF_TOKEN else None,
            )
            # Try enabling inference optimizations
            try:
                FastLanguageModel.for_inference(self.model)
            except Exception as e:
                logger.info(f"for_inference() skipped/failed: {e}")
        except Exception as e:
            logger.exception("Failed to load model")
            raise e
        end = time.time()
        logger.info(f"Loaded Unsloth model in {end - start:.2f}s")
        # Wrap tokenizer with TokenizerWrapper if possible (some models already have chat template)
        try:
            # TokenizerWrapper expects tokenizer name and revision; we try to detect name from env
            self.tokenizer_wrapper = TokenizerWrapper(MODEL_NAME, None, True)
        except Exception:
            # Fallback minimal wrapper using transformers' AutoTokenizer
            try:
                hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
                class MinimalWrapper:
                    def __init__(self, tokenizer):
                        self.tokenizer = tokenizer
                        self.custom_chat_template = os.getenv("CUSTOM_CHAT_TEMPLATE")
                        self.has_chat_template = hasattr(self.tokenizer, "chat_template") and bool(self.tokenizer.chat_template) or bool(self.custom_chat_template)
                        if self.custom_chat_template:
                            self.tokenizer.chat_template = self.custom_chat_template
                    def apply_chat_template(self, inp):
                        if isinstance(inp, str):
                            inp = [{"role":"user","content":inp}]
                        return self.tokenizer.apply_chat_template(inp, tokenize=False, add_generation_prompt=True)
                self.tokenizer_wrapper = MinimalWrapper(hf_tokenizer)
            except Exception:
                logger.warning("Could not initialize TokenizerWrapper or fallback. Chat-template features will be disabled.")
                self.tokenizer_wrapper = None

    async def generate(self, job_input: JobInput) -> AsyncGenerator[dict, None]:
        """
        job_input: JobInput wrapper
        Yields dicts shaped like:
         {"choices":[{"text": "..."}], "usage": {"input": N, "output": M}}
        """
        try:
            # Determine prompt string
            raw_input = job_input.messages if hasattr(job_input, "messages") else job_input.llm_input
            apply_chat_template = getattr(job_input, "apply_chat_template", False)
            if (apply_chat_template or isinstance(raw_input, list)) and self.tokenizer_wrapper:
                try:
                    prompt = self.tokenizer_wrapper.apply_chat_template(raw_input)
                except Exception as e:
                    # If tokenization template fails, fallback to simple concatenation
                    logger.warning(f"apply_chat_template failed: {e}. Falling back to concatenation.")
                    prompt = self._concat_messages(raw_input)
            else:
                prompt = self._concat_messages(raw_input)

            # Chunking
            chunks = split_text_into_chunks(prompt)
            sampling = job_input.sampling_params if isinstance(job_input.sampling_params, dict) else {}
            max_new_tokens = int(sampling.get("max_tokens", 200))
            temperature = float(sampling.get("temperature", 1.0))
            top_p = float(sampling.get("top_p", 0.95))
            do_sample = sampling.get("do_sample", True)

            total_input_tokens = 0  # rough metric (we do not compute tokens accurately here)
            total_output_tokens = 0

            # For each chunk, run blocking generate in threadpool
            for i, chunk in enumerate(chunks):
                logger.info(f"Generating chunk {i+1}/{len(chunks)} (words={len(chunk.split())})")
                # Prepare inputs for model; some unsloth models accept tokenizer directly (we'll use model.tokenize path)
                # We call generate in a thread to avoid blocking event loop
                def blocking_gen(chunk_text):
                    # Use tokenizer to create inputs if available
                    try:
                        inputs = self.tokenizer(chunk_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
                        # Model.generate call
                        out = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=do_sample,
                        )
                        # out might be a tensor or sequences object; handle common cases
                        try:
                            if hasattr(out, "sequences"):
                                seq = out.sequences[0]
                            else:
                                seq = out[0]
                            text = self.tokenizer.decode(seq, skip_special_tokens=True)
                        except Exception:
                            # fallback: try model's text output if available
                            text = str(out)
                    except Exception as e:
                        logger.exception("Synchronous generation error")
                        text = f"[GEN_ERROR] {e}"
                    return text

                generated_text = await asyncio.to_thread(blocking_gen, chunk)
                total_output_tokens += len(generated_text.split())
                # Build response batch (OpenAI-like)
                batch = {
                    "choices": [{"text": generated_text}],
                    "usage": {
                        "input": len(chunk.split()),  # word-level proxy
                        "output": len(generated_text.split())
                    },
                    "chunk_index": i,
                    "chunks_total": len(chunks)
                }
                yield batch

            # final aggregate (optional)
            logger.info(f"Generation complete. total_output_words={total_output_tokens}")
        except Exception as e:
            logger.exception("Error in UnslothEngine.generate")
            yield create_error_response(str(e))

    def _concat_messages(self, raw_input) -> str:
        """
        Convert either a single string prompt or a list of message dicts into a single prompt string.
        """
        if isinstance(raw_input, str):
            return raw_input
        elif isinstance(raw_input, list):
            out = ""
            for m in raw_input:
                role = m.get("role", "user")
                content = m.get("content", "")
                out += f"[{role}]: {content}\n"
            return out
        else:
            return str(raw_input)


class OpenAIUnslothEngine(UnslothEngine):
    """
    Basic OpenAI-compatible wrapper. Supports:
     - /v1/chat/completions (simple mapping to prompt)
     - /v1/completions
     - /v1/models (basic listing)
    """
    def __init__(self):
        super().__init__()
        self.served_model_name = os.getenv("OPENAI_SERVED_MODEL_NAME_OVERRIDE") or MODEL_NAME
        # raw_openai_output determines whether to pass raw SSE strings (not used here) — accept bool
        raw_output_env = os.getenv("RAW_OPENAI_OUTPUT", "0")
        self.raw_openai_output = raw_output_env.lower() in ("1", "true", "yes")

    async def generate(self, openai_request: JobInput) -> AsyncGenerator[dict, None]:
        # If the caller requested model listing
        if getattr(openai_request, "openai_route", None) == "/v1/models":
            # a minimal models response
            yield {"data": [{"id": self.served_model_name, "object": "model"}]}
            return

        # For compeletion/chat endpoints, transform openai_input to prompt/messages
        openai_input = getattr(openai_request, "openai_input", None) or {}
        route = getattr(openai_request, "openai_route", None)
        if route in ("/v1/chat/completions", "/v1/completions"):
            # If messages provided (chat style), use them
            if "messages" in openai_input:
                job = {
                    "messages": openai_input["messages"],
                    "sampling_params": {
                        "max_tokens": openai_input.get("max_tokens", 200),
                        "temperature": openai_input.get("temperature", 1.0),
                        "top_p": openai_input.get("top_p", 0.95),
                        "do_sample": openai_input.get("do_sample", True),
                    },
                    "stream": openai_input.get("stream", False),
                    "apply_chat_template": True,
                }
            else:
                # text completion style (prompt)
                job = {
                    "messages": openai_input.get("prompt", openai_input.get("input", "")),
                    "sampling_params": {
                        "max_tokens": openai_input.get("max_tokens", 200),
                        "temperature": openai_input.get("temperature", 1.0),
                        "top_p": openai_input.get("top_p", 0.95),
                        "do_sample": openai_input.get("do_sample", True),
                    },
                    "stream": openai_input.get("stream", False),
                    "apply_chat_template": False,
                }

            wrapped_job = JobInput(job)
            # Reuse UnslothEngine generate to produce chunks
            async for out in super().generate(wrapped_job):
                # For OpenAI-compat, reformat into choices array
                if "choices" in out and isinstance(out["choices"], list):
                    # convert to a simpler openai-like output
                    text = out["choices"][0].get("text", "")
                    yield {"choices": [{"text": text}], "usage": out.get("usage", {})}
                else:
                    yield out
        else:
            yield create_error_response("Invalid OpenAI route: " + str(route))


# Singletons for importers
UNSLOTH_ENGINE = UnslothEngine()
OPENAI_UNSLOTH_ENGINE = OpenAIUnslothEngine()
