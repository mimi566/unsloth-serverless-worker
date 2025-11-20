import os
import logging
from http import HTTPStatus
from functools import wraps
from time import time
import uuid

logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------
# Parse comma-separated multimodal limits
# example: "image=1,video=0"
# ------------------------------------------------------------
def convert_limit_mm_per_prompt(input_string: str):
    result = {}
    if not input_string:
        return result

    pairs = input_string.split(',')
    for pair in pairs:
        key, value = pair.split('=')
        result[key] = int(value)
    return result


# ------------------------------------------------------------
# Count physical CPU cores
# ------------------------------------------------------------
def count_physical_cores():
    with open('/proc/cpuinfo') as f:
        content = f.readlines()

    cores = set()
    current_physical_id = None
    current_core_id = None

    for line in content:
        if "physical id" in line:
            current_physical_id = line.strip().split(': ')[1]
        elif "core id" in line:
            current_core_id = line.strip().split(': ')[1]
            cores.add((current_physical_id, current_core_id))

    return len(cores)


# ------------------------------------------------------------
# Job Input (custom for Unsloth worker)
# ------------------------------------------------------------
class JobInput:
    """
    A light wrapper around incoming JSON job request.
    No vLLM SamplingParams – Unsloth uses plain Python dict sampling params.
    """
    def __init__(self, job):
        self.messages = job.get("messages", job.get("prompt"))
        self.stream = job.get("stream", False)

        # Sampling parameters (temperature, top_p, etc.)
        self.sampling_params = job.get("sampling_params", {})
        if "max_tokens" not in self.sampling_params:
            self.sampling_params["max_tokens"] = 200

        # Used for chunking & long context
        self.max_batch_size = job.get("max_batch_size")
        self.apply_chat_template = job.get("apply_chat_template", False)

        # OpenAI-format compatibility
        self.use_openai_format = job.get("use_openai_format", False)
        self.openai_route = job.get("openai_route")
        self.openai_input = job.get("openai_input")

        # Unique request ID
        self.request_id = str(uuid.uuid4())

        # Batch processing options
        batch_size_growth_factor = job.get("batch_size_growth_factor")
        self.batch_size_growth_factor = float(batch_size_growth_factor) if batch_size_growth_factor else None

        min_batch_size = job.get("min_batch_size")
        self.min_batch_size = int(min_batch_size) if min_batch_size else None


# ------------------------------------------------------------
# Dummy request used by the worker (serverless internal use)
# ------------------------------------------------------------
class DummyState:
    def __init__(self):
        self.request_metadata = None

class DummyRequest:
    def __init__(self):
        self.headers = {}
        self.state = DummyState()

    async def is_disconnected(self):
        return False


# ------------------------------------------------------------
# Batch Size Manager
# ------------------------------------------------------------
class BatchSize:
    def __init__(self, max_batch_size, min_batch_size, batch_size_growth_factor):
        self.max_batch_size = max_batch_size
        self.batch_size_growth_factor = batch_size_growth_factor
        self.min_batch_size = min_batch_size

        self.is_dynamic = (
            batch_size_growth_factor
            and batch_size_growth_factor > 1
            and min_batch_size >= 1
            and max_batch_size > min_batch_size
        )

        if self.is_dynamic:
            self.current_batch_size = min_batch_size
        else:
            self.current_batch_size = max_batch_size

    def update(self):
        if self.is_dynamic:
            new_size = self.current_batch_size * self.batch_size_growth_factor
            self.current_batch_size = min(new_size, self.max_batch_size)


# ------------------------------------------------------------
# Create error responses
# ------------------------------------------------------------
def create_error_response(message: str, err_type: str = "BadRequestError",
                          status_code: HTTPStatus = HTTPStatus.BAD_REQUEST):
    """Return a generic Python dict (OpenAI style error)."""
    return {
        "error": {
            "message": message,
            "type": err_type,
            "code": status_code.value
        }
    }


# ------------------------------------------------------------
# Boolean ENV parser
# ------------------------------------------------------------
def get_int_bool_env(env_var: str, default: bool) -> bool:
    return int(os.getenv(env_var, int(default))) == 1


# ------------------------------------------------------------
# Timer decorator for profiling
# ------------------------------------------------------------
def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        logging.info(f"{func.__name__} completed in {end - start:.2f} seconds")
        return result
    return wrapper
