import os
import runpod
from utils import JobInput
from engine import vLLMEngine, OpenAIvLLMEngine


# Initialize engines only once at container startup
vllm_engine = vLLMEngine()
openai_engine = OpenAIvLLMEngine(vllm_engine)


async def handler(job):
    """
    Main RunPod handler.
    Supports:
      - Standard vLLM requests
      - OpenAI-compatible ChatCompletions requests
    """

    # Safety check
    if "input" not in job:
        raise ValueError("Job missing 'input' field")

    # Parse job input
    job_input = JobInput(job["input"])

    # Decide which engine to use
    # `openai_route == True` → OpenAI ChatCompletions
    engine = openai_engine if job_input.openai_route else vllm_engine

    # Execute generation (async generator)
    async for output_batch in engine.generate(job_input):
        # Stream each chunk back to RunPod
        yield output_batch


# Start serverless worker
runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda _: vllm_engine.max_concurrency,
        "return_aggregate_stream": True,
    }
)
