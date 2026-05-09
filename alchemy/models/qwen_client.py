# models/qwen_client.py
"""
Client for Qwen2.5-72B-Instruct served via vLLM.

Start vLLM server first:
python -m vllm.entrypoints.openai.api_server \\
    --model Qwen/Qwen2.5-72B-Instruct \\
    --dtype float16 \\
    --max-model-len 8192 \\
    --gpu-memory-utilization 0.75 \\
    --port 8000
"""
import json
import re

from openai import OpenAI

from config.settings import QWEN_MODEL, VLLM_API_KEY, VLLM_HOST

client = OpenAI(
    base_url=f"{VLLM_HOST.rstrip('/')}/v1",
    api_key=VLLM_API_KEY,
)


def qwen_chat(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    """Send a message to Qwen2.5-72B and return the response."""
    try:
        response = client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return (
            f"[vLLM unavailable: {e!s}. Start server at {VLLM_HOST} or set VLLM_HOST in .env]\n"
            "Stub: Complete wet-lab validation before any clinical use."
        )


def qwen_json(system_prompt: str, user_message: str) -> dict:
    """Get a JSON response from Qwen. Forces valid JSON output."""
    system_prompt += (
        "\n\nIMPORTANT: Respond ONLY with valid JSON. No preamble, no explanation, "
        "no markdown code blocks. Raw JSON only."
    )
    response = qwen_chat(system_prompt, user_message, temperature=0.1)
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"error": "Failed to parse JSON", "raw": response}
