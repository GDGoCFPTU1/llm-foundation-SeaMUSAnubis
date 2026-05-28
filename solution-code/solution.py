"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1
"""

import os
import time
from typing import Any, Callable

PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start

    response_text = response.choices[0].message.content
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )

    start = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.time() - start

    response_text = response.text
    usage = {
        "input_tokens": response.usage_metadata.prompt_token_count,
        "output_tokens": response.usage_metadata.candidates_token_count,
    }
    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    start = time.time()
    response = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start

    response_text = response.content[0].text
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    def _calc_cost(model_key: str, input_tokens: int, output_tokens: int) -> float:
        pricing = PRICING_1M_TOKENS[model_key]
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

    gpt4o_text, gpt4o_latency, gpt4o_usage = call_openai(prompt, model=OPENAI_MODEL)
    gpt4o_mini_text, gpt4o_mini_latency, gpt4o_mini_usage = call_openai(prompt, model=OPENAI_MINI_MODEL)
    gemini_text, gemini_latency, gemini_usage = call_gemini(prompt, model=GEMINI_MODEL)

    return {
        "gpt4o": {
            "response": gpt4o_text,
            "latency": gpt4o_latency,
            "cost": _calc_cost("gpt-4o", gpt4o_usage["input_tokens"], gpt4o_usage["output_tokens"]),
            "input_tokens": gpt4o_usage["input_tokens"],
            "output_tokens": gpt4o_usage["output_tokens"],
        },
        "gpt4o_mini": {
            "response": gpt4o_mini_text,
            "latency": gpt4o_mini_latency,
            "cost": _calc_cost("gpt-4o-mini", gpt4o_mini_usage["input_tokens"], gpt4o_mini_usage["output_tokens"]),
            "input_tokens": gpt4o_mini_usage["input_tokens"],
            "output_tokens": gpt4o_mini_usage["output_tokens"],
        },
        "gemini_flash": {
            "response": gemini_text,
            "latency": gemini_latency,
            "cost": _calc_cost("gemini-2.5-flash", gemini_usage["input_tokens"], gemini_usage["output_tokens"]),
            "input_tokens": gemini_usage["input_tokens"],
            "output_tokens": gemini_usage["output_tokens"],
        },
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history = []  # list of {"role": ..., "parts": [...]}

    print("Chatbot Gemini 2.5 Flash — gõ 'quit' hoặc 'exit' để thoát.\n")

    while True:
        user_input = input("Bạn: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("Tạm biệt!")
            break
        if not user_input:
            continue

        # Append user turn
        history.append({"role": "user", "parts": [{"text": user_input}]})

        # Keep only last 3 turns (6 messages)
        history = history[-6:]

        print("Gemini: ", end="", flush=True)
        full_response = ""
        response_stream = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=history,
        )
        for chunk in response_stream:
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text
        print()  # newline after streaming

        # Append assistant turn
        history.append({"role": "model", "parts": [{"text": full_response}]})
        history = history[-6:]


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
    raise last_exception


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    results = []
    for prompt in prompts:
        result = compare_models(prompt)
        result["prompt"] = prompt
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    header = "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |"
    separator = "|--------|-------|----------------------|---------|-----------------|------------|"
    rows = [header, separator]

    model_display = {
        "gpt4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o-Mini",
        "gemini_flash": "Gemini-Flash",
    }

    for entry in results:
        prompt = entry.get("prompt", "")
        prompt_short = prompt[:30] + "..." if len(prompt) > 30 else prompt

        for key, label in model_display.items():
            if key not in entry:
                continue
            data = entry[key]
            response_short = data["response"][:50] + "..." if len(data["response"]) > 50 else data["response"]
            latency = f"{data['latency']:.2f}s"
            tokens = f"{data['input_tokens']}/{data['output_tokens']}"
            cost = f"${data['cost']:.6f}"
            rows.append(f"| {prompt_short} | {label} | {response_short} | {latency} | {tokens} | {cost} |")

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")