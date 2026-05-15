"""
Mistral AI API — plain requests, no SDK needed.
Works on any Python version.
"""
import requests
import json
import os

def _get_key():
    key = os.getenv("MISTRAL_API_KEY", "")
    if not key:
        try:
            from config import MISTRAL_API_KEY
            key = MISTRAL_API_KEY
        except Exception:
            pass
    return key

BASE = "https://api.mistral.ai/v1"

def _h():
    return {"Authorization": f"Bearer {_get_key()}", "Content-Type": "application/json"}

def chat_complete(model, messages, tools=None, tool_choice="auto"):
    payload = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    r = requests.post(f"{BASE}/chat/completions", headers=_h(), json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def get_embedding(model, text):
    payload = {"model": model, "input": [text]}
    r = requests.post(f"{BASE}/embeddings", headers=_h(), json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]

def vision_chat(model, image_b64, prompt):
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": prompt}
            ]
        }]
    }
    r = requests.post(f"{BASE}/chat/completions", headers=_h(), json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
