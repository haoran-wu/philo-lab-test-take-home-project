from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests


MODEL_ID_ENV_KEYS = {
    "gemini-3.1-pro": "MODEL_ID_GEMINI_3_1_PRO",
    "gemini-3-flash": "MODEL_ID_GEMINI_3_FLASH",
    "qwen-3.5-omni": "MODEL_ID_QWEN_3_5_OMNI",
    "gpt-5.4": "MODEL_ID_GPT_5_4",
}

MODEL_REGISTRY = {
    "gemini-3.1-pro": {"openrouter_id": "google/gemini-3.1-pro-preview"},
    "gemini-3-flash": {"openrouter_id": "google/gemini-3-flash-preview"},
    "qwen-3.5-omni": {"openrouter_id": "qwen/qwen3.5-397b-a17b"},
    "gpt-5.4": {"openrouter_id": "openai/gpt-5.4"},
}

PROMPT_TYPES = {"baseline", "probe_a", "probe_b"}


@dataclass
class VLMResponse:
    failure_id: str
    model: str
    prompt_type: str
    judgment: str
    confidence: int | None
    explanation: str
    raw_response: str
    parse_method: str
    frame_ids_used: list[int]
    image_paths_used: list[str]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _path_to_data_url(path: str | Path) -> str:
    suffix = Path(path).suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(content)


def _strip_markdown_fence(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if present."""
    stripped = text.strip()
    match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else text


def _extract_first_json_block(raw_text: str) -> dict[str, Any] | None:
    text = _strip_markdown_fence(raw_text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_text_fallback(raw_text: str) -> tuple[str, int | None, str]:
    judgment_match = re.search(
        r'"judgment"\s*:\s*"(PASS|FAIL)"|\b(PASS|FAIL)\b',
        raw_text,
        flags=re.IGNORECASE,
    )
    confidence_match = re.search(
        r'"confidence"\s*:\s*([1-5])|confidence[^0-9]*([1-5])',
        raw_text,
        flags=re.IGNORECASE,
    )
    if judgment_match:
        judgment = next(group.upper() for group in judgment_match.groups() if group)
    else:
        judgment = "UNKNOWN"
    if confidence_match:
        confidence = int(next(group for group in confidence_match.groups() if group))
    else:
        confidence = None
    return judgment, confidence, raw_text.strip()


def parse_vlm_output(raw_text: str) -> tuple[str, int | None, str, str]:
    parsed = _extract_first_json_block(raw_text)
    if parsed is not None:
        judgment = str(parsed.get("judgment", "UNKNOWN")).upper()
        confidence_value = parsed.get("confidence")
        confidence = int(confidence_value) if isinstance(confidence_value, (int, float, str)) and str(confidence_value).isdigit() else None
        explanation = str(parsed.get("explanation") or parsed.get("cross_frame_comparison") or raw_text).strip()
        return judgment, confidence, explanation, "json"
    judgment, confidence, explanation = _parse_text_fallback(raw_text)
    return judgment, confidence, explanation, "regex_fallback"


class OpenRouterClient:
    def __init__(self) -> None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to .env or your shell environment.")
        self.api_key = api_key
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
        self.http_referer = os.getenv("OPENROUTER_HTTP_REFERER")
        self.x_title = os.getenv("OPENROUTER_X_TITLE", "vlm-blindspot-discovery")
        self.timeout = 180

    def _build_prompt(self, question: str, prompt_type: str) -> str:
        if prompt_type == "baseline":
            return (
                "You are evaluating an AI-generated video for a specific visual defect.\n\n"
                "You are given key frames extracted from the same video segment.\n\n"
                f"Question: {question}\n\n"
                "Respond in the following JSON format only:\n"
                '{\n  "judgment": "PASS or FAIL",\n  "confidence": <1-5>,\n  "explanation": "<brief explanation>"\n}'
            )
        if prompt_type in {"probe_a", "probe_b"}:
            return (
                "You are evaluating whether a specific defect exists across the provided frames.\n\n"
                "Please first describe what you observe in EACH frame using one short sentence per frame, "
                "then compare across frames briefly, then answer the question.\n\n"
                f"Question: {question}\n\n"
                "Respond in the following JSON format only:\n"
                '{\n'
                '  "evidence_by_frame": {"Frame N": "<observations>"},\n'
                '  "cross_frame_comparison": "<comparison>",\n'
                '  "judgment": "PASS or FAIL",\n'
                '  "confidence": <1-5>,\n'
                '  "explanation": "<brief explanation>"\n'
                '}'
            )
        raise ValueError(f"unsupported prompt_type: {prompt_type}")

    def _build_user_content(self, image_paths: list[Path], frame_ids: list[int], question: str, prompt_type: str) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": self._build_prompt(question, prompt_type)}]
        for frame_id, image_path in zip(frame_ids, image_paths):
            content.append({"type": "text", "text": f"Frame {frame_id}"})
            content.append({"type": "image_url", "image_url": {"url": _path_to_data_url(image_path)}})
        return content

    def query_vlm(self, model_name: str, image_paths: list[Path], frame_ids: list[int], question: str, failure_id: str, prompt_type: str) -> VLMResponse:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"unsupported model: {model_name}")
        if prompt_type not in PROMPT_TYPES:
            raise ValueError(f"unsupported prompt_type: {prompt_type}")

        env_key = MODEL_ID_ENV_KEYS[model_name]
        model_id = os.getenv(env_key, MODEL_REGISTRY[model_name]["openrouter_id"])

        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": self._build_user_content(image_paths, frame_ids, question, prompt_type),
                }
            ],
            "temperature": 0,
            "max_tokens": 1024 if prompt_type in {"probe_a", "probe_b"} else 512,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title

        response = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]["content"]
        raw_text = _extract_text_content(message)
        judgment, confidence, explanation, parse_method = parse_vlm_output(raw_text)
        return VLMResponse(
            failure_id=failure_id,
            model=model_name,
            prompt_type=prompt_type,
            judgment=judgment,
            confidence=confidence,
            explanation=explanation,
            raw_response=raw_text,
            parse_method=parse_method,
            frame_ids_used=frame_ids,
            image_paths_used=[str(path) for path in image_paths],
        )
