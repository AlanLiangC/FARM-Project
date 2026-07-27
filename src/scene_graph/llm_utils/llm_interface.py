"""
Unified LLM/VLM Interface (vLLM)

Routes all LLM/VLM requests through vLLM's OpenAI-compatible API, matching
the pattern used by the mapping stack.

Usage::

    from scene_graph.llm_utils import LLMInterface

    llm = LLMInterface(verbose=True)
    text = llm.query("What is the capital of France?")
    vision = llm.query("What's in this image?", images=my_rgb_np_image)

Environment Variables (see :mod:`scene_graph.llm_utils.llm_config`):
    VLLM_BASE_URL, VLLM_API_KEY, VLLM_TIMEOUT_S
    VLLM_VL_MODEL / VLLM_LLM_MODEL / VLLM_MODEL
"""

import base64
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from pydantic import BaseModel

try:
    import cv2
except ImportError:  # pragma: no cover - optional
    cv2 = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional
    np = None

import requests

from scene_graph.llm_utils.llm_config import LLMConfig, get_config

REASONING_ONLY_PREFIX = "__QWEN_REASONING_ONLY__\n"


class LLMInterface:
    """Unified LLM/VLM interface using vLLM's OpenAI-compatible endpoint."""

    def __init__(
        self,
        verbose: bool = False,
        log_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        config: Optional[LLMConfig] = None,
    ):
        self.config = config or get_config()
        self.verbose = verbose
        if log_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = Path.cwd() / "log"
            self.log_dir = base / f"llm_logs_{ts}"
        else:
            self.log_dir = Path(log_dir)
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Read-only installs (e.g. a bind-mounted repo) must not make the
            # interface unusable — fall back to the system temp dir.
            import tempfile

            fallback = Path(tempfile.gettempdir()) / "farm_llm_logs" / self.log_dir.name
            fallback.mkdir(parents=True, exist_ok=True)
            self.log_dir = fallback
        self._model_override = model_name

    @property
    def model_name(self) -> str:
        if self._model_override:
            return self.config.resolve_model(self._model_override) or self.config.llm_model
        return self.config.llm_model

    def _vllm_chat_url(self) -> str:
        base = str(self.config.llm_url).rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _encode_image(self, image: "np.ndarray") -> Optional[str]:
        """Convert a numpy image (RGB) to base64 JPEG string."""
        if image is None or np is None:
            return None

        if cv2 is not None:
            try:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except Exception:
                image_bgr = image

            success, buffer = cv2.imencode(".jpg", image_bgr)
            if not success:
                raise ValueError("Could not encode image to JPEG format.")
            return base64.b64encode(buffer).decode("utf-8")

        try:
            from PIL import Image  # type: ignore
        except Exception as e:
            raise RuntimeError("cv2 not available and PIL not installed; cannot encode images.") from e

        img = Image.fromarray(image)
        import io

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _log_interaction(self, prompt: str, response: str, has_image: bool = False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = self.log_dir / f"reasoning_{timestamp}.txt"

        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"TIMESTAMP: {timestamp}\n")
            f.write("BACKEND: vllm\n")
            f.write(f"BASE_URL: {self.config.llm_url}\n")
            f.write(f"MODEL: {self.model_name}\n")
            f.write(f"HAS IMAGE: {has_image}\n")
            f.write("-" * 80 + "\n")
            f.write("PROMPT:\n")
            f.write(prompt + "\n\n")
            f.write("-" * 80 + "\n")
            f.write("RESPONSE:\n")
            f.write("-" * 80 + "\n")
            f.write(response + "\n")

    def _try_parse_json(self, resp: str) -> Optional[dict]:
        """Try ``json.loads`` first; fall back to a greedy ``{.*}`` regex."""
        try:
            return json.loads(resp)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", resp, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return None

    def _message_text(self, message: Dict, *, enable_thinking: Optional[bool]) -> str:
        """Extract assistant text from vLLM's OpenAI-compatible message.

        Qwen thinking mode can return the hidden chain in a non-standard
        ``reasoning`` field and leave ``content`` null if the generation budget
        is exhausted before the final answer. For normal calls we keep returning
        only content; for explicit thinking calls we fall back to reasoning so
        downstream parsers can still recover a final/partial JSON answer when it
        appears there.
        """

        content = message.get("content")
        content_text = str(content).strip() if content is not None else ""
        if content_text:
            return content_text
        if enable_thinking is True:
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if reasoning is not None:
                return REASONING_ONLY_PREFIX + str(reasoning).strip()
        return ""

    def query_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        images: Union["np.ndarray", List["np.ndarray"], None] = None,
        model: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
    ) -> BaseModel:
        """Send prompt with json_mode, parse and validate with Pydantic."""
        resp = self.query(prompt, images=images, model=model, json_mode=True, enable_thinking=enable_thinking)
        data = self._try_parse_json(resp)
        if data is not None:
            return schema.model_validate(data)

        # --- Runaway CoT fallback ---
        # The response was not valid JSON (most likely the reasoning field ran on forever
        # and hit max_tokens before the JSON was closed). Retry twice with progressively
        # tighter constraints on the reasoning field.
        logging.warning(
            "[LLM] JSON parse failed (possible runaway CoT, response length=%d). "
            "Retrying with reasoning length constraint (fallback 1/2).",
            len(resp),
        )
        prompt_f1 = (
            prompt
            + "\n\n[IMPORTANT: your previous response was not valid JSON. "
            "Respond again with the exact same JSON structure, but keep the "
            '"reasoning" field to 1-2 sentences maximum. All other fields must be present.]'
        )
        resp2 = self.query(prompt_f1, images=images, model=model, json_mode=True, enable_thinking=enable_thinking)
        data2 = self._try_parse_json(resp2)
        if data2 is not None:
            logging.info("[LLM] Runaway CoT fallback 1 succeeded.")
            return schema.model_validate(data2)

        logging.warning("[LLM] JSON parse failed again. Retrying with empty reasoning (fallback 2/2).")
        prompt_f2 = (
            prompt
            + "\n\n[IMPORTANT: your previous two responses were not valid JSON. "
            'Set "reasoning" to an empty string "" and complete all other required '
            "fields with valid values. Respond with nothing but the JSON object.]"
        )
        resp3 = self.query(prompt_f2, images=images, model=model, json_mode=True, enable_thinking=enable_thinking)
        data3 = self._try_parse_json(resp3)
        if data3 is not None:
            logging.info("[LLM] Runaway CoT fallback 2 succeeded.")
            return schema.model_validate(data3)

        raise ValueError(f"No valid JSON after 3 attempts. Last response: {resp3[:200]}")

    def query(
        self,
        prompt: str,
        images: Union["np.ndarray", List["np.ndarray"], None] = None,
        model: Optional[str] = None,
        json_mode: bool = False,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        """
        Send prompt and optional images to LLM.

        Args:
            prompt: Text prompt
            images: Single image (np.ndarray) or list of images in RGB format
            model: Optional model name override
            json_mode: If True, request JSON output from vLLM
            enable_thinking: Optional per-request override for Qwen-style
                ``chat_template_kwargs.enable_thinking``. ``None`` preserves
                the configured default.

        Returns:
            Generated text response
        """
        start_time = time.time()
        encoded_images = []

        if images is not None:
            if np is not None and isinstance(images, np.ndarray):
                target_images = [images]
            elif isinstance(images, list):
                target_images = images
            else:
                raise ValueError("Images must be a numpy array or list of numpy arrays")

            for i, img in enumerate(target_images):
                try:
                    b64_img = self._encode_image(img)
                    if b64_img:
                        encoded_images.append(b64_img)
                except Exception as e:
                    raise RuntimeError(f"Failed to process image index {i}: {e}")

        effective_model = self.config.resolve_model(model or self._model_override) or self.config.llm_model

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                if self.verbose and attempt == 0:
                    img_count = len(encoded_images)
                    print(f"[LLM] Sending request... (Images: {img_count}, Backend: vllm)")

                # OpenAI-compatible message format.
                if encoded_images:
                    content = [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                        for img_b64 in encoded_images
                    ]
                    content.append({"type": "text", "text": prompt})
                    messages = [{"role": "user", "content": content}]
                else:
                    messages = [{"role": "user", "content": prompt}]

                payload = {
                    "model": effective_model,
                    "messages": messages,
                    "temperature": float(self.config.temperature),
                    "top_p": float(getattr(self.config, "top_p", 0.9)),
                    "max_tokens": int(self.config.max_tokens),
                    "stream": False,
                }

                if json_mode:
                    payload["response_format"] = {"type": "json_object"}

                if enable_thinking is not None:
                    payload["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
                elif getattr(self.config, "disable_thinking", False):
                    payload["chat_template_kwargs"] = {"enable_thinking": False}

                headers = {"Content-Type": "application/json"}
                if self.config.vllm_api_key:
                    headers["Authorization"] = f"Bearer {self.config.vllm_api_key}"

                url = self._vllm_chat_url()
                resp = requests.post(url, json=payload, headers=headers, timeout=float(self.config.vllm_timeout_s))
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    response = ""
                else:
                    response = self._message_text(choices[0].get("message") or {}, enable_thinking=enable_thinking)

                self._log_interaction(prompt, response, has_image=bool(encoded_images))

                if self.verbose:
                    elapsed_time = time.time() - start_time
                    print(f"[LLM] Response time: {elapsed_time:.2f}s")
                    print(f"[LLM] Response: '{response[:100]}...'")

                return response

            except requests.exceptions.Timeout:
                last_error = f"Request timed out (attempt {attempt + 1})"
                if self.verbose:
                    print(f"[LLM WARNING] {last_error}")
            except requests.exceptions.RequestException as e:
                last_error = f"Request failed: {e}"
                if self.verbose:
                    print(f"[LLM ERROR] Attempt {attempt + 1} failed: {e}")

            if attempt < self.config.max_retries - 1:
                time.sleep(1)

        raise RuntimeError(f"LLM request failed after {self.config.max_retries} attempts: {last_error}")

    # ------------------------------------------------------------------
    # Multi-turn conversation methods
    # ------------------------------------------------------------------

    def query_messages(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        json_mode: bool = False,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        """Send pre-built messages array (text-only, no images)."""
        start_time = time.time()
        effective_model = self.config.resolve_model(model or self._model_override) or self.config.llm_model

        log_prompt = "\n".join(f"[{m['role'].upper()}] {m['content']}" for m in messages)

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                if self.verbose and attempt == 0:
                    print(f"[LLM] Sending multi-turn request ({len(messages)} messages, Backend: vllm)")

                payload: Dict = {
                    "model": effective_model,
                    "messages": messages,
                    "temperature": float(self.config.temperature),
                    "top_p": float(getattr(self.config, "top_p", 0.9)),
                    "max_tokens": int(self.config.max_tokens),
                    "stream": False,
                }

                if json_mode:
                    payload["response_format"] = {"type": "json_object"}

                if enable_thinking is not None:
                    payload["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
                elif getattr(self.config, "disable_thinking", False):
                    payload["chat_template_kwargs"] = {"enable_thinking": False}

                headers: Dict[str, str] = {"Content-Type": "application/json"}
                if self.config.vllm_api_key:
                    headers["Authorization"] = f"Bearer {self.config.vllm_api_key}"

                url = self._vllm_chat_url()
                resp = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=float(self.config.vllm_timeout_s),
                )
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    response = ""
                else:
                    response = self._message_text(choices[0].get("message") or {}, enable_thinking=enable_thinking)

                self._log_interaction(log_prompt, response, has_image=False)

                if self.verbose:
                    elapsed_time = time.time() - start_time
                    print(f"[LLM] Response time: {elapsed_time:.2f}s")
                    print(f"[LLM] Response: '{response[:100]}...'")

                return response

            except requests.exceptions.Timeout:
                last_error = f"Request timed out (attempt {attempt + 1})"
                if self.verbose:
                    print(f"[LLM WARNING] {last_error}")
            except requests.exceptions.RequestException as e:
                last_error = f"Request failed: {e}"
                if self.verbose:
                    print(f"[LLM ERROR] Attempt {attempt + 1} failed: {e}")

            if attempt < self.config.max_retries - 1:
                time.sleep(1)

        raise RuntimeError(f"LLM request failed after {self.config.max_retries} attempts: {last_error}")

    def query_structured_messages(
        self,
        messages: List[Dict[str, str]],
        schema: Type[BaseModel],
        model: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
    ) -> BaseModel:
        """Send messages and validate response with Pydantic schema."""
        resp = self.query_messages(messages, model=model, json_mode=True, enable_thinking=enable_thinking)
        data = self._try_parse_json(resp)
        if data is not None:
            return schema.model_validate(data)

        # --- Runaway CoT fallback (mirror of query_structured) ---
        logging.warning(
            "[LLM] JSON parse failed (possible runaway CoT, response length=%d). "
            "Retrying with reasoning length constraint (fallback 1/2).",
            len(resp),
        )
        msgs_f1 = list(messages) + [{
            "role": "user",
            "content": (
                "Your previous response was not valid JSON. "
                "Respond again with the exact same JSON structure, but keep the "
                '"reasoning" field to 1-2 sentences maximum. All other fields must be present.'
            ),
        }]
        resp2 = self.query_messages(msgs_f1, model=model, json_mode=True, enable_thinking=enable_thinking)
        data2 = self._try_parse_json(resp2)
        if data2 is not None:
            logging.info("[LLM] Runaway CoT fallback 1 succeeded.")
            return schema.model_validate(data2)

        logging.warning("[LLM] JSON parse failed again. Retrying with empty reasoning (fallback 2/2).")
        msgs_f2 = list(messages) + [{
            "role": "user",
            "content": (
                "Your previous two responses were not valid JSON. "
                'Set "reasoning" to an empty string "" and complete all other required '
                "fields with valid values. Respond with nothing but the JSON object."
            ),
        }]
        resp3 = self.query_messages(msgs_f2, model=model, json_mode=True, enable_thinking=enable_thinking)
        data3 = self._try_parse_json(resp3)
        if data3 is not None:
            logging.info("[LLM] Runaway CoT fallback 2 succeeded.")
            return schema.model_validate(data3)

        raise ValueError(f"No valid JSON after 3 attempts. Last response: {resp3[:200]}")
