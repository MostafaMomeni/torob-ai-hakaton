
import os
import logging
from typing import Optional, Dict, Any, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("shopping_assistant.llm")


# Environment configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

# Base URL; /responses will be appended automatically.
LLM_API_URL = os.getenv(
    "LLM_API_URL",
    "https://ai.boxapi.ir/ai/v1",
).rstrip("/")

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    os.getenv("OPENAI_MODEL", "openai/gpt-6-luna"),
)

# Correct API key configuration
LLM_API_KEY = (
    os.getenv("LLM_API_KEY")
    or "box-ai-27c38eaf53b05d7d994c00092b742d644b9e861d497d7c0a"
)

LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "50"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_RETRY_BACKOFF = float(os.getenv("LLM_RETRY_BACKOFF", "1.0"))

if not LLM_API_KEY:
    logger.warning(
        "No LLM_API_KEY or OPENAI_API_KEY configured."
    )


class LLMError(Exception):
    """Raised when the LLM request fails."""
    pass


class LLMAdapter:
    """
    OpenAI Responses API adapter.

    Supports:
    - /responses endpoint
    - instructions and input
    - max_output_tokens
    - output_text and output[].content[].text
    - HTTP retries and timeout
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        retry_backoff: Optional[float] = None,
        session: Optional[requests.Session] = None,
    ):
        self.provider = provider or LLM_PROVIDER
        self.api_url = (api_url or LLM_API_URL).rstrip("/")
        self.model = model or LLM_MODEL
        self.api_key = api_key or LLM_API_KEY

        self.timeout = (
            timeout if timeout is not None else LLM_TIMEOUT
        )
        self.max_retries = (
            max_retries
            if max_retries is not None
            else LLM_MAX_RETRIES
        )
        self.retry_backoff = (
            retry_backoff
            if retry_backoff is not None
            else LLM_RETRY_BACKOFF
        )

        self.session = session or requests.Session()

        retries = Retry(
            total=self.max_retries,
            connect=self.max_retries,
            read=self.max_retries,
            status=self.max_retries,
            backoff_factor=self.retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
            respect_retry_after_header=True,
        )

        adapter = HTTPAdapter(max_retries=retries)

        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _endpoint(self) -> str:
        """
        Accept either a base URL or a complete /responses URL.
        """
        if self.api_url.endswith("/responses"):
            return self.api_url

        return f"{self.api_url}/responses"

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    @staticmethod
    def _prepare_messages(
        messages: List[Dict[str, Any]],
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert chat-style messages to Responses API input.

        System/developer messages become instructions.
        Other messages are sent through input.
        """
        instructions = []
        input_messages = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if isinstance(content, str):
                text_content = content
            elif isinstance(content, list):
                # Preserve text parts in a simple text-only adapter.
                parts = []

                for part in content:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict):
                        text_value = part.get("text")
                        if isinstance(text_value, str):
                            parts.append(text_value)

                text_content = "\n".join(parts)
            else:
                text_content = str(content)

            if role in ("system", "developer"):
                instructions.append(text_content)
            else:
                input_messages.append({
                    "role": role,
                    "content": text_content,
                })

        combined_instructions = (
            "\n\n".join(instructions) if instructions else None
        )

        return combined_instructions, input_messages

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        """
        Extract text from common Responses API response formats.

        This is defensive: providers vary. We scan known fields and fall back to a recursive
        search for plausible text-containing fields, but avoid binary or non-text outputs.
        """
        # Quick check for common top-level text fields
        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        # Some providers use 'output' (list of items with content)
        output = data.get("output") or data.get("outputs")
        collected_text: List[str] = []

        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue

                # direct text fields
                for key in ("text", "content_text", "output_text"):
                    v = item.get(key)
                    if isinstance(v, str) and v.strip():
                        collected_text.append(v)

                # content can be a list of parts
                content = item.get("content") or item.get("data")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, str) and part.strip():
                            collected_text.append(part)
                        elif isinstance(part, dict):
                            for k in ("text", "content_text", "html", "caption"):
                                val = part.get(k)
                                if isinstance(val, str) and val.strip():
                                    collected_text.append(val)

        if collected_text:
            return "\n".join(collected_text)

        # Responses sometimes nest text under 'result' or 'data'
        for key in ("result", "data", "response"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val
            if isinstance(val, dict):
                # try a shallow text extraction
                for tk in ("text", "content", "message"):
                    tv = val.get(tk)
                    if isinstance(tv, str) and tv.strip():
                        return tv

        # Chat-compatible fallback (choices/message/text)
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                # OpenAI chat-style
                message = choice.get("message") or choice.get("delta")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content
                text_value = choice.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    return text_value

        # As a last resort, recursively scan for text-like fields but avoid large binary blobs.
        def _recurse_find_text(obj: Any, depth: int = 0) -> Optional[str]:
            if depth > 6:
                return None
            if isinstance(obj, str) and obj.strip():
                return obj
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in ("text", "output_text", "content", "message", "result"):
                        if isinstance(v, str) and v.strip():
                            return v
                    res = _recurse_find_text(v, depth + 1)
                    if isinstance(res, str) and res.strip():
                        return res
            if isinstance(obj, list):
                for it in obj:
                    res = _recurse_find_text(it, depth + 1)
                    if isinstance(res, str) and res.strip():
                        return res
            return None

        found = _recurse_find_text(data)
        if found:
            return found

        # If an error block exists, surface it
        error = data.get("error")
        if error:
            if isinstance(error, dict):
                message = error.get("message", str(error))
            else:
                message = str(error)
            raise LLMError(f"LLM API error: {message}")

        # No extractable text found
        raise LLMError(
            "LLM response did not contain extractable text. "
            "The response may contain only non-text output or use an unsupported format."
        )

    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1024,
        reasoning_effort: Optional[str] = None,
        text_verbosity: Optional[str] = None,
    ) -> str:
        """
        Send a request using the Responses API.

        max_tokens is retained for compatibility with existing callers.
        It is mapped to max_output_tokens in the request.
        """

        if not self.api_key:
            raise LLMError(
                "Missing API key. Set LLM_API_KEY or OPENAI_API_KEY."
            )

        if not self.model:
            raise LLMError("Missing LLM_MODEL configuration.")

        instructions, input_messages = self._prepare_messages(messages)

        payload: Dict[str, Any] = {
            "model": self.model,
            "input": input_messages,
            "max_output_tokens": max_tokens,
        }

        if instructions:
            payload["instructions"] = instructions

        # Optional Responses API parameters.
        if reasoning_effort:
            payload["reasoning"] = {
                "effort": reasoning_effort,
            }

        if text_verbosity:
            payload["text"] = {
                "verbosity": text_verbosity,
            }

        url = self._endpoint()

        # Logging options: use LLM_LOG_FULL env to include longer snippets
        try:
            show_full = os.getenv("LLM_LOG_FULL", "false").lower() in ("1", "true", "yes")
        except Exception:
            show_full = False

        def _snip(s: str, n: int = 500) -> str:
            if s is None:
                return ''
            s = str(s)
            return s if len(s) <= n else s[:n] + "..."

        # Log sanitized request summary (never log API key or headers)
        try:
            req_summary = {
                "model": self.model,
                "max_output_tokens": payload.get("max_output_tokens"),
                "instructions_present": bool(payload.get("instructions")),
                "input_messages_count": len(payload.get("input") or []),
            }
            first_msg = (payload.get("input") or [{}])[0].get("content") if (payload.get("input") or []) else None
            if first_msg:
                req_summary["first_input_snippet"] = _snip(first_msg, 300)
            logger.debug("LLM request summary: %s", req_summary)
            # Also print a clear console-visible summary so the user sees the LLM input
            try:
                print("LLM_IN_SUMMARY:", req_summary)
                if show_full:
                    # sanitized full payload (do not include headers or API key)
                    full_payload = {k: v for k, v in payload.items()}
                    logger.debug("LLM full request payload (sanitized): %s", full_payload)
                    print("LLM_IN_FULL_PAYLOAD:", full_payload)
            except Exception:
                # Printing should never break the request flow
                logger.exception("Failed to print LLM input summary")
        except Exception:
            logger.exception("Failed to log LLM request summary")

        try:
            response = self.session.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            logger.error("LLM network request failed: %s", exc)
            raise LLMError(
                f"LLM request failed: {exc}"
            ) from exc

        # Log response status and a safe snippet of the body for diagnostics (never log authorization headers)
        try:
            resp_snip = _snip(response.text, 1500)
            logger.debug("LLM HTTP %s response snippet: %s", response.status_code, resp_snip)
            # Print a concise response snippet to console for visibility
            try:
                print("LLM_HTTP_STATUS:", response.status_code)
                print("LLM_HTTP_SNIPPET:", _snip(response.text, 1000))
            except Exception:
                logger.exception("Failed to print LLM HTTP snippet")
        except Exception:
            logger.exception("Failed to log LLM response snippet")

        if response.status_code >= 400:
            # Do not log the API key or request headers.
            try:
                error_data = response.json()
                error = error_data.get("error", error_data)

                if isinstance(error, dict):
                    error_message = error.get(
                        "message",
                        str(error),
                    )
                    error_code = error.get("code")
                else:
                    error_message = str(error)
                    error_code = None

                # Avoid exposing arbitrary response bodies.
                logger.error(
                    "LLM API returned HTTP %s; code=%s",
                    response.status_code,
                    error_code,
                )

                raise LLMError(
                    f"LLM API error {response.status_code}: "
                    f"{error_message}"
                )
            except ValueError as exc:
                logger.error(
                    "LLM API returned HTTP %s with non-JSON body",
                    response.status_code,
                )
                raise LLMError(
                    f"LLM API error: HTTP {response.status_code}"
                ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError(
                "LLM returned a non-JSON response."
            ) from exc

        if not isinstance(data, dict):
            raise LLMError("Unexpected LLM response structure.")

        # Extract text and print output summary to console
        try:
            extracted = self._extract_text(data)
            try:
                print("LLM_OUT:", _snip(extracted, 2000))
                if show_full:
                    # when enabled show raw JSON response (safe to print)
                    try:
                        print("LLM_RAW_JSON_RESPONSE:", data)
                    except Exception:
                        logger.exception("Failed to print raw LLM JSON response")
            except Exception:
                logger.exception("Failed to print LLM output")
        except Exception:
            # If extraction failed, re-raise to keep original behavior
            raise

        return extracted


if __name__ == "__main__":
    # Use centralized logging configuration
    from logging_config import configure_logging
    configure_logging()
    adapter = LLMAdapter()
    try:
        result = adapter.generate(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hi in Persian."},
            ],
            max_tokens=200,
        )
        logger = logging.getLogger('shopping_assistant.llm')
        logger.info('LLM demo call successful')
        # Make the output visible on the console by using the explicit prefix
        logger.info(f"LLM_OUTPUT: {result}")
    except LLMError as exc:
        logger = logging.getLogger('shopping_assistant.llm')
        logger.error('LLM demo failed: %s', exc)
