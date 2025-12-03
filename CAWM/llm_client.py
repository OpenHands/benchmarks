import os
import json
import logging
from typing import Optional, Dict, Any, Union, Tuple

from tenacity import Retrying, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logger = logging.getLogger(__name__)

# Try to import LLM libraries
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    openai = None  # type: ignore[assignment]
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    anthropic = None  # type: ignore[assignment]
    HAS_ANTHROPIC = False

class LLMClient:
    """
    Unified client for LLM API calls (OpenAI, Anthropic, OpenRouter).
    """
    
    def __init__(
        self,
        provider: str = "openrouter",
        model: str = "anthropic/claude-3.5-sonnet",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,  # None = use provider's default (no limit)
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens  # None means no limit, use provider's max
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

        # Resolve API key
        self.api_key = api_key
        if not self.api_key:
            if provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif provider == "openrouter":
                self.api_key = os.getenv("OPENROUTER_API_KEY")
        
        self._validate_setup()

    def _validate_setup(self):
        """Validate dependencies and API keys."""
        if self.provider in ["openai", "openrouter"]:
            if not HAS_OPENAI:
                raise ImportError("openai package not installed. Run: pip install openai")
        elif self.provider == "anthropic":
            if not HAS_ANTHROPIC:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        if not self.api_key:
            # For mock/testing purposes, we might allow missing key, 
            # but generally it should be present.
            logger.warning(f"API key not found for {self.provider}. Calls may fail.")

    def _get_retry_exceptions(self) -> Tuple[Any, ...]:
        """Return tuple of exceptions to retry on."""
        exceptions = []
        if HAS_OPENAI:
            exceptions.extend([
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.InternalServerError,
                openai.APITimeoutError,
            ])
        if HAS_ANTHROPIC:
            exceptions.extend([
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
                anthropic.APITimeoutError,
            ])
        return tuple(exceptions)

    def complete(self, prompt: str, system: str = "You are a helpful assistant.") -> str:
        """
        Get completion from LLM with retry logic.
        
        Args:
            prompt: User prompt
            system: System prompt (optional)
            
        Returns:
            str: Model response text
        """
        retry_exceptions = self._get_retry_exceptions()
        
        for attempt in Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(retry_exceptions),
            reraise=True
        ):
            with attempt:
                if self.provider == "openai":
                    return self._complete_openai(prompt, system)
                elif self.provider == "openrouter":
                    return self._complete_openrouter(prompt, system)
                elif self.provider == "anthropic":
                    return self._complete_anthropic(prompt, system)
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")
        return ""

    def _complete_openai(self, prompt: str, system: str) -> str:
        """OpenAI API completion."""
        if not HAS_OPENAI:
             raise ImportError("openai package missing")

        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

        # Build request kwargs, only include max_tokens if explicitly set
        request_kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            request_kwargs["max_tokens"] = self.max_tokens

        response = client.chat.completions.create(**request_kwargs)

        return response.choices[0].message.content or ""

    def _complete_openrouter(self, prompt: str, system: str) -> str:
        """OpenRouter API completion (uses OpenAI client)."""
        if not HAS_OPENAI:
             raise ImportError("openai package missing")

        base_url = self.base_url or "https://openrouter.ai/api/v1"
        client = openai.OpenAI(api_key=self.api_key, base_url=base_url, timeout=self.timeout)

        extra_headers = {
            "HTTP-Referer": "https://github.com/OpenHands/benchmarks",
            "X-Title": "OpenHands Benchmarks CAWM",
        }

        # Build request kwargs, only include max_tokens if explicitly set
        request_kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "extra_headers": extra_headers,
        }
        if self.max_tokens is not None:
            request_kwargs["max_tokens"] = self.max_tokens

        # Add provider routing for Kimi K2 models
        # Priority: Groq > Moonshot (Groq has better throughput)
        # allow_fallbacks=True to use other providers if these are unavailable
        if "kimi-k2" in self.model.lower():
            request_kwargs["extra_body"] = {
                "provider": {
                    "order": ["groq", "moonshotai"],
                    "allow_fallbacks": True,
                }
            }

        response = client.chat.completions.create(**request_kwargs)

        return response.choices[0].message.content or ""

    def _complete_anthropic(self, prompt: str, system: str) -> str:
        """Anthropic API completion."""
        if not HAS_ANTHROPIC:
             raise ImportError("anthropic package missing")

        client = anthropic.Anthropic(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

        # Anthropic requires max_tokens; use 8192 as default (high enough for most use cases)
        max_tokens = self.max_tokens if self.max_tokens is not None else 8192

        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        
        # Anthropic content is a list of blocks
        if response.content and len(response.content) > 0:
            block = response.content[0]
            if hasattr(block, "text"):
                return block.text
        return ""

    def parse_structured_response(self, response: str) -> Any:
        """
        Parse JSON from response, handling Markdown code blocks.
        """
        # Strip markdown code blocks if present
        clean_response = response.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        elif clean_response.startswith("```"):
            clean_response = clean_response[3:]
        
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]
            
        clean_response = clean_response.strip()
        
        try:
            return json.loads(clean_response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response}")
            raise