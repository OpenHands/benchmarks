# Stage 3: OpenRouter Provider Routing Fix

## Objective

Refactor the OpenRouter API calls in `CAWM/llm_client.py` to enforce provider routing for Kimi K2 0905 model. Only Groq or Moonshot AI providers should be used, with priority: Groq > Moonshot (Groq has better throughput).

## Background

OpenRouter supports specifying provider routing via the `provider` parameter. References:
- [OpenRouter Provider Routing](https://openrouter.ai/docs/features/provider-routing)
- [Groq Provider](https://openrouter.ai/provider/groq) - slug: `groq`
- [Moonshot AI Provider](https://openrouter.ai/provider/moonshotai) - slug: `moonshotai`

## Files to Modify

| File | Changes |
|------|---------|
| `CAWM/llm_client.py` | Modify `_complete_openrouter` method to add provider routing logic |

## Implementation Details

### Modify `_complete_openrouter` method (line 144-168)

**Current implementation**:
```python
def _complete_openrouter(self, prompt: str, system: str) -> str:
    ...
    response = client.chat.completions.create(
        model=self.model,
        messages=[...],
        temperature=self.temperature,
        max_tokens=self.max_tokens,
        extra_headers=extra_headers
    )
```

**New implementation**:
```python
def _complete_openrouter(self, prompt: str, system: str) -> str:
    ...
    # Build request kwargs
    request_kwargs = {
        "model": self.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": self.temperature,
        "max_tokens": self.max_tokens,
        "extra_headers": extra_headers,
    }

    # Add provider routing for Kimi K2 models
    if "kimi-k2" in self.model.lower():
        request_kwargs["extra_body"] = {
            "provider": {
                "order": ["groq", "moonshotai"],  # Groq first (better throughput)
                "allow_fallbacks": False,  # Only use these providers
            }
        }

    response = client.chat.completions.create(**request_kwargs)
```

### Provider Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `order` | `["groq", "moonshotai"]` | Priority order: Groq > Moonshot |
| `allow_fallbacks` | `False` | Do not fallback to other providers |

## Verification

1. Run existing tests to ensure no breakage
2. Manual API call test, check `x-openrouter-provider` response header to confirm correct provider is used
