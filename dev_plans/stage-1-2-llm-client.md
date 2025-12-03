# Stage 1-2: LLM客户端 (CAWM/llm_client.py)

## 目标
创建OpenRouter兼容的LLM客户端，供HierarchicalSummarization和InductionModule使用。

## 文件路径
`/Users/tangyiq/dev/benchmarks/CAWM/llm_client.py`

---

## 类设计

### LLMClient
```python
class LLMClient:
    """
    OpenRouter兼容的LLM客户端

    支持的provider:
    - openrouter: OpenRouter API (默认)
    - openai: OpenAI API
    - anthropic: Anthropic API

    Usage:
        client = LLMClient(provider="openrouter", model="anthropic/claude-3.5-sonnet")
        response = client.complete("What is 2+2?")
    """

    def __init__(
        self,
        provider: str = "openrouter",
        model: str = "anthropic/claude-3.5-sonnet",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 120,
        **kwargs
    ):
        """
        初始化LLM客户端

        Args:
            provider: API提供商 ("openrouter", "openai", "anthropic")
            model: 模型名称
            api_key: API密钥 (可从环境变量获取)
            base_url: API基础URL (可选覆盖)
            temperature: 采样温度
            max_tokens: 最大生成token数
            timeout: 请求超时秒数
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # 解析API key
        self.api_key = self._resolve_api_key(api_key)

        # 解析base URL
        self.base_url = self._resolve_base_url(base_url)

        # 延迟初始化HTTP客户端
        self._client = None

    def _resolve_api_key(self, api_key: Optional[str]) -> str:
        """从参数或环境变量获取API key"""
        if api_key:
            return api_key

        env_vars = {
            "openrouter": "OPENROUTER_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY"
        }
        env_var = env_vars.get(self.provider)
        if env_var:
            key = os.getenv(env_var)
            if key:
                return key

        raise ValueError(f"API key not found. Set {env_var} environment variable or pass api_key parameter.")

    def _resolve_base_url(self, base_url: Optional[str]) -> str:
        """解析API base URL"""
        if base_url:
            return base_url

        default_urls = {
            "openrouter": "https://openrouter.ai/api/v1",
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com"
        }
        return default_urls.get(self.provider, "")

    def _init_client(self):
        """延迟初始化HTTP客户端"""
        if self._client is not None:
            return

        if self.provider in ["openrouter", "openai"]:
            import openai
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=self.api_key,
                timeout=self.timeout
            )

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        获取LLM completion

        Args:
            prompt: 用户提示
            system: 系统提示 (可选)
            temperature: 覆盖默认温度 (可选)
            max_tokens: 覆盖默认max_tokens (可选)

        Returns:
            LLM生成的文本
        """
        self._init_client()

        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider in ["openrouter", "openai"]:
            return self._complete_openai_style(prompt, system, temp, tokens)
        elif self.provider == "anthropic":
            return self._complete_anthropic(prompt, system, temp, tokens)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _complete_openai_style(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """OpenAI/OpenRouter风格的completion"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        content = response.choices[0].message.content
        return content if content else ""

    def _complete_anthropic(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Anthropic风格的completion"""
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
        if system:
            kwargs["system"] = system
        if temperature > 0:
            kwargs["temperature"] = temperature

        response = self._client.messages.create(**kwargs)

        # 获取第一个content block的文本
        if response.content and hasattr(response.content[0], "text"):
            return response.content[0].text
        return ""

    def complete_with_retry(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> str:
        """带重试的completion"""
        import time

        last_error = None
        for attempt in range(max_retries):
            try:
                return self.complete(prompt, system)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))

        raise last_error
```

---

## 辅助函数

### 响应解析
```python
def parse_workflow_blocks(response: str) -> List[Dict[str, Any]]:
    """
    解析LLM返回的workflow block格式

    格式:
    WORKFLOW: Name
    CATEGORY: category
    DESCRIPTION: description

    STEP 1:
    ENV: ...
    REASONING: ...
    ACTION: ...
    ACTION_TYPE: ...

    ---
    """
    blocks = []
    # 按 --- 分割
    raw_blocks = response.split("---")

    for block in raw_blocks:
        if "WORKFLOW:" not in block.upper():
            continue

        parsed = _parse_single_block(block)
        if parsed:
            blocks.append(parsed)

    return blocks

def _parse_single_block(block: str) -> Optional[Dict[str, Any]]:
    """解析单个workflow block"""
    lines = block.strip().split("\n")

    result = {
        "name": "",
        "category": "general",
        "description": "",
        "steps": []
    }

    # 解析header字段
    for line in lines:
        line_upper = line.strip().upper()
        if line_upper.startswith("WORKFLOW:"):
            result["name"] = line.split(":", 1)[1].strip()
        elif line_upper.startswith("CATEGORY:"):
            result["category"] = line.split(":", 1)[1].strip().lower()
        elif line_upper.startswith("DESCRIPTION:"):
            result["description"] = line.split(":", 1)[1].strip()

    # 解析steps
    steps = _parse_steps(block)
    result["steps"] = steps

    if not result["name"] or len(steps) < 2:
        return None

    return result

def _parse_steps(block: str) -> List[Dict[str, str]]:
    """解析步骤"""
    import re

    steps = []
    step_matches = list(re.finditer(r"STEP\s*(\d+):", block, re.IGNORECASE))

    for i, match in enumerate(step_matches):
        start = match.end()
        end = step_matches[i + 1].start() if i + 1 < len(step_matches) else len(block)

        content = block[start:end]
        step = {
            "env_description": "",
            "reasoning": "",
            "action": "",
            "action_type": "terminal"
        }

        for line in content.split("\n"):
            line_stripped = line.strip()
            line_upper = line_stripped.upper()

            if line_upper.startswith("ENV:"):
                step["env_description"] = line_stripped.split(":", 1)[1].strip()
            elif line_upper.startswith("REASONING:"):
                step["reasoning"] = line_stripped.split(":", 1)[1].strip()
            elif line_upper.startswith("ACTION:"):
                step["action"] = line_stripped.split(":", 1)[1].strip()
            elif line_upper.startswith("ACTION_TYPE:"):
                step["action_type"] = line_stripped.split(":", 1)[1].strip().lower()

        if step["action"]:
            steps.append(step)

    return steps
```

---

## 配置类

```python
@dataclass
class LLMConfig:
    """LLM配置"""
    provider: str = "openrouter"
    model: str = "anthropic/claude-3.5-sonnet"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: int = 120

    def to_client(self) -> LLMClient:
        """创建LLMClient实例"""
        return LLMClient(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout
        )
```

---

## 依赖
- `openai` (可选, 用于openrouter/openai provider)
- `anthropic` (可选, 用于anthropic provider)

## 环境变量
- `OPENROUTER_API_KEY`: OpenRouter API密钥
- `OPENAI_API_KEY`: OpenAI API密钥
- `ANTHROPIC_API_KEY`: Anthropic API密钥

## 测试要点
1. 各provider的API调用正常
2. 环境变量fallback正确
3. 重试机制工作正常
4. `parse_workflow_blocks()` 正确解析LLM输出
