"""Deepseek"""

from types import SimpleNamespace

import httpx


class DeepSeekLLM:
    """Deepseek в качестве LLM-модели"""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        num_output: int = 1024,
        timeout_sec: float = 30.0,
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._num_output = int(num_output)
        self._timeout_sec = float(timeout_sec)

    async def achat(self, messages) -> SimpleNamespace:
        payload_messages = []
        for m in messages:
            role = getattr(getattr(m, "role", None), "value", None) or str(
                getattr(m, "role", "user")
            )
            content = getattr(m, "content", "")
            payload_messages.append({"role": str(role).lower(), "content": str(content)})

        payload = {
            "model": self._model_name,
            "messages": payload_messages,
            "max_tokens": self._num_output,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self._timeout_sec) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return SimpleNamespace(message=SimpleNamespace(content=content))
