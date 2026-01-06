"""
Ollama Client

Async client for local Ollama LLM integration.
Supports streaming, retries, and connection pooling.
"""

import asyncio
from typing import Any, AsyncGenerator, Optional

import httpx

from config import get_settings


class OllamaClient:
    """
    Async client for Ollama API.
    
    Features:
    - Connection pooling
    - Automatic retries
    - Streaming support
    - Error handling
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[httpx.AsyncClient] = None
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.settings.ollama.base_url,
                timeout=httpx.Timeout(self.settings.ollama.timeout),
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system: Optional system prompt
            model: Model name (uses default if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        if model is None:
            model = self.settings.ollama.model
        
        client = await self.get_client()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if system:
            payload["system"] = system
        
        # Retry logic
        last_error = None
        for attempt in range(self.settings.ollama.max_retries):
            try:
                response = await client.post("/api/generate", json=payload)
                response.raise_for_status()
                
                data = response.json()
                return data.get("response", "")
                
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code >= 500:
                    # Server error, retry
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            except httpx.TimeoutException as e:
                last_error = e
                await asyncio.sleep(2 ** attempt)
                continue
            except Exception as e:
                last_error = e
                break
        
        raise RuntimeError(f"Ollama request failed after retries: {last_error}")
    
    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response from the LLM.
        
        Yields:
            Response chunks as they arrive
        """
        if model is None:
            model = self.settings.ollama.model
        
        client = await self.get_client()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
            }
        }
        
        if system:
            payload["system"] = system
        
        async with client.stream("POST", "/api/generate", json=payload) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line:
                    import json
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
    
    async def chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        Chat completion with message history.
        
        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": "..."}
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Assistant response
        """
        if model is None:
            model = self.settings.ollama.model
        
        client = await self.get_client()
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        response = await client.post("/api/chat", json=payload)
        response.raise_for_status()
        
        data = response.json()
        return data.get("message", {}).get("content", "")
    
    async def list_models(self) -> list[str]:
        """List available models."""
        client = await self.get_client()
        
        try:
            response = await client.get("/api/tags")
            response.raise_for_status()
            
            data = response.json()
            models = data.get("models", [])
            return [m.get("name", "") for m in models]
        except Exception:
            return []
    
    async def is_available(self) -> bool:
        """Check if Ollama is running and responsive."""
        try:
            client = await self.get_client()
            response = await client.get("/")
            return response.status_code == 200
        except Exception:
            return False


# Global instance
ollama_client = OllamaClient()
