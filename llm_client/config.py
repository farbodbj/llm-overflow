import abc
from typing import Tuple, Optional, Dict, Any
from pydantic import BaseModel

class RequestConfig(BaseModel):
    """The configuration for a request to the LLM API.

    Args:
        model: The model to use.
        system_prompt: The system prompt to provide to the LLM API.
        user_prompt: The user prompt to provide to the LLM API.
        metadata: Additional metadata to attach to the request for logging or validation purposes.
    """

    model: str
    system_prompt: str
    user_prompt: str
    metadata: Optional[Dict[str, Any]] = None