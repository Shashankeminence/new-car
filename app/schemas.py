from typing import List, Optional, Dict, Literal, Any, Union
from pydantic import BaseModel, Field, validator


class Message(BaseModel):
    role: Literal["system", "assistant", "user", "function", "tool", "developer"] = Field(
        description="Message role"
    )
    content: str = Field(description="Message content")


class ChatRequest(BaseModel):
    message: str = Field(description="The latest user message", min_length=1)
    context: Optional[List[Message]] = Field(default=None, description="Optional prior conversation messages")


class ElevenLabsMessage(BaseModel):
    role: str = Field(description="Message role")
    content: str = Field(description="Message content")


class ElevenLabsRequest(BaseModel):
    messages: List[ElevenLabsMessage] = Field(description="Conversation messages from ElevenLabs")
    model: Optional[str] = Field(default="gpt-4o-mini", description="Model name")
    max_tokens: Optional[int] = Field(default=8192, description="Max tokens")
    temperature: Optional[float] = Field(default=0.3, description="Temperature")
    stream: Optional[bool] = Field(default=False, description="Stream response")


class ChatResponse(BaseModel):
    reply: str = Field(description="Assistant reply text")


class ElevenLabsResponse(BaseModel):
    choices: List[Dict[str, Any]] = Field(description="Response choices in ElevenLabs format")


# Very flexible OpenAI-compatible schemas
class OpenAIMessage(BaseModel):
    role: str = Field(description="Message role")
    content: Union[str, List[Dict[str, Any]]] = Field(description="Message content")
    name: Optional[str] = Field(default=None)
    function_call: Optional[Dict[str, Any]] = Field(default=None)
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None)


class OpenAIRequest(BaseModel):
    messages: List[OpenAIMessage] = Field(description="Conversation messages")
    model: Optional[str] = Field(default="gpt-4o-mini")
    max_tokens: Optional[int] = Field(default=8192)
    temperature: Optional[float] = Field(default=0.3)
    stream: Optional[bool] = Field(default=False)
    # Allow any additional fields
    class Config:
        extra = "allow"


class OpenAIChoice(BaseModel):
    index: int = Field(default=0)
    message: OpenAIMessage
    finish_reason: str = Field(default="stop")
    logprobs: Optional[Dict[str, Any]] = Field(default=None)


class OpenAIUsage(BaseModel):
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)


class OpenAIResponse(BaseModel):
    id: str = Field(description="Unique identifier for the completion")
    object: str = Field(default="chat.completion")
    created: int = Field(description="Unix timestamp of creation")
    model: str = Field(description="Model used for completion")
    choices: List[OpenAIChoice] = Field(description="List of completion choices")
    usage: OpenAIUsage = Field(description="Token usage information")
    system_fingerprint: Optional[str] = Field(default=None)
