import os
import time
import uuid
import json
import logging
import asyncio
from typing import List, Optional, Dict, Any, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError
from pydantic import ConfigDict

from .service import AgentService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ai_agent.log')
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

print("🚀 Starting AI Agent Server...")
print(f"📁 Working directory: {os.getcwd()}")
print(f"🌍 Environment variables loaded: {os.getenv('OPENAI_API_KEY', 'NOT_SET')[:10]}...")

app = FastAPI(title="AI Agent")

# Enable CORS for local dev and any allowed origins from env
allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "*")
print(f"🌐 CORS allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allowed_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_service: Optional[AgentService] = None

def get_agent_service() -> AgentService:
    global agent_service
    if agent_service is None:
        print("🔧 Initializing AgentService...")
        agent_service = AgentService()
        print("✅ AgentService initialized successfully")
    return agent_service

@app.get("/health")
async def health() -> Dict[str, str]:
    print("🏥 Health check requested")
    return {"status": "ok"}

@app.get("/v1")
async def v1_root() -> Dict[str, str]:
    return {"status": "ok", "message": "Custom LLM base path"}

# ElevenLabs compatible endpoint - following their exact documentation pattern
class Message(BaseModel):
    role: str = Field(description="The role of the message sender", example="user")
    content: Union[str, List[Dict[str, Any]]] = Field(
        description="The content of the message. Can be a string or list of content parts per OpenAI format",
        example="Hello, I want to rent a car",
    )

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    messages: List[Message] = Field(description="List of conversation messages")
    model: str = Field(description="The model to use", example="gpt-4o-mini")
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature", example=0.7)
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate", example=1000)
    stream: Optional[bool] = Field(default=False, description="Whether to stream the response", example=False)
    user_id: Optional[str] = Field(default=None, description="User identifier", example="user123")
    tools: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of tools available to the model")
    # ElevenLabs specific parameter
    elevenlabs_extra_body: Optional[Dict[str, Any]] = Field(default=None, description="ElevenLabs extra parameters")

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """ElevenLabs compatible endpoint with proper streaming and non-streaming support"""
    request_id = str(uuid.uuid4())[:8]

    def extract_text(content: Union[str, List[Dict[str, Any]]]) -> str:
        if isinstance(content, str):
            return content
        # content is a list of parts; concatenate text-bearing parts
        parts: List[str] = []
        for part in content:
            part_type = part.get("type")
            if part_type in {"text", "input_text"}:
                # OpenAI-style: {"type":"text","text":"..."} or {"type":"input_text","text":"..."}
                text_value = part.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
            elif part_type == "tool_result":
                # Some models may echo tool results as text
                tool_text = part.get("content") or part.get("text")
                if isinstance(tool_text, str):
                    parts.append(tool_text)
        return "".join(parts).strip()

    # Enhanced logging for incoming requests
    print(f"\n" + "="*80)
    print(f"🎯 [{request_id}] NEW REQUEST RECEIVED")
    print(f"="*80)
    print(f"📊 [{request_id}] Request details:")
    print(f"   - Model: {request.model}")
    print(f"   - Stream: {request.stream}")
    print(f"   - Messages count: {len(request.messages)}")
    print(f"   - Temperature: {request.temperature}")
    print(f"   - Max tokens: {request.max_tokens}")
    print(f"   - User ID: {request.user_id}")
    print(f"   - ElevenLabs extra body: {request.elevenlabs_extra_body}")

    # Log all messages in detail
    print(f"\n📝 [{request_id}] DETAILED MESSAGE BREAKDOWN:")
    print(f"-" * 50)
    for i, msg in enumerate(request.messages):
        content_preview = extract_text(msg.content)
        print(f"   Message {i+1}:")
        print(f"   ├─ Role: '{msg.role}'")
        print(f"   ├─ Content: '{content_preview}'")
        print(f"   └─ Length: {len(content_preview)} characters")
        print()

    try:
        # Extract the latest user message - be more flexible
        user_message = ""
        context_messages: List[Dict[str, str]] = []

        print(f"💬 [{request_id}] Processing messages:")
        for i, msg in enumerate(request.messages):
            normalized_content = extract_text(msg.content)
            print(f"   [{i+1}] Role: '{msg.role}', Content: '{normalized_content[:100]}{'...' if len(normalized_content) > 100 else ''}'")

            if msg.role == "user":
                user_message = normalized_content
            elif msg.role in ["assistant", "system"]:
                context_messages.append({"role": msg.role, "content": normalized_content})

        print(f"\n🎯 [{request_id}] EXTRACTED USER MESSAGE:")
        print(f"   '{user_message}'")
        print(f"   Length: {len(user_message)} characters")

        # If no user message found, try to get the last message regardless of role
        if not user_message and request.messages:
            last_message = request.messages[-1]
            user_message = extract_text(last_message.content)
            print(f"🔄 [{request_id}] Using last message as user message: '{user_message}'")

        if not user_message:
            print(f"❌ [{request_id}] No user message found in any message")
            raise HTTPException(status_code=400, detail="No user message found")

        # Generate response data
        response_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        current_time = int(time.time())

        print(f"\n🤖 [{request_id}] CALLING AGENT SERVICE...")
        print(f"   - Context messages: {len(context_messages)}")
        print(f"   - ElevenLabs extra body: {request.elevenlabs_extra_body}")

        # Get response from agent with proper error handling
        try:
            reply = get_agent_service().chat(user_message, context_messages, request.elevenlabs_extra_body)

            print(f"\n✅ [{request_id}] AGENT RESPONSE RECEIVED:")
            print(f"="*60)
            print(f"📝 Response: '{reply}'")
            print(f"📊 Response stats:")
            print(f"   - Length: {len(reply)} characters")
            print(f"   - Word count: {len(reply.split())} words")
            print(f"   - Lines: {len(reply.splitlines())} lines")
            print(f"="*60)

            # Log to file as well
            logger.info(f"[{request_id}] USER QUESTION: {user_message}")
            logger.info(f"[{request_id}] AGENT RESPONSE: {reply}")

        except Exception as e:
            print(f"⚠️ [{request_id}] Agent service error: {e}")
            logger.error(f"Agent service error for request {request_id}: {e}")
            # Fallback response for ElevenLabs
            reply = "I'm Alex from MyCarCar. I can help you with car rentals and sales. How can I assist you today?"
            print(f"🔄 [{request_id}] Using fallback response: '{reply}'")

        # Ensure reply is not empty
        if not reply or not reply.strip():
            reply = "I'm Alex from MyCarCar. I can help you with car rentals and sales. How can I assist you today?"
            print(f"🔄 [{request_id}] Using empty reply fallback: '{reply}'")

        # Create the final response
        final_response = {
            "id": response_id,
            "object": "chat.completion",
            "created": current_time,
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": reply
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": max(1, len(user_message.split())),
                "completion_tokens": max(1, len(reply.split())),
                "total_tokens": max(2, len(user_message.split()) + len(reply.split()))
            }
        }

        print(f"\n📤 [{request_id}] FINAL RESPONSE PREPARED:")
        print(f"   - Response ID: {response_id}")
        print(f"   - Reply length: {len(reply)} characters")
        print(f"   - Usage tokens: {final_response['usage']}")
        print(f"   - Stream mode: {request.stream}")

        # Return streaming response if requested
        if request.stream:
            print(f"\n🌊 [{request_id}] RETURNING STREAMING RESPONSE")
            print(f"="*60)

            async def event_stream():
                try:
                    # Send initial buffer chunk while processing (like ElevenLabs example)
                    initial_chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": current_time,
                        "model": request.model,
                        "choices": [{
                            "delta": {"content": ""},
                            "index": 0,
                            "finish_reason": None
                        }]
                    }
                    print(f"📡 [{request_id}] Sending initial chunk")
                    yield f"data: {json.dumps(initial_chunk)}\n\n"

                    # Simulate streaming by sending the response in chunks
                    words = reply.split()
                    chunk_size = max(1, len(words) // 3)  # Split into 3 chunks
                    print(f"📦 [{request_id}] Splitting into {len(words)} words, chunk size: {chunk_size}")

                    for i in range(0, len(words), chunk_size):
                        chunk_words = words[i:i + chunk_size]
                        chunk_content = " ".join(chunk_words)
                        if i + chunk_size < len(words):
                            chunk_content += " "

                        chunk_data = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": current_time,
                            "model": request.model,
                            "choices": [{
                                "delta": {"content": chunk_content},
                                "index": 0,
                                "finish_reason": None
                            }]
                        }
                        print(f"📡 [{request_id}] Sending chunk {i//chunk_size + 1}: '{chunk_content[:50]}{'...' if len(chunk_content) > 50 else ''}'")
                        yield f"data: {json.dumps(chunk_data)}\n\n"

                        # Small delay to simulate streaming
                        await asyncio.sleep(0.05)  # Reduced delay for faster response

                    # Send final chunk with finish_reason
                    final_chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": current_time,
                        "model": request.model,
                        "choices": [{
                            "delta": {},
                            "index": 0,
                            "finish_reason": "stop"
                        }]
                    }
                    print(f"🏁 [{request_id}] Sending final chunk")
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    print(f"✅ [{request_id}] Streaming completed")

                except Exception as e:
                    print(f"❌ [{request_id}] Streaming error: {e}")
                    logger.error("An error occurred during streaming: %s", str(e))
                    yield f"data: {json.dumps({'error': 'Internal error occurred during streaming!'})}\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")
        else:
            # Return regular JSON response for non-streaming
            print(f"\n📄 [{request_id}] RETURNING JSON RESPONSE")
            print(f"="*60)
            return JSONResponse(content=final_response)

    except HTTPException:
        print(f"❌ [{request_id}] HTTP Exception raised")
        raise
    except Exception as exc:
        print(f"💥 [{request_id}] Exception occurred: {exc}")
        logger.error(f"Exception in chat completion for request {request_id}: {exc}")
        import traceback
        traceback.print_exc()
        # Return a proper error response for ElevenLabs
        error_response = {
            "id": f"chatcmpl-error-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model if hasattr(request, 'model') else "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I'm Alex from MyCarCar. I can help you with car rentals and sales. How can I assist you today?"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 20,
                "total_tokens": 21
            }
        }
        print(f"🔄 [{request_id}] Returning error response")
        return JSONResponse(content=error_response, status_code=200)

@app.post("/debug")
async def debug_endpoint(request: Request):
    """Debug endpoint to see what ElevenLabs is actually sending"""
    print("\n🔍 DEBUG ENDPOINT CALLED")
    try:
        body = await request.body()
        print(f"📦 Raw body: {body.decode('utf-8')}")

        import json
        try:
            data = json.loads(body)
            print(f"📋 Parsed JSON: {data}")
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            return {"error": "Invalid JSON", "raw_body": body.decode('utf-8')}

        return {"status": "received", "data": data}
    except Exception as e:
        print(f"💥 DEBUG ENDPOINT Exception: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    print("🚀 Starting server directly...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
