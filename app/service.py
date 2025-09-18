import os
import random
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI
from .prompts import get_agent_system_prompt

# Configure logging
logger = logging.getLogger(__name__)

CAR_KEYWORDS = {
    "car", "cars", "rental", "rentals", "rent", "booking", "book", "vehicle", "vehicles",
    "sedan", "suv", "pickup", "insurance", "roadside", "sale", "sales", "buy", "purchase",
}

class AgentService:
    def __init__(self, openai_api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        print("🔧 Initializing AgentService...")
        
        mock_env = os.getenv("AGENT_MOCK", "").strip().lower()
        self.mock: bool = mock_env in {"1", "true", "yes", "on"}
        print(f"🎭 Mock mode: {self.mock}")

        # RAG configuration (optional)
        self.rag_enabled: bool = os.getenv("RAG_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
        self.rag_index_dir: str = os.getenv("RAG_INDEX_DIR", "./rag_index")
        self.rag_top_k: int = int(os.getenv("RAG_TOP_K", "5"))
        self.rag_strict: bool = os.getenv("RAG_STRICT", "1").strip().lower() in {"1", "true", "yes", "on"}
        self.rag_build_on_start: bool = os.getenv("RAG_BUILD_ON_START", "1").strip().lower() in {"1", "true", "yes", "on"}
        self.rag_source_file: str = os.getenv("RAG_SOURCE_FILE", "./sample.txt")
        self.rag = None
        if self.rag_enabled:
            try:
                from .rag import FaissRag  # lazy import
                self.rag = FaissRag(self.rag_index_dir)
                try:
                    self.rag.load()
                    print(f"📚 RAG enabled. Loaded index from: {self.rag_index_dir}")
                except Exception as load_err:
                    print(f"ℹ️ RAG index not found or failed to load: {load_err}")
                    if self.rag_build_on_start and os.path.exists(self.rag_source_file):
                        print(f"🧱 Building RAG index from source: {self.rag_source_file}")
                        self.rag.build_from_file(self.rag_source_file)
                        self.rag.load()
                        print(f"✅ RAG index built and loaded from: {self.rag_index_dir}")
                    else:
                        print("⚠️ RAG disabled for this run (no index and no build).")
                        self.rag = None
            except Exception as e:
                print(f"⚠️ RAG initialization failed: {e}. Continuing without RAG.")
                self.rag = None

        # Use OpenAI API key
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            # If no key is provided and mock is not explicitly disabled, enable mock mode
            if not self.mock:
                self.mock = True
                print("⚠️ No OpenAI API key found, enabling mock mode")
        else:
            print(f"🔑 OpenAI API key found: {api_key[:10]}...")
        
        if not self.mock:
            try:
                self.client = OpenAI(api_key=api_key)
                print("✅ OpenAI client initialized successfully")
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize OpenAI client: {e}")
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.mock = True
        else:
            self.client = None  # type: ignore
            print("🎭 Running in mock mode - no OpenAI client")
            
        # Use OpenAI model
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        print(f"🤖 Using model: {self.model}")
        print("✅ AgentService initialization complete")

    def _is_car_domain(self, text: str) -> bool:
        t = text.lower()
        return any(kw in t for kw in CAR_KEYWORDS)

    def chat(self, user_message: str, context_messages: Optional[List[Dict[str, str]]] = None, elevenlabs_extra_body: Optional[Dict[str, Any]] = None) -> str:
        print(f"\n" + "="*60)
        print(f"💬 AgentService.chat() called")
        print(f"="*60)
        print(f"📝 USER QUESTION:")
        print(f"   '{user_message}'")
        print(f"   Length: {len(user_message)} characters")
        print(f"   Word count: {len(user_message.split())} words")
        print(f"\n📚 CONTEXT MESSAGES: {len(context_messages) if context_messages else 0}")
        if context_messages:
            for i, ctx in enumerate(context_messages):
                print(f"   [{i+1}] {ctx['role']}: {ctx['content'][:100]}{'...' if len(ctx['content']) > 100 else ''}")
        
        print(f"\n🔧 ELEVENLABS EXTRA BODY: {elevenlabs_extra_body}")
        print(f"🔎 RAG enabled: {self.rag_enabled}, index: {self.rag_index_dir}, top_k: {self.rag_top_k}, strict: {self.rag_strict}")
        
        if self.mock:
            print(f"\n🎭 USING MOCK RESPONSE MODE")
            response = self._get_mock_response(user_message)
            logger.info(f"USER QUESTION: {user_message}")
            logger.info(f"MOCK RESPONSE: {response}")
            return response
        
        try:
            messages: List[Dict[str, str]] = []
            is_car = self._is_car_domain(user_message)
            use_rag = (self.rag_enabled and self.rag is not None) and not is_car

            if use_rag:
                try:
                    rag_context = self.rag.build_context(user_message, top_k=self.rag_top_k)
                    print("📥 RAG context built:")
                    print(rag_context[:800] + ("..." if len(rag_context) > 800 else ""))
                    if self.rag_strict and not rag_context.strip():
                        print("⚠️ RAG strict mode: no relevant context found")
                        return "There is no trained data available for this."

                    # Choose prompt style based on strictness
                    if self.rag_strict:
                        # Neutral, digits-allowed, strict-context prompt
                        rag_system = (
                            "You answer strictly and only using the provided CONTEXT. "
                            "If the answer is not contained in the context, respond exactly with: There is no trained data available for this.\n\n"
                            f"CONTEXT:\n{rag_context}"
                        )
                    else:
                        # Softer prompt: prefer context but allow concise general knowledge when needed
                        rag_system = (
                            "Use the following CONTEXT to answer the user's question. Prefer facts from the CONTEXT. "
                            "If some details are not explicitly covered, you may answer briefly using reasonable general knowledge and inference. "
                            "Be concise and accurate.\n\n"
                            f"CONTEXT:\n{rag_context}"
                        )
                    system_prompt = rag_system
                except Exception as e:
                    print(f"⚠️ RAG retrieval failed: {e}. Falling back to non-RAG domain handling.")
                    system_prompt = get_agent_system_prompt()
            else:
                # Car domain or RAG unavailable → use original persona
                system_prompt = get_agent_system_prompt()

            messages.append({"role": "system", "content": system_prompt})
            if context_messages:
                messages.extend(context_messages)
            messages.append({"role": "user", "content": user_message})

            print(f"\n📋 PREPARED MESSAGES FOR OPENAI:")
            print(f"-" * 50)
            for i, msg in enumerate(messages):
                print(f"   Message {i+1}:")
                print(f"   ├─ Role: '{msg['role']}'")
                print(f"   ├─ Content: '{msg['content'][:200]}{'...' if len(msg['content']) > 200 else ''}'")
                print(f"   └─ Length: {len(msg['content'])} characters")
                print()

            if elevenlabs_extra_body:
                print(f"🔧 ElevenLabs extra body: {elevenlabs_extra_body}")

            print(f"\n🚀 CALLING OPENAI API...")
            print(f"   - Model: {self.model}")
            print(f"   - Temperature: 0.3")
            print(f"   - Max tokens: 500")
            
            response = self.client.chat.completions.create(  # type: ignore
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=500,
            )
            content = response.choices[0].message.content if response.choices else ""
            
            print(f"\n✅ OPENAI API RESPONSE RECEIVED:")
            print(f"="*60)
            print(f"📝 Response: '{content}'")
            print(f"📊 Response stats:")
            print(f"   - Length: {len(content)} characters")
            print(f"   - Word count: {len(content.split())} words")
            print(f"   - Lines: {len(content.splitlines())} lines")
            print(f"   - Finish reason: {response.choices[0].finish_reason if response.choices else 'N/A'}")
            print(f"   - Usage: {response.usage.dict() if hasattr(response, 'usage') and response.usage else 'N/A'}")
            print(f"="*60)
            
            logger.info(f"USER QUESTION: {user_message}")
            logger.info(f"OPENAI RESPONSE: {content}")
            if hasattr(response, 'usage') and response.usage:
                logger.info(f"OPENAI USAGE: {response.usage.dict()}")
            
            if not content:
                print("⚠️ Empty response from OpenAI, using fallback")
                return self._get_fallback_response(user_message)
            
            return content
            
        except Exception as e:
            print(f"\n❌ OPENAI API ERROR:")
            print(f"="*60)
            print(f"Error: {e}")
            print(f"Error type: {type(e).__name__}")
            print(f"="*60)
            
            logger.error(f"OpenAI API error: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            
            return self._get_fallback_response(user_message)

    def _get_fallback_response(self, user_message: str) -> str:
        """Fallback response when OpenAI API fails"""
        print(f"\n🔄 USING FALLBACK RESPONSE")
        print(f"-" * 40)
        print(f"Original question: '{user_message[:50]}{'...' if len(user_message) > 50 else ''}'")
        
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in ["rent", "rental", "renting"]):
            response = "I can help you with car rentals. What type of vehicle are you looking for and what dates do you need it?"
        elif any(word in message_lower for word in ["buy", "purchase", "buying"]):
            response = "I'd be happy to help you find a car to purchase. What's your budget range and what type of vehicle interests you?"
        else:
            response = "There is no trained data available for this."
        
        print(f"Fallback response: '{response}'")
        print(f"Response length: {len(response)} characters")
        print(f"-" * 40)
        
        logger.info(f"FALLBACK - USER QUESTION: {user_message}")
        logger.info(f"FALLBACK - RESPONSE: {response}")
        
        return response

    def _get_mock_response(self, user_message: str) -> str:
        """Generate contextual mock responses based on user input"""
        print(f"\n🎭 GENERATING MOCK RESPONSE")
        print(f"-" * 40)
        print(f"User question: '{user_message[:50]}{'...' if len(user_message) > 50 else ''}'")
        
        message_lower = user_message.lower()
        
        # Car brand specific responses
        if "audi" in message_lower:
            response = (
                "Audi offers a smooth drive, premium interiors, and strong safety features. "
                "For rentals we can check options like compact sedans or sporty SUVs. "
                "Pricing varies by city, dates, and coverage; I can check availability now if you share dates and pickup location."
            )
        elif "bmw" in message_lower:
            response = (
                "BMW provides excellent performance and luxury features. "
                "We have various models available for rental from sporty coupes to spacious SUVs. "
                "What dates and location are you considering for your rental?"
            )
        elif "mercedes" in message_lower or "benz" in message_lower:
            response = (
                "Mercedes-Benz offers premium comfort and advanced technology. "
                "Our fleet includes sedans, SUVs, and luxury vehicles. "
                "Could you tell me your preferred dates and pickup location?"
            )
        elif "toyota" in message_lower:
            response = (
                "Toyota vehicles are known for reliability and fuel efficiency. "
                "We have options from compact cars to family SUVs. "
                "What type of vehicle and rental period are you looking for?"
            )
        
        # Intent-based responses
        elif any(word in message_lower for word in ["rent", "rental", "renting"]):
            response = (
                "Great! I can help you find the perfect rental car. "
                "What type of vehicle are you looking for and what dates do you need it? "
                "Also, which city will you be picking up from?"
            )
        elif any(word in message_lower for word in ["buy", "purchase", "buying"]):
            response = (
                "I'd be happy to help you find a car to purchase! "
                "What's your budget range and what type of vehicle interests you? "
                "Are you looking for new or pre-owned options?"
            )
        elif any(word in message_lower for word in ["price", "cost", "how much"]):
            response = (
                "Pricing depends on several factors like vehicle type, rental duration, and location. "
                "For rentals, we typically start around fifty dollars per day for economy cars. "
                "Could you share more details about what you're looking for?"
            )
        elif any(word in message_lower for word in ["available", "availability"]):
            response = (
                "I can check availability for you right away! "
                "Just let me know your preferred dates, pickup location, and vehicle type. "
                "I'll find the best options available for your needs."
            )
        elif any(word in message_lower for word in ["book", "booking", "reserve"]):
            response = (
                "Perfect! I can help you make a booking. "
                "I'll need your rental dates, pickup location, and vehicle preference. "
                "Do you have any specific requirements or preferences?"
            )
        
        # Greeting responses
        elif any(word in message_lower for word in ["hello", "hi", "hey"]):
            greetings = [
                "Hi there! I'm Alex from MyCarCar. How can I help you today?",
                "Hello! I'm here to assist with car rentals and sales. What can I do for you?",
                "Hi! Welcome to MyCarCar. Are you looking to rent a car or explore our sales options?"
            ]
            response = random.choice(greetings)
        
        # Default contextual response
        else:
            response = "There is no trained data available for this."
        
        print(f"Mock response: '{response[:100]}{'...' if len(response) > 100 else ''}'")
        print(f"Response length: {len(response)} characters")
        print(f"-" * 40)
        
        return response
