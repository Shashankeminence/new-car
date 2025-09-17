import os
import random
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI
from .prompts import get_agent_system_prompt

# Configure logging
logger = logging.getLogger(__name__)

class AgentService:
    def __init__(self, openai_api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        print("🔧 Initializing AgentService...")
        
        mock_env = os.getenv("AGENT_MOCK", "").strip().lower()
        self.mock: bool = mock_env in {"1", "true", "yes", "on"}
        print(f"🎭 Mock mode: {self.mock}")

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

    def chat(self, user_message: str, context_messages: Optional[List[Dict[str, str]]] = None, elevenlabs_extra_body: Optional[Dict[str, Any]] = None) -> str:
        print(f"\n💬 AgentService.chat() called")
        print(f"📝 User message: '{user_message}'")
        print(f"📚 Context messages: {len(context_messages) if context_messages else 0}")
        print(f"🔧 ElevenLabs extra body: {elevenlabs_extra_body}")
        
        if self.mock:
            print("🎭 Using mock response")
            response = self._get_mock_response(user_message)
            print(f"🎭 Mock response: '{response[:100]}{'...' if len(response) > 100 else ''}'")
            return response
        
        try:
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": get_agent_system_prompt()},
            ]
            if context_messages:
                messages.extend(context_messages)
            messages.append({"role": "user", "content": user_message})

            print(f"📋 Prepared messages for OpenAI:")
            for i, msg in enumerate(messages):
                print(f"   [{i+1}] {msg['role']}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")

            # Log ElevenLabs extra body for debugging
            if elevenlabs_extra_body:
                print(f"🔧 ElevenLabs extra body: {elevenlabs_extra_body}")

            print("🚀 Calling OpenAI API...")
            # Call OpenAI API
            response = self.client.chat.completions.create(  # type: ignore
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=500,  # Limit tokens for faster response
            )
            content = response.choices[0].message.content if response.choices else ""
            print(f"✅ OpenAI response received: '{content[:100]}{'...' if len(content) > 100 else ''}'")
            logger.info(f"OpenAI response received: {content[:100]}...")
            
            if not content:
                print("⚠️ Empty response from OpenAI, using fallback")
                return self._get_fallback_response(user_message)
            
            return content
            
        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            logger.error(f"OpenAI API error: {e}")
            return self._get_fallback_response(user_message)

    def _get_fallback_response(self, user_message: str) -> str:
        """Fallback response when OpenAI API fails"""
        print(f"🔄 Using fallback response for: '{user_message[:50]}{'...' if len(user_message) > 50 else ''}'")
        
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in ["rent", "rental", "renting"]):
            response = "I can help you with car rentals. What type of vehicle are you looking for and what dates do you need it?"
        elif any(word in message_lower for word in ["buy", "purchase", "buying"]):
            response = "I'd be happy to help you find a car to purchase. What's your budget range and what type of vehicle interests you?"
        else:
            response = "Hi! I'm Alex from MyCarCar. I can help you with car rentals and sales. How can I assist you today?"
        
        print(f"🔄 Fallback response: '{response}'")
        return response

    def _get_mock_response(self, user_message: str) -> str:
        """Generate contextual mock responses based on user input"""
        print(f"🎭 Generating mock response for: '{user_message[:50]}{'...' if len(user_message) > 50 else ''}'")
        
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
            responses = [
                "I'm Alex from MyCarCar and I can help with rentals or car sales. Tell me what you're looking for and your city and dates if renting. I'll find clear options and next steps for you.",
                "Hi! I can assist you with car rentals or purchases. What type of vehicle are you interested in and when do you need it?",
                "Hello! I'm here to help you find the perfect car. Are you looking to rent for a trip or buy a vehicle? Let me know your preferences and I'll guide you through the options."
            ]
            response = random.choice(responses)
        
        print(f"🎭 Mock response generated: '{response[:100]}{'...' if len(response) > 100 else ''}'")
        return response
