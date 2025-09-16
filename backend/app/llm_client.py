import os
import json
import logging
import openai
from groq import Groq
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClient:
    """
    LLM client for generating call summaries and context
    supports OpenAI, Groq, and other providers
    """

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")

        # Initialize clients based on provider
        if self.provider == "openai" and self.openai_key:
            self.openai_client = openai.OpenAI(api_key=self.openai_key)
        elif self.provider == "groq" and self.groq_key:
            self.groq_client = Groq(api_key=self.groq_key)
            logger.info("Initialized Groq client")
        else:
            raise ValueError(f"No valid API key found for provider: {self.provider}")
        
    async def generate_call_summary(
        self,
        conversation_history: List[Dict[str, Any]],
        caller_info: Optional[Dict[str, Any]] = None,
        summary_type: str = "transfer"
    ) -> str:
        """
        Generate AI-powered call summary for warm transfer

        Args:
            conversation_history: List of conversation exchanges
            caller_info: Information about the caller
            summary_type: Type of summary (transfer, detailed, brief)

        Returns:
            Generated call summary string
        """

        try:
            # Prepare context
            context = self._prepare_context(conversation_history, caller_info)

            # Generate summary based on provider
            if self.provider == "openai":
                summary = await self._generate_openai_summary(context, summary_type)
            elif self.provider == "groq":
                summary = await self._generate_groq_summary(context, summary_type)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            logger.info(f"Generated {summary_type} summary using {self.provider}")
            return summary
        
        except Exception as e:
            logger.error(f"Failed to generate call summary: {str(e)}")
            raise Exception(f"Summary generation failed: {str(e)}")
    
    def _prepare_context(
        self,
        conversation_history: List[Dict[str, Any]],
        caller_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Prepare conversation context for LLM
        """

        # Build context string
        context_parts = ["CALL CONTEXT FOR WARM TRANSFER:\n"]

        # Add caller information
        if caller_info:
            context_parts.append("CALLER INFORMATION:")
            for key, value in caller_info.items():
                context_parts.append(f"- {key}: {value}")
            context_parts.append("")

        # Add conversation history
        if conversation_history:
            context_parts.append("CONVERSATION HISTORY:")
            for i, exchange in enumerate(conversation_history):
                timestamp = exchange.get("timestamp", "Unknown time")
                speaker = exchange.get("speaker", "Unknown")
                message = exchange.get("message", "")

                context_parts.append(f"[{timestamp}] {speaker}: {message}")
            context_parts.append("")

        context_parts.append(f"Summary generated at: {datetime.now(timezone.utc).isoformat()}")

        return "\n".join(context_parts)
    
    async def _generate_openai_summary(self, context: str, summary_type: str) -> str:
        """
        Generate summary using OpenAI
        """

        system_prompt = self._get_system_prompt(summary_type)

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                max_tokens=500,
                temperature=0.3,
                top_p=0.9
            )

            summary = response.choices[0].message.content.strip()
            return summary
        
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise Exception(f"OpenAI summary generation failed: {str(e)}")
        
    async def _generate_groq_summary(self, context: str, summary_type: str) -> str:
        """
        Generate summary using Groq
        """

        system_prompt = self._get_system_prompt(summary_type)

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                max_tokens=500,
                temperature=0.3,
                top_p=0.9
            )

            summary = response.choices[0].message.content.strip()
            return summary

        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            raise Exception(f"Groq summary generation failed: {str(e)}")
        
    def _get_system_prompt(self, summary_type: str) -> str:
        """
        Get system prompt based on summary type
        """

        base_instructions = """
            You are an AI assistant specialized in generating concise, actionable call summaries for warm transfers in customer service environments.
            Your task is to analyze the conversation and create a summary that will help the receiving agent (Agent B) understand the context and continue the conversation seamlessly.
        """

        if summary_type == "transfer":
            return base_instructions + """
                WARM TRANSFER SUMMARY FORMAT:

                Create a structured summary with following sections:

                1. CALLER PROFILE:
                    - Name and key identifying information
                    - Account/reference numbers if mentioned
                    - Contact preferences or constraints

                2. REASON FOR CALL:
                    - Primary issue or request
                    - Urgency level (Low/Medium/High)
                    - Category (Support, Sales, Billing, etc.)

                3. CONVERSATION HIGHLIGHTS:
                    - Key points discussed
                    - Solutions attempted by Agent A
                    - Customer reactions and responses
                
                4. CURRENT STATUS:
                    - Where the conversation stands
                    - What has been resolved/unresolved
                    - Next steps needed
                
                5. TRANSFER CONTEXT
                    - Why the transfer is occuring
                    - What Agent B should focus on
                    - Any sensitive information to be aware of 

                Keep the summary concise but comprehensive. Use bullet points for clarity. Limit to 300-400 words total.
            """
        elif summary_type == "detailed":
            return base_instructions + """
                DETAILED CALL SUMMARY FORMAT:

                Provide a comprehensie analysis including:
                - Complete conversation timeline
                - All topics discussed
                - Customer sentiment analysis
                - Technical details and specifications
                - Follow-up requirements
                - Potential upsell/cross-sell opportunities

                Be thorough but organized. Use clear headings and bullet points.
            """
        elif summary_type == "brief":
            return base_instructions + """
                BRIEF TRANSFER SUMMARY FORMAT:

                Create a concise 2-3 sentence summary convering:
                1. Who is calling and why
                2. What has been discussed/attempted
                3. What Agent B needs to do next

                Keep it under 100 words, focus on actionable information only.
            """
        
        return base_instructions + "\nProvide a clear, professional call summary suitable for agent handoff."
    
    async def generate_context_questions(
        self,
        summary: str,
        caller_info: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Generated suggested questions for Agent B based on call summary

        Args:
            summary: Generated call summary
            caller_info: Additional caller information

        Returns:
            List of suggested questions
        """

        try:
            system_prompt = """
                You are an AI assistant helping generate relevant follow-up questions for customer service agents.

                Based on the call summary provided, generate 3-5 intelligent questions that Agent B should consider asking to:
                1. Clarify remaining issues
                2. Gather additional needed information
                3. Confirm customer understanding 
                4. Move toward resolution

                Format as a simple list of questions, each starting with "- "
            """

            context = f"CALL SUMMARY:\n{summary}\n\n"
            if caller_info:
                context += f"CALLER INFO:\n{json.dumps(caller_info, indent=2)}"
            
            if self.provider == "openai":
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": context}
                    ],
                    max_tokens=200,
                    temperature=0.4
                )

                questions_text = response.choices[0].message.content.strip()
            
            elif self.provider == "groq":
                response = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": context}
                    ],
                    max_tokens=200,
                    temperature=0.4
                )

                questions_text = response.choices[0].message.content.strip()

            # Parse questions from response
            questions = []
            for line in questions_text.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    questions.append(line[2:])
                elif line and not line.startswith("#") and "?" in line:
                    questions.append(line)
            
            return questions[:5]
        
        except Exception as e:
            logger.error(f"Failed to generate context questions: {str(e)}")
            return [
                "Can you confirm your account information?",
                "What is the main issue you need help with today?",
                "Have you tried any troubleshooting steps already?"
            ]
    
    async def analyze_sentiment(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze customer sentiment from conversation

        Args:
            conversation_history: List of conversation exchanges

        Returns:
            Sentiment analysis results
        """

        try:
            # Extract customer messages
            customer_messages = []
            for exchange in conversation_history:
                if exchange.get("speaker", "").lower() in ["caller", "customer", "user"]:
                    customer_messages.append(exchange.get("message", ""))
            
            if not customer_messages:
                return {"sentiment": "neutral", "confidence": 0.0, "analysis": "No customer messages found"}
            
            combined_text = " ".join(customer_messages)

            system_prompt = """
                Analyze the customer sentiment from the provided conversation messages.

                Return a JSON response with:
                - sentiment: "positive", "negative", or "neural"
                - confidence: float between 0.0 and 1.0
                - analysis: brief explanation of the sentiment analysis
                - key_emotions: list of detected emotions
            """

            if self.provider == "openai":
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Customer messages: {combined_text}"}
                    ],
                    max_tokens=200,
                    temperature=0.1
                )

                result_text = response.choices[0].message.content.strip()

            elif self.provider == "groq":
                response = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Customer messages: {combined_text}"}
                    ],
                    max_tokens=200,
                    temperature=0.1
                )

                result_text = response.choices[0].message.content.strip()

            # Try to parse JSON response
            try:
                return json.loads(result_text)
            except json.JSONDecodeError:
                # Fallback to basic analysis
                if any(word in combined_text.lower() for word in ['angry', 'frustrated', 'upset', 'horrible']):
                    sentiment = "negative"
                elif any(word in combined_text.lower() for word in ['thank', 'great', 'excellent', 'happy']):
                    sentiment = "positive"
                else:
                    sentiment = "neutral"
                
                return {
                    "sentiment": sentiment,
                    "confidence": 0.6,
                    "analysis": "Basic keyword-based sentiment analysis",
                    "key_emotions": []
                }
        
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "analysis": f"Analysis failed: {str(e)}",
                "key_emotions": []
            }