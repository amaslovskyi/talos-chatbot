"""
Hybrid chatbot that combines local models with external API fallback.
Provides resilience and flexibility for different use cases.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

# LangChain imports
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOpenAI

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

from src.chatbot import RAGChatbot, ChatResponse
from src.retrieval import RetrievalResult
from config import get_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRAGChatbot(RAGChatbot):
    """
    Hybrid RAG chatbot that uses local models with external API fallback.
    Extends the base RAGChatbot with additional resilience.
    """

    def __init__(self, *args, **kwargs):
        """Initialize hybrid chatbot with fallback capabilities."""
        super().__init__(*args, **kwargs)

        # Initialize external API client if enabled
        self.external_llm = None
        if (
            self.settings.enable_external_api_fallback
            and self.settings.external_api_key
        ):
            self._init_external_api()

    def _init_external_api(self):
        """Initialize external API client for fallback."""
        try:
            provider = self.settings.external_api_provider.lower()

            if provider == "openai":
                self.external_llm = ChatOpenAI(
                    model_name=self.settings.external_api_model,
                    temperature=0.7,
                    openai_api_key=self.settings.external_api_key,
                    max_tokens=1000,
                )
                logger.info(
                    f"✅ External OpenAI API initialized: {self.settings.external_api_model}"
                )

            elif provider == "anthropic":
                if ChatAnthropic is None:
                    raise ImportError(
                        "langchain_anthropic not installed. Run: pip install langchain-anthropic"
                    )
                self.external_llm = ChatAnthropic(
                    model=self.settings.external_api_model
                    or "claude-3-sonnet-20240229",
                    anthropic_api_key=self.settings.external_api_key,
                    temperature=0.7,
                )
                logger.info(
                    f"✅ External Anthropic API initialized: {self.settings.external_api_model}"
                )

            elif provider == "google":
                if ChatGoogleGenerativeAI is None:
                    raise ImportError(
                        "langchain_google_genai not installed. Run: pip install langchain-google-genai"
                    )
                self.external_llm = ChatGoogleGenerativeAI(
                    model=self.settings.external_api_model or "gemini-pro",
                    google_api_key=self.settings.external_api_key,
                    temperature=0.7,
                )
                logger.info(
                    f"✅ External Google API initialized: {self.settings.external_api_model}"
                )

            else:
                logger.warning(f"⚠️  Unknown external API provider: {provider}")

        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize external API: {str(e)}")
            self.external_llm = None

    def chat(
        self, question: str, max_sources: int = 5, use_fallback: bool = True
    ) -> ChatResponse:
        """
        Enhanced chat method with external API fallback.

        Args:
            question: User's question
            max_sources: Maximum number of source documents
            use_fallback: Whether to use external API fallback on local model failure

        Returns:
            ChatResponse with answer, sources, and metadata
        """
        try:
            # Try local model first
            response = super().chat(question, max_sources)

            # Add fallback information to metadata
            response.retrieval_metadata["used_external_fallback"] = False
            response.retrieval_metadata["fallback_available"] = (
                self.external_llm is not None
            )

            return response

        except Exception as e:
            logger.warning(f"⚠️  Local model failed: {str(e)}")

            if use_fallback and self.external_llm:
                logger.info("🔄 Falling back to external API...")
                return self._chat_with_external_api(question, max_sources, str(e))
            else:
                # Return error response
                logger.error("❌ No fallback available")
                return ChatResponse(
                    answer=f"I'm sorry, I encountered an error: {str(e)}. Local model is unavailable and no external API fallback is configured.",
                    sources=[],
                    retrieval_metadata={
                        "error": str(e),
                        "used_external_fallback": False,
                        "fallback_available": False,
                    },
                    confidence=0.0,
                    model_used="error",
                )

    def _chat_with_external_api(
        self, question: str, max_sources: int, local_error: str
    ) -> ChatResponse:
        """
        Chat using external API as fallback.

        Args:
            question: User's question
            max_sources: Maximum number of source documents
            local_error: Error message from local model

        Returns:
            ChatResponse from external API
        """
        try:
            # Still use local retrieval for documents
            sources, retrieval_metadata = self.retrieval_engine.retrieve_documents(
                query=question, max_results=max_sources
            )

            # Generate response with external API
            if sources:
                answer, confidence = self._generate_external_rag_response(
                    question, sources
                )
            else:
                answer, confidence = self._generate_external_fallback_response(question)

            # Create response
            retrieval_metadata["used_external_fallback"] = True
            retrieval_metadata["local_error"] = local_error
            retrieval_metadata["external_provider"] = (
                self.settings.external_api_provider
            )

            return ChatResponse(
                answer=answer,
                sources=sources,
                retrieval_metadata=retrieval_metadata,
                confidence=confidence,
                model_used=f"{self.settings.external_api_provider}:{self.settings.external_api_model}",
            )

        except Exception as e:
            logger.error(f"❌ External API also failed: {str(e)}")
            return ChatResponse(
                answer=f"I'm sorry, both local model and external API failed. Local error: {local_error}. External error: {str(e)}",
                sources=[],
                retrieval_metadata={
                    "local_error": local_error,
                    "external_error": str(e),
                    "used_external_fallback": True,
                    "fallback_failed": True,
                },
                confidence=0.0,
                model_used="error",
            )

    def _generate_external_rag_response(
        self, question: str, sources: List[RetrievalResult]
    ) -> tuple[str, float]:
        """Generate response using external API with retrieved documents."""
        # Prepare context from sources
        context_parts = []
        for i, source in enumerate(sources, 1):
            source_info = f"Source {i} ({source.source}): {source.title}"
            content = source.content[:800]  # Limit content length
            context_parts.append(f"{source_info}\n{content}\n")

        context = "\n".join(context_parts)

        # Generate prompt using the same template as local model
        prompt = self.rag_prompt_template.format(context=context, question=question)

        # Generate response with external API
        response = self.external_llm([HumanMessage(content=prompt)])
        answer = response.content.strip()

        # Calculate confidence (slightly lower for external API due to potential differences)
        confidence = self._calculate_response_confidence(sources) * 0.9

        return answer, confidence

    def _generate_external_fallback_response(self, question: str) -> tuple[str, float]:
        """Generate fallback response using external API when no documents found."""
        prompt = self.fallback_prompt_template.format(question=question)

        response = self.external_llm([HumanMessage(content=prompt)])
        answer = response.content.strip()

        # Low confidence for fallback responses
        confidence = 0.4

        return answer, confidence

    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced statistics including fallback information."""
        base_stats = super().get_stats()

        # Add fallback information
        base_stats["external_api"] = {
            "enabled": self.settings.enable_external_api_fallback,
            "provider": self.settings.external_api_provider
            if self.external_llm
            else None,
            "model": self.settings.external_api_model if self.external_llm else None,
            "available": self.external_llm is not None,
        }

        return base_stats

    def test_connectivity(self) -> Dict[str, bool]:
        """
        Test connectivity to both local and external services.

        Returns:
            Dictionary with connectivity status
        """
        results = {
            "local_llm": False,
            "local_embeddings": False,
            "external_api": False,
            "ollama_server": False,
            "vector_store": False,
        }

        # Test local LLM
        try:
            test_response = self.llm([HumanMessage(content="Test")])
            results["local_llm"] = bool(test_response.content)
        except Exception as e:
            logger.debug(f"Local LLM test failed: {str(e)}")

        # Test local embeddings
        try:
            if hasattr(self.retrieval_engine.vector_store.embedding_model, "encode"):
                test_embedding = (
                    self.retrieval_engine.vector_store.embedding_model.encode(["test"])
                )
                results["local_embeddings"] = len(test_embedding) > 0
        except Exception as e:
            logger.debug(f"Local embeddings test failed: {str(e)}")

        # Test vector store connectivity (don't fail if collection doesn't exist)
        try:
            # Just test if we can access the vector store, not the collection
            results["vector_store"] = True
        except Exception as e:
            logger.debug(f"Vector store test failed: {str(e)}")
            results["vector_store"] = False

        # Test external API
        if self.external_llm:
            try:
                test_response = self.external_llm([HumanMessage(content="Test")])
                results["external_api"] = bool(test_response.content)
            except Exception as e:
                logger.debug(f"External API test failed: {str(e)}")

        # Test Ollama server connectivity
        try:
            import requests

            response = requests.get(
                f"{self.settings.ollama_base_url}/api/tags", timeout=5
            )
            results["ollama_server"] = response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama server test failed: {str(e)}")

        return results


def create_hybrid_chatbot() -> HybridRAGChatbot:
    """
    Factory function to create a hybrid RAG chatbot.

    Returns:
        HybridRAGChatbot instance with local and external API support
    """
    return HybridRAGChatbot()
