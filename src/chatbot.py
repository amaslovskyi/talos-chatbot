"""
Main chatbot orchestrator module.
Combines document retrieval with language model generation for RAG responses.

MIT License - Copyright (c) 2025 talos-chatbot
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# LangChain and LLM imports
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

from src.retrieval import RetrievalEngine, RetrievalResult, create_retrieval_engine
from src.document_loader import load_and_chunk_documents
from src.vector_store import create_vector_store
from config import get_settings, validate_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    """
    Represents a chatbot response with metadata.
    """

    answer: str
    sources: List[RetrievalResult]
    retrieval_metadata: Dict[str, Any]
    confidence: float
    model_used: str


class RAGChatbot:
    """
    Main RAG chatbot that orchestrates document retrieval and answer generation.
    Provides conversational interface with source attribution.
    """

    def __init__(self, retrieval_engine: Optional[RetrievalEngine] = None):
        """
        Initialize the RAG chatbot.

        Args:
            retrieval_engine: RetrievalEngine instance. Creates new one if None.
        """
        self.settings = get_settings()

        # Validate required settings
        if not validate_settings():
            raise ValueError("Invalid configuration. Please check your settings.")

        # Initialize components
        self.retrieval_engine = retrieval_engine or create_retrieval_engine()

        # Initialize language model
        self._init_language_model()

        # Create system prompt template
        self._init_prompt_templates()

        logger.info("Initialized RAG chatbot")

    def _init_language_model(self):
        """Initialize the language model for generation."""
        try:
            if self.settings.use_local_llm:
                # Use local Ollama model
                if ChatOllama is None:
                    raise ImportError(
                        "langchain_ollama not installed. Run: pip install langchain-ollama"
                    )

                self.llm = ChatOllama(
                    model=self.settings.ollama_model,
                    base_url=self.settings.ollama_base_url,
                    temperature=0.7,
                )
                logger.info(f"Initialized Ollama model: {self.settings.ollama_model}")
                self.model_name = self.settings.ollama_model
            else:
                # Use OpenAI model
                self.llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=0.7,
                    openai_api_key=self.settings.openai_api_key,
                    max_tokens=1000,
                )
                logger.info("Initialized ChatOpenAI model")
                self.model_name = "gpt-3.5-turbo"

        except Exception as e:
            logger.error(f"Error initializing language model: {str(e)}")
            raise

    def _init_prompt_templates(self):
        """Initialize prompt templates for different scenarios."""

        # Main RAG prompt template - More conversational and user-friendly
        self.rag_prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a document assistant that ONLY provides information from the user's documents. You must be friendly and conversational, but you can ONLY use the information provided below. DO NOT add external knowledge or general information.

📄 **Information from your documents:**
{context}

❓ **Question:** {question}

**CRITICAL:** Answer ONLY based on the information provided above. Do not add external knowledge.

Please provide a helpful answer following these guidelines:
• 🎯 **Be conversational and friendly** - but stick to document content only
• 📝 **Use clear structure** - bullet points, short paragraphs, emojis for readability  
• 📚 **Reference sources naturally** - "According to your document..." or "I found that..."
• 💡 **Focus on what's in the documents** - don't add external explanations
• 🤔 **Be honest about limitations** - if info is incomplete in documents, say so
• 🚫 **Never add general knowledge** - only use the provided document content

💬 **Answer based ONLY on the documents:**""",
        )

        # Fallback prompt when no relevant documents found
        self.fallback_prompt_template = PromptTemplate(
            input_variables=["question"],
            template="""You are a document assistant that ONLY provides information from the user's uploaded documents and corporate portal. You cannot provide general knowledge or external information.

❓ **Question:** {question}

🔍 **What happened:** I searched through your documents and corporate portal but couldn't find any information about this topic.

**IMPORTANT:** You must ONLY respond based on the user's documents and corporate portal. DO NOT provide general knowledge, external information, or suggestions to look elsewhere.

Please respond following these strict guidelines:
• 😊 **Be friendly but clear** - explain that no information was found in their sources
• 🚫 **Do NOT provide general knowledge** - only mention what you searched
• 📄 **Be specific about sources searched** - mention documents and corporate portal
• 🤝 **Offer to help with other questions** about their actual documents
• 💡 **Suggest they add more documents** if they want information on this topic

💬 **Response:**""",
        )

    def chat(self, question: str, max_sources: int = 5) -> ChatResponse:
        """
        Process a chat question and return a response with sources.

        Args:
            question: User's question.
            max_sources: Maximum number of source documents to use.

        Returns:
            ChatResponse with answer, sources, and metadata.
        """
        logger.info(f"Processing question: '{question}'")

        try:
            # Step 1: Retrieve relevant documents
            sources, retrieval_metadata = self.retrieval_engine.retrieve_documents(
                query=question, max_results=max_sources
            )

            # Step 2: Generate response based on retrieved documents
            if sources:
                answer, confidence = self._generate_rag_response(question, sources)
            else:
                answer, confidence = self._generate_fallback_response(question)

            # Step 3: Create response object
            response = ChatResponse(
                answer=answer,
                sources=sources,
                retrieval_metadata=retrieval_metadata,
                confidence=confidence,
                model_used=getattr(self, "model_name", "unknown"),
            )

            logger.info(f"Generated response with {len(sources)} sources")
            return response

        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            # Return error response
            return ChatResponse(
                answer=f"I'm sorry, I encountered an error while processing your question: {str(e)}",
                sources=[],
                retrieval_metadata={},
                confidence=0.0,
                model_used="error",
            )

    def _generate_rag_response(
        self, question: str, sources: List[RetrievalResult]
    ) -> tuple[str, float]:
        """
        Generate response using retrieved documents as context.

        Args:
            question: User's question.
            sources: Retrieved source documents.

        Returns:
            Tuple of (answer, confidence_score).
        """
        # Prepare context from sources in a more natural format
        context_parts = []
        for i, source in enumerate(sources, 1):
            # Create a more user-friendly source description
            doc_name = (
                source.title.replace(".pdf", "")
                .replace(".docx", "")
                .replace(".txt", "")
            )
            source_info = (
                f"📄 **From {doc_name}** (relevance: {source.relevance_score:.0%}):"
            )
            content = source.content[:600]  # Shorter content for better readability
            context_parts.append(f"{source_info}\n{content}...\n")

        context = "\n---\n".join(context_parts)

        # Generate prompt
        prompt = self.rag_prompt_template.format(context=context, question=question)

        # Generate response
        response = self.llm([HumanMessage(content=prompt)])
        answer = response.content.strip()

        # Calculate confidence based on source quality
        confidence = self._calculate_response_confidence(sources)

        return answer, confidence

    def _generate_fallback_response(self, question: str) -> tuple[str, float]:
        """
        Generate fallback response when no relevant documents found.

        Args:
            question: User's question.

        Returns:
            Tuple of (answer, confidence_score).
        """
        prompt = self.fallback_prompt_template.format(question=question)

        response = self.llm([HumanMessage(content=prompt)])
        answer = response.content.strip()

        # Low confidence for fallback responses
        confidence = 0.3

        return answer, confidence

    def _calculate_response_confidence(self, sources: List[RetrievalResult]) -> float:
        """
        Calculate confidence score for the response based on source quality.

        Args:
            sources: List of source documents used.

        Returns:
            Confidence score between 0 and 1.
        """
        if not sources:
            return 0.0

        # Base confidence on average relevance score of sources
        avg_relevance = sum(s.relevance_score for s in sources) / len(sources)

        # Boost confidence if we have local sources (more trusted)
        local_sources = [s for s in sources if s.source == "local"]
        local_boost = len(local_sources) / len(sources) * 0.1

        # Boost confidence based on number of sources
        source_count_boost = min(len(sources) / 5, 1.0) * 0.1

        confidence = min(avg_relevance + local_boost + source_count_boost, 1.0)

        return confidence

    def initialize_knowledge_base(self, documents_directory: Optional[str] = None):
        """
        Initialize the knowledge base by loading and indexing documents.

        Args:
            documents_directory: Path to documents directory. Uses config default if None.
        """
        try:
            logger.info("Initializing knowledge base...")

            # Load and chunk documents
            documents = load_and_chunk_documents(documents_directory)

            if not documents:
                logger.warning("No documents found to index")
                return

            # Add documents to vector store
            self.retrieval_engine.vector_store.add_documents(documents)

            logger.info(f"Successfully indexed {len(documents)} document chunks")

        except Exception as e:
            logger.error(f"Error initializing knowledge base: {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        Get chatbot statistics and status.

        Returns:
            Dictionary with chatbot statistics.
        """
        retrieval_stats = self.retrieval_engine.get_retrieval_stats()

        return {
            "model": getattr(self, "model_name", "unknown"),
            "model_type": "ollama" if self.settings.use_local_llm else "openai",
            "retrieval_engine": retrieval_stats,
            "settings": {
                "similarity_threshold": self.settings.similarity_threshold,
                "max_docs_to_retrieve": self.settings.max_docs_to_retrieve,
                "chunk_size": self.settings.chunk_size,
                "use_local_llm": self.settings.use_local_llm,
                "use_local_embeddings": self.settings.use_local_embeddings,
            },
        }

    def clear_knowledge_base(self):
        """Clear the local knowledge base."""
        self.retrieval_engine.vector_store.clear_collection()
        logger.info("Cleared knowledge base")


def create_chatbot() -> RAGChatbot:
    """
    Factory function to create a RAG chatbot instance.

    Returns:
        Initialized RAGChatbot instance.
    """
    return RAGChatbot()
