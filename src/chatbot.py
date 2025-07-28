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
from src.conversation_memory import ConversationMemory, create_conversation_memory
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
    session_id: Optional[str] = None  # Session identifier for conversation tracking


class RAGChatbot:
    """
    Main RAG chatbot that orchestrates document retrieval and answer generation.
    Provides conversational interface with source attribution.
    """

    def __init__(
        self,
        retrieval_engine: Optional[RetrievalEngine] = None,
        conversation_memory: Optional[ConversationMemory] = None,
    ):
        """
        Initialize the RAG chatbot.

        Args:
            retrieval_engine: RetrievalEngine instance. Creates new one if None.
            conversation_memory: ConversationMemory instance. Creates new one if None.
        """
        self.settings = get_settings()

        # Validate required settings
        if not validate_settings():
            raise ValueError("Invalid configuration. Please check your settings.")

        # Initialize components
        self.retrieval_engine = retrieval_engine or create_retrieval_engine()
        self.conversation_memory = conversation_memory or create_conversation_memory()

        # Initialize language model
        self._init_language_model()

        # Create system prompt template
        self._init_prompt_templates()

        logger.info("Initialized RAG chatbot with conversation memory")

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

        # Main RAG prompt template - Comprehensive responses from knowledge base
        self.rag_prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a specialized technical documentation assistant that provides comprehensive, detailed information from the configured knowledge base sources. You must be thorough and provide in-depth answers while strictly adhering to the provided information.

📚 **Available information sources:**
{context}

❓ **Question:** {question}

**CRITICAL:** Answer ONLY based on the information provided above. Do not add external knowledge or general information from outside these sources.

Please provide a COMPREHENSIVE and DETAILED answer following these guidelines:
• 🎯 **Be thorough and comprehensive** - provide detailed explanations with step-by-step instructions when available
• 📝 **Use detailed structure** - create well-organized sections with headers, bullet points, numbered steps, and code examples
• 📚 **Quote extensively from sources** - include specific details, commands, and instructions from the provided sources
• 🌐 **External sources integration** - when using external sources, provide the specific URL and integrate the detailed information
• 💡 **Combine all relevant information** - synthesize information from multiple sources to provide complete answers
• 🔧 **Include technical details** - provide specific commands, file paths, configuration options, and technical specifications
• 📜 **Extract exact commands** - when sources contain build scripts, installation commands, or configuration scripts (like configure_cmake.sh), quote them exactly as they appear in ```shell``` code blocks or other formatted sections
• 📋 **Create actionable guides** - organize information into clear, actionable steps that users can follow
• 🔗 **Provide direct links** - include clickable URLs for further reading and reference
• 📖 **Expand on examples** - if the sources contain examples, explain them in detail
• 🚫 **Never add general knowledge** - only use information explicitly provided in the sources above

**Response Structure Requirements:**
- Start with a brief overview
- Provide detailed step-by-step instructions when applicable
- Include code examples or commands when available in sources (look for ```shell``` blocks and extract them exactly)
- When sources contain specific commands like ./configure_cmake.sh, export statements, or make commands, include them verbatim
- End with additional resources and links for further reading
- Aim for comprehensive coverage of the topic

💬 **Your comprehensive, detailed answer based ONLY on the provided sources:**""",
        )

        # Fallback prompt when no relevant documents found
        self.fallback_prompt_template = PromptTemplate(
            input_variables=["question"],
            template="""You are a specialized document assistant that ONLY provides information from the configured knowledge base (uploaded documents and specified external sources).

❓ **Question:** {question}

🔍 **What happened:** I searched through the available knowledge base but couldn't find any relevant information about this topic.

**IMPORTANT:** You must ONLY respond about topics covered in the knowledge base. DO NOT provide general knowledge or information from outside the configured sources.

Please respond following these strict guidelines:
• 😊 **Be polite and friendly** - but clearly state the limitation
• 🚫 **Do NOT provide general knowledge** - only mention what sources were searched
• 📄 **Be specific about knowledge base** - mention documents and configured external sources
• 🤝 **Offer to help with topics that ARE in the knowledge base**
• 💡 **Suggest they add relevant documents** if they want coverage of this topic

💬 **Polite response explaining the limitation:**""",
        )

    def chat(
        self, question: str, max_sources: int = 5, session_id: Optional[str] = None
    ) -> ChatResponse:
        """
        Process a chat question and return a response with sources and conversation context.

        Args:
            question: User's question.
            max_sources: Maximum number of source documents to use.
            session_id: Optional session ID for conversation context. Creates new session if None.

        Returns:
            ChatResponse with answer, sources, metadata, and session ID.
        """
        logger.info(f"Processing question: '{question}' (session: {session_id})")

        try:
            # Step 1: Handle session management
            if session_id is None:
                session_id = self.conversation_memory.create_session()

            # Add user message to conversation history
            if not self.conversation_memory.add_message(session_id, "user", question):
                logger.warning(
                    f"Failed to add message to session {session_id}, creating new session"
                )
                session_id = self.conversation_memory.create_session()
                self.conversation_memory.add_message(session_id, "user", question)

            # Step 2: Get conversation context for enhanced understanding
            conversation_context = self.conversation_memory.get_conversation_context(
                session_id, max_messages=6
            )

            # Step 3: Retrieve relevant documents (consider conversation context in query)
            enhanced_query = self._enhance_query_with_context(
                question, conversation_context
            )
            sources, retrieval_metadata = self.retrieval_engine.retrieve_documents(
                query=enhanced_query, max_results=max_sources
            )

            # Step 4: Generate response based on retrieved documents and conversation context
            if sources:
                answer, confidence = self._generate_rag_response(
                    question, sources, conversation_context
                )
            else:
                answer, confidence = self._generate_fallback_response(
                    question, conversation_context
                )

            # Step 5: Add assistant response to conversation history
            sources_data = [
                {
                    "title": s.title,
                    "source": s.source,
                    "relevance_score": s.relevance_score,
                }
                for s in sources
            ]
            self.conversation_memory.add_message(
                session_id,
                "assistant",
                answer,
                sources=sources_data,
                confidence=confidence,
            )

            # Step 6: Create response object
            response = ChatResponse(
                answer=answer,
                sources=sources,
                retrieval_metadata=retrieval_metadata,
                confidence=confidence,
                model_used=getattr(self, "model_name", "unknown"),
                session_id=session_id,
            )

            logger.info(
                f"Generated response with {len(sources)} sources (session: {session_id})"
            )
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
                session_id=session_id,
            )

    def _enhance_query_with_context(
        self, question: str, conversation_context: List[Dict[str, Any]]
    ) -> str:
        """
        Enhance the search query using conversation context.

        Args:
            question: Current user question
            conversation_context: Previous conversation messages

        Returns:
            Enhanced query string for better document retrieval
        """
        if not conversation_context or len(conversation_context) < 2:
            return question

        # Look for potential pronouns or references that need context
        context_keywords = []

        # Get recent user messages for context
        recent_user_messages = [
            msg for msg in conversation_context[-4:] if msg["role"] == "user"
        ]

        if recent_user_messages:
            # If current question is short and might reference previous topics
            if len(question.split()) <= 5:
                # Add keywords from recent questions
                for msg in recent_user_messages[-2:]:
                    words = msg["content"].split()
                    # Extract potential topic words (nouns, longer words)
                    topic_words = [
                        w
                        for w in words
                        if len(w) > 4
                        and w.lower() not in ["what", "where", "when", "which", "about"]
                    ]
                    context_keywords.extend(topic_words[:3])  # Limit to avoid noise

        if context_keywords:
            enhanced_query = f"{question} {' '.join(context_keywords[:3])}"
            logger.debug(f"Enhanced query: '{question}' -> '{enhanced_query}'")
            return enhanced_query

        return question

    def _generate_rag_response(
        self,
        question: str,
        sources: List[RetrievalResult],
        conversation_context: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[str, float]:
        """
        Generate response using retrieved documents as context.

        Args:
            question: User's question.
            sources: Retrieved source documents.
            conversation_context: Previous conversation for context

        Returns:
            Tuple of (answer, confidence_score).
        """
        # Prepare context from sources in a more natural format
        context_parts = []
        for i, source in enumerate(sources, 1):
            # Handle different source types with appropriate icons and descriptions
            if source.source.startswith("external_url_github"):
                icon = "🌐"
                source_type = "GitHub"
                doc_name = source.title
                if source.url:
                    source_info = f"{icon} **From {source_type} - {doc_name}** (relevance: {source.relevance_score:.0%}):\n🔗 {source.url}"
                else:
                    source_info = f"{icon} **From {source_type} - {doc_name}** (relevance: {source.relevance_score:.0%}):"
            elif source.source.startswith("external_url_"):
                icon = "🌐"
                source_type = "External Web Source"
                doc_name = source.title
                if source.url:
                    source_info = f"{icon} **From {source_type} - {doc_name}** (relevance: {source.relevance_score:.0%}):\n🔗 {source.url}"
                else:
                    source_info = f"{icon} **From {source_type} - {doc_name}** (relevance: {source.relevance_score:.0%}):"
            elif source.source == "corporate_portal":
                icon = "🏢"
                doc_name = source.title
                source_info = f"{icon} **From Corporate Portal - {doc_name}** (relevance: {source.relevance_score:.0%}):"
            else:
                # Local document
                icon = "📄"
                doc_name = (
                    source.title.replace(".pdf", "")
                    .replace(".docx", "")
                    .replace(".txt", "")
                )
                source_info = f"{icon} **From {doc_name}** (relevance: {source.relevance_score:.0%}):"

            content = source.content[:600]  # Shorter content for better readability
            context_parts.append(f"{source_info}\n{content}...\n")

        context = "\n---\n".join(context_parts)

        # Generate prompt with conversation context if available
        if conversation_context and len(conversation_context) > 1:
            # Build conversation history summary
            recent_exchanges = []
            for msg in conversation_context[-4:]:  # Last 4 messages
                role_indicator = "🧑 User" if msg["role"] == "user" else "🤖 Assistant"
                recent_exchanges.append(f"{role_indicator}: {msg['content'][:100]}...")

            conversation_summary = "\n".join(recent_exchanges)

            # Use enhanced prompt template with conversation context
            prompt = f"""You are a specialized document assistant that ONLY provides information from the configured knowledge base sources. You have access to previous conversation context to better understand follow-up questions.

📚 **Available information sources:**
{context}

🗣️ **Recent conversation context:**
{conversation_summary}

❓ **Current question:** {question}

**CRITICAL:** Answer ONLY based on the information provided above. Do not add external knowledge or general information from outside these sources.

Please provide a helpful answer following these strict guidelines:
• 🎯 **Consider conversation context** - if this seems like a follow-up question, acknowledge the connection
• 📝 **Use clear structure** - bullet points, short paragraphs, emojis for readability  
• 📚 **Reference sources naturally** - clearly indicate which specific source each piece of information comes from
• 🌐 **External sources note** - if using external sources, mention the specific URL and that it's from the configured knowledge base
• 💡 **Stay within knowledge base** - only combine information from the sources provided above
• 🤔 **Be honest about limitations** - if information is incomplete in the provided sources, say so clearly
• 🔗 **Include relevant links** - when referencing external sources, include the URL
• 🚫 **Never add general knowledge** - do not supplement with information not in the sources above

💬 **Answer based ONLY on the provided sources:**"""
        else:
            # Standard prompt for first message or no context
            prompt = self.rag_prompt_template.format(context=context, question=question)

        # Generate response
        response = self.llm([HumanMessage(content=prompt)])
        answer = response.content.strip()

        # Calculate confidence based on source quality
        confidence = self._calculate_response_confidence(sources)

        return answer, confidence

    def _generate_fallback_response(
        self, question: str, conversation_context: Optional[List[Dict[str, Any]]] = None
    ) -> tuple[str, float]:
        """
        Generate fallback response when no relevant documents found.
        Provides a polite but clear limitation message.

        Args:
            question: User's question.
            conversation_context: Previous conversation for context

        Returns:
            Tuple of (answer, confidence_score).
        """
        # Always use the restrictive fallback template
        prompt = self.fallback_prompt_template.format(question=question)

        response = self.llm([HumanMessage(content=prompt)])
        answer = response.content.strip()

        # Very low confidence for fallback responses to indicate limitation
        confidence = 0.1

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

    # Conversation management methods
    def create_conversation_session(self) -> str:
        """Create a new conversation session and return session ID."""
        return self.conversation_memory.create_session()

    def get_conversation_history(
        self, session_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        return self.conversation_memory.get_conversation_context(
            session_id, max_messages=limit, include_sources=True
        )

    def list_conversation_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent conversation sessions."""
        return self.conversation_memory.list_sessions(limit=limit)

    def cleanup_old_conversations(self) -> int:
        """Clean up old conversation sessions."""
        return self.conversation_memory.cleanup_old_sessions()


def create_chatbot() -> RAGChatbot:
    """
    Factory function to create a RAG chatbot instance.

    Returns:
        Initialized RAGChatbot instance.
    """
    return RAGChatbot()
