# RAG Chatbot

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated **Retrieval-Augmented Generation (RAG)** chatbot that combines local document search with corporate portal fallback. The system intelligently retrieves relevant information from your documents and falls back to a corporate portal when local knowledge is insufficient.

## 🌟 Features

### Core Capabilities
- **📄 Multi-format Document Support**: PDF, DOCX, TXT, Markdown, and HTML files
- **🔍 Intelligent Retrieval**: Vector similarity search using ChromaDB and embeddings with local-first approach
- **🤖 Local & External LLMs**: Support for Ollama (local) and external APIs (OpenAI, Anthropic, Google)
- **🌐 Advanced Web Crawling**: Comprehensive repository crawling with 7+ specialized libraries
- **🎯 Smart Confidence Scoring**: Determines when to use fallback based on local result quality
- **📊 Real-time Progress Indicators**: Visual feedback during external knowledge base searches
- **🗂️ Multi-Knowledge Base Support**: Connect multiple document directories and external sources

### **🧠 Conversation Memory & Context**
- **💭 Persistent Chat History**: All conversations automatically saved with session management
- **🔗 Context-Aware Responses**: Understands follow-up questions and references to previous messages
- **📝 Smart Query Enhancement**: Short questions get context from conversation history
- **🆔 Session Management**: UUID-based conversation tracking with automatic session creation
- **📂 Conversation Export**: Download chat sessions as text files for record keeping
- **🗂️ History Browser**: View and manage previous conversation sessions

### Document & System Management
- **🔄 Automatic Document Management**: Real-time file watching and smart indexing across multiple knowledge bases
- **🗑️ Deletion Handling**: Automatically removes deleted documents from the index
- **👀 File System Monitoring**: Automatic detection of document changes using watchdog
- **📊 Status Reporting**: Real-time indexing status and document synchronization
- **📁 Multi-Directory Support**: Index documents from multiple directories simultaneously
- **🌐 External Knowledge Base Integration**: Connect to remote repositories and documentation sites

### Interfaces & Configuration
- **💬 Multiple Interfaces**: Web UI with conversation controls, command-line chat, and REST API
- **⚙️ Highly Configurable**: Support for local models, external APIs, and extensive settings
- **🎮 Interactive Web UI**: Modern interface with conversation management, session tracking, and real-time progress indicators
- **🚀 Streaming Responses**: Real-time progress updates during external knowledge base searches
- **🔧 Advanced Crawler Settings**: Configurable depth, content length, and extraction options

## 🕷️ Advanced Web Crawling & External Knowledge Bases

The system now includes sophisticated web crawling capabilities that can comprehensively search external repositories and documentation sites:

### **🌟 Comprehensive Repository Crawling**
- **📂 Full Repository Discovery**: Crawls ALL files in GitHub repositories, not just README files
- **🎯 Intelligent File Filtering**: Automatically identifies documentation files (.md, .txt, .rst, .adoc) across entire repository structures
- **📊 Query-Specific Prioritization**: Prioritizes relevant files based on query content:
  - **Upgrade queries** → `doc/upgrade/`, `CHANGELOG.md`, migration guides
  - **Plugin queries** → `doc/user/plugins.txt`, `doc/modules/`, plugin documentation
  - **Build queries** → `BUILD.md`, `INSTALL.md`, `configure_cmake.sh` scripts
- **🔄 Smart Relevance Scoring**: Advanced relevance calculation with content-aware ranking

### **📚 Specialized Crawler Architecture**
The system now uses **specialized crawlers** with intelligent URL routing:

#### **🐙 GitHub Crawler** (`github_crawler.py`)
- **Comprehensive Repository Discovery**: Crawls ALL files in repositories
- **Technical Documentation Focus**: Optimized for README, INSTALL, BUILD files
- **Query-Specific Prioritization**: Upgrade guides, plugin docs, build instructions
- **GitHub API Integration**: Search repository contents efficiently

#### **🌐 Web Crawler** (`web_crawler.py`)  
- **Multi-Library Content Extraction**: Uses 7+ specialized libraries
- **Site Type Detection**: Automatically adapts extraction strategy
- **Documentation Sites**: Optimized for docs.*, wiki.*, readthedocs.io
- **Corporate Portals**: JavaScript support for modern knowledge bases
- **General Websites**: Multiple extraction methods with best result selection

#### **🚦 Unified Crawler Manager** (`crawler_manager.py`)
- **Intelligent URL Routing**: Automatically routes GitHub URLs vs. general web URLs
- **Unified Result Format**: Seamless integration regardless of crawler type
- **Performance Optimization**: Parallel crawling of different URL types
- **Extensible Architecture**: Easy to add new specialized crawlers

**Extraction Libraries:**
- **`requests-html`**: JavaScript-enabled web scraping
- **`trafilatura`**: Clean text extraction from web pages
- **`newspaper3k`**: Article and documentation parsing
- **`lxml`**: Fast XML/HTML processing
- **`pyquery`**: jQuery-like content selection
- **`selenium`**: Browser automation for complex sites
- **`BeautifulSoup`**: HTML parsing and navigation

### **🚀 Search Strategy & Performance**
- **Local-First Approach**: Always searches local documents first, only uses external sources when truly needed
- **Smart Scaling Architecture**: Automatically adapts performance based on knowledge base size:
  - **Small KB (< 20 docs)**: 3 local results, 0.05 similarity threshold, < 2 results trigger external
  - **Medium KB (20-100 docs)**: 8 local results, 0.10 similarity threshold, < 3 results trigger external  
  - **Large KB (100+ docs)**: 15 local results, 0.15 similarity threshold, < 5 results trigger external
- **Advanced Speed Optimizations**: 
  - **Early termination** when sufficient results found (5 general, 3 build-specific)
  - **Query-specific file prioritization** (BUILD.md, INSTALL.md before README.md)
  - **Limited external crawling** (max 2 URLs, 50 files per repo instead of 138)
  - **Short query filtering** (skips external search for < 3 words)
  - **Reduced timeouts** (10s GitHub, 15s general web)
- **Conservative Fallback**: External search triggers only when:
  - No local results found, OR
  - Too few results for specialized queries (dynamic threshold), OR
  - Low confidence (< 0.5) for technical documentation needs
- **Progress Indicators**: Real-time visual feedback during external searches
- **Rate Limit Handling**: Intelligent GitHub API usage with fallback strategies

### **🎯 Specialized Documentation Discovery**
The crawler is optimized for technical documentation and can find:
- **Build Instructions**: CMake files, configure scripts, dependency lists
- **Installation Guides**: Step-by-step setup procedures, requirements
- **Upgrade Procedures**: Version migration guides, breaking changes
- **Plugin Documentation**: Module guides, API references, examples
- **Configuration Files**: Settings, environment variables, deployment guides

## 🏗️ Architecture

The system uses a modular architecture with clear separation of concerns, conversation memory, and advanced external knowledge integration:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│ Retrieval Engine│───▶│   Chat Bot      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                    ┌───────────┼───────────┐            ▼
                    ▼           ▼           ▼   ┌─────────────────┐
          ┌─────────────┐ ┌─────────────┐ ┌─────┤ Language Model  │
          │Local Docs   │ │Multi-KB     │ │Ext. │ (Ollama/OpenAI) │
          │(Primary)    │ │Directories  │ │URLs └─────────────────┘
          └─────────────┘ └─────────────┘ └─────┘         │
                    │           │           │              ▼
                    └───────────┼───────────┘     ┌─────────────────┐
                                ▼                  │Conversation     │
                       ┌─────────────────┐         │Memory System    │
                       │ Vector Store    │         │ (Session Mgmt)  │
                       │ (ChromaDB)      │         └─────────────────┘
                       └─────────────────┘                  │
                                │                           ▼
                                ▼                  ┌─────────────────┐
                    ┌─────────────────────────────┐ │Advanced Crawler │
                    │   External Knowledge        │ │  Progress UI    │
                    │                             │ └─────────────────┘
                    │ ┌─────────────────┐         │          │
                    │ │Advanced Crawler │         │          ▼
                    │ │• GitHub Repos   │         │ ┌─────────────────┐
                    │ │• Documentation  │         │ │Persistent Storage│
                    │ │• 7+ Libraries   │         │ │   (JSON files)  │
                    │ └─────────────────┘         │ └─────────────────┘
                    └─────────────────────────────┘
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- **For Local LLMs (Recommended)**: [Ollama](https://ollama.ai/) installed with models:
  - `ollama pull gemma3n:latest` (LLM)
  - `ollama pull nomic-embed-text:latest` (embeddings)
- **For External APIs (Optional)**: API keys for OpenAI, Anthropic, or Google

### 2. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd talos-chatbot

# Option 1: Use shell script (recommended - handles virtual environment)
./install.sh

# Option 2: Use Python setup script (if you prefer Python-based setup)
python3 setup.py

# Option 3: Manual installation (if you already have virtual environment active)
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Recommended**: Use `./install.sh` as it automatically detects and uses existing virtual environments (like `talos`) and handles Python 3.13 compatibility issues.

### 3. Configuration

Create a `.env` file in the project root:

```env
# LLM Configuration (Choose one)
USE_LOCAL_LLM=true                    # Use local Ollama models (recommended)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3n:latest

# Embedding Configuration
USE_LOCAL_EMBEDDINGS=true             # Use local Ollama embeddings
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest
EMBEDDING_MODEL=all-MiniLM-L6-v2      # Fallback for sentence-transformers

# External API Fallback (Optional)
ENABLE_EXTERNAL_API_FALLBACK=true
EXTERNAL_API_PROVIDER=openai          # openai, anthropic, google
EXTERNAL_API_KEY=your_external_api_key_here
EXTERNAL_API_MODEL=gpt-3.5-turbo

# Knowledge Base Configuration
DOCUMENTS_DIRECTORY=documents
ADDITIONAL_KNOWLEDGE_PATHS=/path/to/kb1,/path/to/kb2

# External URL Search Configuration
ENABLE_EXTERNAL_URL_SEARCH=true
# Mix of GitHub repos and web documentation sites
EXTERNAL_SEARCH_URLS=https://github.com/snort3/snort3,https://docs.nginx.com,https://kubernetes.io/docs
URL_SEARCH_TIMEOUT=15
URL_FALLBACK_THRESHOLD=0.5
STRICT_KNOWLEDGE_BASE_MODE=true

# Advanced Web Crawler Configuration
USE_ADVANCED_CRAWLER=true
MAX_CRAWL_DEPTH=2
MAX_CONTENT_LENGTH=15000
EXTRACT_CODE_BLOCKS=true
RENDER_JAVASCRIPT=false

# Corporate Portal Configuration (Optional)
CORPORATE_PORTAL_URL=https://your-company-portal.com
CORPORATE_PORTAL_API_KEY=your_portal_api_key_here
CORPORATE_PORTAL_USERNAME=your_username
CORPORATE_PORTAL_PASSWORD=your_password

# Document Processing
DOCS_DIRECTORY=./documents
VECTOR_DB_PATH=./vector_db
SIMILARITY_THRESHOLD=0.15             # Lowered for nomic-embed-text compatibility
MAX_DOCS_TO_RETRIEVE=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=100

# Web Interface
FLASK_HOST=127.0.0.1
FLASK_PORT=5001
FLASK_DEBUG=true
```

### 4. Add Your Documents

```bash
# Create documents directory
mkdir documents

# Add your files (PDF, DOCX, TXT, MD, HTML)
cp /path/to/your/documents/* documents/

# The system will automatically detect and index new files!
# 🔄 Auto-indexing happens when you:
#   - Start the web interface
#   - Add/remove files (with file watching enabled)
#   - Run manual indexing commands
```

### 5. Run the Application

#### Web Interface (Recommended)
```bash
python main.py web
```
Then open http://localhost:5001 in your browser.

#### Command Line Chat
```bash
python main.py chat
```

#### Document Management
```bash
# Check indexing status
python main.py status

# Auto-sync documents (add new, remove deleted)
python main.py auto-index

# Manual indexing (legacy method)
python main.py index /path/to/your/documents

# Test system connectivity
python main.py test
```

#### Check Configuration
```bash
python main.py config
```

## 📚 Usage Examples

### **🚀 Streaming Web Interface with Progress Indicators**
The enhanced web interface now provides real-time feedback during searches:

```
🔍 Starting search...           ████░░░░░░ 20%
📄 Searching local documents... ████████░░ 80%
🕷️ Crawling external repos...   ██████████ 100%
```

**Key Features:**
- **Real-time Progress Bars**: Visual feedback during external knowledge base searches
- **Smart Search Strategy**: Local documents first, external sources only when needed
- **Streaming Endpoint**: `/chat-stream` for real-time updates
- **Conservative External Search**: Only triggers for specialized queries with insufficient local results

### Web Interface
The web interface provides a modern chat UI with:
- Real-time messaging with emoji and formatting support
- Source attribution with confidence scores
- System statistics and status monitoring
- **Progress indicators** with animated progress bars during external searches
- **Document management controls**:
  - **"Check Index Status"** - View current indexing state
  - **"Re-index Documents"** - Manual sync with progress feedback
  - Smart sync vs. force rebuild options
- **Automatic file watching** - Real-time detection of document changes
- **Multi-knowledge base support** - Connect multiple document directories
- Responsive design with floating assistant avatar

### **🕷️ Advanced Repository Crawling Example**
The system can now comprehensively crawl entire GitHub repositories to find specialized documentation:

```bash
# Example: Comprehensive Snort3 upgrade query
❓ Your question: How to upgrade Snort from version 2 to 3 with plugins?

🔍 Starting search...
📄 Searching local documents... (Found 3 local docs)
🌐 Checking for additional sources... (Low confidence for specialized query)
🕷️ Crawling external knowledge base...
   🚦 Routing URLs: 1 GitHub, 2 Web
   📂 GitHub Crawler: Discovering all files in snort3/snort3
   📁 Found 276 documentation files to process
   📄 Added doc/upgrade/snort_upgrade.txt (relevance: 1.0)
   📄 Added doc/user/plugins.txt (relevance: 0.87)
   🌐 Web Crawler: Processing documentation sites
   📄 Added nginx installation guide (relevance: 0.82)
   📄 Added kubernetes setup tutorial (relevance: 0.75)
🤖 Generating comprehensive response...

🤖 Answer (confidence: 95%):
# Snort 2 to 3 Upgrade Guide

## Overview
Snort 3 is a complete rewrite with significant architectural changes...

## Step-by-Step Upgrade Process
1. **Install Dependencies**: CMake, DAQ, LuaJIT...
2. **Build Snort 3**: Run ./configure_cmake.sh...
3. **Convert Configuration**: Use snort2lua tool...
4. **Plugin Migration**: Update custom plugins...

📚 Sources (5):
  1. doc/upgrade/snort_upgrade.txt (external) - 100% relevance
  2. doc/user/plugins.txt (external) - 87% relevance  
  3. doc/upgrade/differences.txt (external) - 85% relevance
  4. README.md (external) - 78% relevance
  5. ChangeLog.md (external) - 65% relevance
```

### Command Line
```bash
# Start interactive chat
python main.py chat

# Example interaction:
❓ Your question: What is the company's vacation policy?
🤔 Thinking...

🤖 Answer (confidence: 85%):
According to the HR handbook, employees are entitled to 20 days of vacation per year...

📚 Sources (2):
  1. HR_Handbook.pdf (local) - 92% relevance
  2. Company_Policies.docx (local) - 78% relevance
```

### REST API
```bash
# Send a chat request
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the security guidelines?"}'

# Send a streaming chat request with progress updates
curl -X POST http://localhost:5001/chat-stream \
  -H "Content-Type: application/json" \
  -d '{"message": "How to upgrade Snort plugins?"}' \
  --no-buffer

# Example streaming response:
# data: {"type": "progress", "message": "🔍 Starting search...", "step": 1, "total": 5}
# data: {"type": "progress", "message": "📄 Searching local documents...", "step": 2, "total": 5}
# data: {"type": "progress", "message": "🕷️ Crawling external knowledge base...", "step": 4, "total": 5}
# data: {"type": "result", "response": "...", "sources": [...], "confidence": 0.95}

# Send a chat request with session ID (for conversation continuity)
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me more about that policy", "session_id": "your-session-id"}'

# Get system statistics
curl http://localhost:5001/api/stats

# Conversation management endpoints
curl http://localhost:5001/api/conversations                    # List recent conversations
curl -X POST http://localhost:5001/api/conversations           # Create new conversation
curl http://localhost:5001/api/conversations/SESSION_ID/history # Get conversation history
```

## 🧠 Conversation Memory Features

The system now includes sophisticated conversation memory that enables natural follow-up questions and contextual understanding:

### **How Conversation Memory Works**
```bash
# Example conversation flow:
User: "What is the company's vacation policy?"
Bot: "According to the HR handbook, employees get 20 days of vacation per year..."

User: "What about sick leave?"  # ← Bot understands this relates to HR policies
Bot: "Based on the same HR document, employees receive 10 days of sick leave..."

User: "Can I combine them?"    # ← Bot knows "them" refers to vacation and sick leave
Bot: "The HR handbook states that vacation and sick leave cannot be combined..."
```

### **Key Conversation Features**
- **🔗 Context Continuity**: Bot remembers what you discussed in the current session
- **📝 Smart References**: Understands pronouns and references like "that", "it", "them"
- **🎯 Enhanced Queries**: Short follow-up questions automatically include context
- **💾 Persistent Sessions**: Conversations survive browser refresh and restarts
- **📊 Session Tracking**: Real-time session status and message counts in web UI

### **Web Interface Conversation Controls**
- **🆕 New Conversation**: Start fresh topics without losing context
- **📜 View History**: Browse previous conversation sessions with timestamps
- **📁 Export Chat**: Download complete conversations as text files
- **🔄 Session Management**: Automatic session creation and management

### **Conversation API Usage**
```bash
# Create a new conversation session
curl -X POST http://localhost:5001/api/conversations

# Get conversation history
curl http://localhost:5001/api/conversations/SESSION_ID/history

# Continue an existing conversation
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me more", "session_id": "existing-session-id"}'
```

## 🔄 Automatic Document Management

The system now includes comprehensive automatic document management:

### **Real-time File Watching**
- **Automatic detection** of new, modified, or deleted files
- **Background monitoring** using the `watchdog` library
- **Smart debouncing** to avoid duplicate processing
- **Instant re-indexing** when documents change

### **Smart Document Synchronization**
```bash
# Check what needs to be synchronized
python main.py status
📁 Files in directory: 15
🆕 New files detected: 2
🗑️ Deleted files detected: 1
💾 Documents in database: 12
⚠️ Index needs updating!

# Auto-sync changes
python main.py auto-index
🔄 Document sync completed!
📄 New documents: 2
🗑️ Deleted documents: 1
📊 Total chunks: 45
```

### **Web Interface Controls**
- **Check Index Status** button - Real-time status monitoring
- **Re-index Documents** button - Manual control with options:
  - **Smart Sync**: Only process changes (fast)
  - **Force Rebuild**: Complete re-index (thorough)
- **Progress feedback** with detailed results
- **Automatic refresh** of document counts

### **File Change Detection**
The system automatically handles:
- ✅ **New files added** to documents folder
- ✅ **Files modified** (content changes detected via hash)
- ✅ **Files deleted** (removed from index automatically)
- ✅ **Folder reorganization** (recursive monitoring)

### **Zero Manual Work Required**
Once running, the system:
1. **Watches** the documents folder continuously
2. **Detects** changes within 2 seconds
3. **Re-indexes** automatically in the background
4. **Updates** the chatbot's knowledge instantly

## 🔧 Configuration Options

### Environment Variables

#### LLM Configuration
| Variable                       | Default                  | Description                                 |
| ------------------------------ | ------------------------ | ------------------------------------------- |
| `USE_LOCAL_LLM`                | `true`                   | Use local Ollama models                     |
| `OLLAMA_BASE_URL`              | `http://localhost:11434` | Ollama server URL                           |
| `OLLAMA_MODEL`                 | `gemma3n:latest`         | Local LLM model name                        |
| `ENABLE_EXTERNAL_API_FALLBACK` | `true`                   | Enable external API fallback                |
| `EXTERNAL_API_PROVIDER`        | `openai`                 | External provider (openai/anthropic/google) |
| `EXTERNAL_API_KEY`             | (optional)               | External API key                            |
| `EXTERNAL_API_MODEL`           | `gpt-3.5-turbo`          | External model name                         |

#### Embedding Configuration
| Variable                 | Default                   | Description                         |
| ------------------------ | ------------------------- | ----------------------------------- |
| `USE_LOCAL_EMBEDDINGS`   | `true`                    | Use local Ollama embeddings         |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text:latest` | Local embedding model               |
| `EMBEDDING_MODEL`        | `all-MiniLM-L6-v2`        | Fallback sentence transformer model |

#### Document Processing
| Variable               | Default       | Description                          |
| ---------------------- | ------------- | ------------------------------------ |
| `DOCS_DIRECTORY`       | `./documents` | Local documents directory            |
| `VECTOR_DB_PATH`       | `./vector_db` | Vector database storage path         |
| `SIMILARITY_THRESHOLD` | `0.15`        | Minimum similarity for local results |
| `MAX_DOCS_TO_RETRIEVE` | `5`           | Maximum documents per query          |
| `CHUNK_SIZE`           | `1000`        | Document chunk size in characters    |
| `CHUNK_OVERLAP`        | `100`         | Overlap between chunks               |

#### Knowledge Base Configuration
| Variable                     | Default     | Description                         |
| ---------------------------- | ----------- | ----------------------------------- |
| `DOCUMENTS_DIRECTORY`        | `documents` | Primary documents directory         |
| `ADDITIONAL_KNOWLEDGE_PATHS` | (optional)  | Comma-separated additional KB paths |

#### External URL Search & Advanced Crawler
| Variable                     | Default                            | Description                                    |
| ---------------------------- | ---------------------------------- | ---------------------------------------------- |
| `ENABLE_EXTERNAL_URL_SEARCH` | `true`                             | Enable external URL search fallback            |
| `EXTERNAL_SEARCH_URLS`       | `https://github.com/snort3/snort3` | Comma-separated URLs to crawl                  |
| `URL_SEARCH_TIMEOUT`         | `15`                               | Timeout for external URL requests (seconds)    |
| `URL_FALLBACK_THRESHOLD`     | `0.5`                              | Confidence threshold for triggering URL search |
| `STRICT_KNOWLEDGE_BASE_MODE` | `true`                             | Only answer from configured knowledge sources  |
| `USE_ADVANCED_CRAWLER`       | `true`                             | Use advanced multi-library crawler             |
| `MAX_CRAWL_DEPTH`            | `2`                                | Maximum crawl depth for following links        |
| `MAX_CONTENT_LENGTH`         | `15000`                            | Maximum content length per crawl result        |
| `EXTRACT_CODE_BLOCKS`        | `true`                             | Extract code blocks from documentation         |
| `RENDER_JAVASCRIPT`          | `false`                            | Enable JavaScript rendering for complex sites  |

#### Corporate Portal (Optional)
| Variable                    | Default    | Description               |
| --------------------------- | ---------- | ------------------------- |
| `CORPORATE_PORTAL_URL`      | (optional) | Corporate portal base URL |
| `CORPORATE_PORTAL_API_KEY`  | (optional) | Portal API authentication |
| `CORPORATE_PORTAL_USERNAME` | (optional) | Portal login username     |
| `CORPORATE_PORTAL_PASSWORD` | (optional) | Portal login password     |

#### Web Interface
| Variable      | Default     | Description             |
| ------------- | ----------- | ----------------------- |
| `FLASK_HOST`  | `127.0.0.1` | Web server host         |
| `FLASK_PORT`  | `5001`      | Web server port         |
| `FLASK_DEBUG` | `true`      | Enable Flask debug mode |

#### Conversation Memory
| Setting                  | Default                      | Description                             |
| ------------------------ | ---------------------------- | --------------------------------------- |
| **Storage Location**     | `./vector_db/conversations/` | Where conversation files are stored     |
| **Session Timeout**      | 24 hours                     | How long conversations are kept         |
| **Auto-cleanup**         | Enabled                      | Automatically removes old conversations |
| **Max Context Messages** | 10                           | Maximum messages included in context    |
| **Session ID Format**    | UUID4                        | Unique identifier format for sessions   |

### **🧠 Enhanced Retrieval Logic with Advanced Crawling & Smart Scaling**

The system uses sophisticated retrieval logic with conversation awareness, comprehensive external search, and automatic performance scaling:

1. **Conversation Context**: Retrieve recent conversation history for context
2. **Query Enhancement**: Enhance short queries with conversation keywords from history
3. **Dynamic Local Search**: Query all configured knowledge bases with intelligent scaling:
   - **Small KB (< 20 docs)**: Conservative search with 3 results, 0.05 threshold
   - **Medium KB (20-100 docs)**: Balanced search with 8 results, 0.10 threshold
   - **Large KB (100+ docs)**: Comprehensive search with 15 results, 0.15 threshold
4. **Confidence Evaluation**: Analyze result quality and relevance scores across all local sources
5. **Smart Fallback Decision**: Use external crawler only when truly needed:
   - No local results found, OR
   - Too few results for specialized queries (dynamic threshold based on KB size), OR
   - Low confidence (< 0.5) for technical documentation needs
   - Skip external search for very short queries (< 3 words) for speed
6. **Optimized External Crawling**: If triggered:
   - **Early termination**: Stop at 5 general results or 3 build-specific results
   - **File prioritization**: Check BUILD.md, INSTALL.md before general README files
   - **Limited scope**: Process max 50 files per repo (down from 138) and 2 URLs max
   - **Faster timeouts**: 10s for GitHub, 15s for general web crawling
   - Use 7+ specialized libraries for robust content extraction
   - Extract up to 15,000 characters of relevant technical content
7. **Comprehensive Response Generation**: 
   - Combine local and external content with conversation context
   - Provide detailed, actionable information with exact commands and steps
   - Include direct links to source documentation
8. **Memory Storage**: Save complete interaction with sources and session ID for continuity

**Search Priority Order:**
```
Local Documents (Primary Dir) → Additional KB Directories → External URLs (Conservative)
     ↓                              ↓                           ↓
Always searched first     Multi-directory support    Only for specialized queries
Fast & comprehensive     Real-time file watching     GitHub repos, docs sites
```

## 🛠️ Development

### Project Structure

```
talos-chatbot/
├── src/
│   ├── __init__.py
│   ├── chatbot.py              # Main RAG orchestrator with conversation support
│   ├── hybrid_chatbot.py       # Hybrid LLM support (local + external)
│   ├── conversation_memory.py  # Conversation memory and session management
│   ├── document_loader.py      # Document processing with multi-KB support
│   ├── vector_store.py         # ChromaDB vector operations
│   ├── retrieval.py           # Enhanced retrieval logic with smart scaling & speed optimizations
│   ├── corporate_portal.py    # Corporate portal integration
│   ├── ollama_embeddings.py   # Ollama embedding integration
│   ├── auto_indexer.py        # Automatic document indexing (multi-directory)
│   ├── file_watcher.py        # Real-time file system monitoring
│   ├── url_search.py          # External URL search and GitHub integration
│   ├── github_crawler.py      # Specialized GitHub repository crawler
│   ├── web_crawler.py         # General HTML website crawler
│   └── crawler_manager.py     # Unified crawler manager with URL routing
├── templates/
│   └── index.html             # Modern web interface with progress indicators and streaming
├── static/
│   └── avatar.svg             # Assistant avatar image
├── vector_db/
│   └── conversations/         # Persistent conversation storage (JSON files)
├── config.py                  # Configuration management
├── web_interface.py           # Flask web server with streaming and progress APIs
├── main.py                    # Main CLI entry point
├── install.sh                 # Installation script with venv support
├── setup.py                   # Python setup script
├── requirements.txt           # Python dependencies
└── README.md                  # This documentation
```

### Adding New Document Types

To support additional document formats, extend the `DocumentLoader` class:

```python
def _load_new_format(self, file_path: Path) -> str:
    """Load content from new format."""
    # Implement your loader logic
    return extracted_text
```

### Customizing Corporate Portal Integration

Modify the `CorporatePortalClient` class to match your portal's API:

```python
def _search_via_api(self, query: str, max_results: int) -> List[Dict[str, Any]]:
    """Customize for your portal's API endpoints."""
    # Implement your specific API integration
    pass
```

## 🔍 API Reference

### Chat Endpoints

#### Standard Chat
```
POST /api/chat
Content-Type: application/json

{
  "message": "Your question here",
  "max_sources": 5,          // optional
  "session_id": "uuid"       // optional - for conversation continuity
}
```

#### Streaming Chat with Progress
```
POST /chat-stream
Content-Type: application/json

{
  "message": "How to upgrade Snort plugins?",
  "max_sources": 5,          // optional
  "session_id": "uuid"       // optional
}
```

**Streaming Response Format:**
```
data: {"type": "progress", "message": "🔍 Starting search...", "step": 1, "total": 5}
data: {"type": "progress", "message": "📄 Searching local documents...", "step": 2, "total": 5}
data: {"type": "progress", "message": "🌐 Checking for additional sources...", "step": 3, "total": 5}
data: {"type": "progress", "message": "🕷️ Crawling external knowledge base...", "step": 4, "total": 5}
data: {"type": "progress", "message": "🤖 Generating response...", "step": 5, "total": 5}
data: {"type": "result", "response": "...", "sources": [...], "confidence": 0.95, "session_id": "..."}
```

Response:
```json
{
  "answer": "Generated response",
  "sources": [
    {
      "title": "Document title",
      "content": "Relevant excerpt...",
      "source": "local|corporate_portal",
      "relevance_score": 0.85,
      "url": "source_url",
      "metadata": {}
    }
  ],
  "confidence": 0.85,
  "retrieval_metadata": {
    "local_results_count": 3,
    "portal_results_count": 0,
    "used_fallback": false
  },
  "model_used": "gpt-3.5-turbo"
}
```

### Management Endpoints
- `GET /api/stats` - System statistics
- `POST /api/initialize` - Reinitialize knowledge base
- `POST /api/clear` - Clear knowledge base

### Document Management Endpoints
- `GET /index-status` - Get current indexing status and file watcher state
- `POST /reindex` - Manually trigger document synchronization
  ```json
  {
    "force": false  // true for complete rebuild, false for smart sync
  }
  ```

## 🚨 Troubleshooting

### Common Issues

1. **Ollama connection issues**
   ```bash
   # Ensure Ollama is running
   ollama serve
   
   # Pull required models
   ollama pull gemma3n:latest
   ollama pull nomic-embed-text:latest
   
   # Test connectivity
   python main.py test
   ```

2. **File watching not working**
   ```bash
   # Install watchdog if missing
   pip install watchdog>=3.0.0
   
   # Check file watcher status in web interface
   # Look for "👀 File watcher: ✅ Active" message
   ```

3. **Documents not indexing automatically**
   ```bash
   # Check index status
   python main.py status
   
   # Force manual sync
   python main.py auto-index
   
   # Verify documents directory exists
   ls -la ./documents/
   ```

4. **External API fallback issues**
   ```bash
   # Check external API configuration
   export EXTERNAL_API_KEY=your_actual_key
   export EXTERNAL_API_PROVIDER=openai  # or anthropic, google
   ```

5. **Memory issues with large documents**
   ```bash
   # Reduce chunk size in configuration
   export CHUNK_SIZE=500
   export CHUNK_OVERLAP=50
   ```

6. **Corporate portal authentication**
   ```bash
   # Verify portal credentials
   export CORPORATE_PORTAL_USERNAME=your_username
   export CORPORATE_PORTAL_PASSWORD=your_password
   ```

7. **Conversation memory not working**
   ```bash
   # Check if conversations directory exists
   ls -la ./vector_db/conversations/
   
   # Verify conversation storage permissions
   mkdir -p ./vector_db/conversations
   
   # Clear old conversation data if corrupted
   rm -rf ./vector_db/conversations/*.json
   ```

8. **Session not persisting between browser refreshes**
   ```bash
   # This is expected behavior - each browser session starts fresh
   # Use "View History" to access previous conversations
   # Export important conversations before closing browser
   ```

9. **Python 3.13 setuptools issues**
   ```bash
   # Run the setup script to handle compatibility
   python setup.py
   
   # Or manually upgrade setuptools
   pip install --upgrade pip setuptools wheel
   ```

10. **Package installation failures**
    ```bash
    # Use the installation script (recommended)
    ./install.sh
    
    # Or install dependencies manually
    pip install --upgrade setuptools pip wheel
    pip install langchain langchain-community langchain-ollama
    pip install chromadb sentence-transformers watchdog
    ```

11. **Slow response times with large knowledge base**
    ```bash
    # Check knowledge base size - system auto-optimizes
    python main.py status
    
    # For 100+ documents, the system automatically:
    # - Increases similarity threshold to 0.15
    # - Limits local search to 15 results
    # - Requires 5+ results before external search
    
    # Manual optimization for very large KBs (1000+ docs)
    export SIMILARITY_THRESHOLD=0.20
    export MAX_DOCS_TO_RETRIEVE=10
    ```

12. **External search taking too long**
    ```bash
    # System automatically limits external search:
    # - Max 2 URLs processed
    # - Max 50 files per repository  
    # - 10s timeout for GitHub API
    # - Skips external search for short queries
    
    # To disable external search entirely
    export ENABLE_EXTERNAL_URL_SEARCH=false
    
    # To reduce external search threshold (less aggressive)
    export URL_FALLBACK_THRESHOLD=0.7
    ```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
python main.py --debug web
```

## 📈 Performance Optimization

### **🚀 Automatic Performance Scaling**
The system automatically optimizes performance based on your knowledge base size:

#### **📚 Small Knowledge Base (< 20 documents)**
- **Local Search**: 3 results maximum (fast, focused)
- **Similarity Threshold**: 0.05 (inclusive, keeps more results)
- **External Trigger**: < 2 results for specialized queries
- **Target Use Case**: Personal projects, small teams

#### **📊 Medium Knowledge Base (20-100 documents)**  
- **Local Search**: 8 results maximum (balanced)
- **Similarity Threshold**: 0.10 (moderate selectivity)
- **External Trigger**: < 3 results for specialized queries
- **Target Use Case**: Department knowledge bases, project documentation

#### **🏢 Large Knowledge Base (100+ documents)**
- **Local Search**: 15 results maximum (comprehensive)
- **Similarity Threshold**: 0.15 (high selectivity, quality over quantity)
- **External Trigger**: < 5 results for specialized queries
- **Target Use Case**: Enterprise knowledge bases, extensive documentation

### **⚡ Speed Optimizations**
- **Early Termination**: Stops crawling when sufficient results found
- **Query-Specific Prioritization**: BUILD.md and INSTALL.md checked before README.md
- **Limited External Scope**: Max 2 URLs and 50 files per repository (down from 138)
- **Short Query Filtering**: Skips external search for queries < 3 words
- **Reduced Timeouts**: 10s GitHub API, 15s general web requests
- **Smart Caching**: Repository results cached for 5 minutes

### **🎯 General Performance Tips**
- **Document Indexing**: Index documents once, query many times
- **Chunk Size**: Balance between context and retrieval speed
- **Similarity Threshold**: Higher thresholds reduce false positives
- **Corporate Portal**: Cache results to reduce external API calls
- **File Organization**: Use descriptive filenames for better retrieval
- **Query Specificity**: More specific queries get better, faster results

### **⏱️ Expected Response Times**
Performance varies based on knowledge base size and query complexity:

- **Small KB (< 20 docs)**: 2-5 seconds typical
- **Medium KB (20-100 docs)**: 3-8 seconds typical  
- **Large KB (100+ docs)**: 5-15 seconds typical
- **With External Search**: Add 10-30 seconds for specialized queries
- **Build/Install Queries**: Faster due to file prioritization (BUILD.md first)

**Factors affecting speed:**
- ✅ **Local documents only**: Fastest (2-8s)
- ⚡ **Short queries** (< 3 words): Skip external search automatically
- 🔍 **Specialized queries** (build, install, upgrade): Optimized file prioritization
- 🌐 **External search triggered**: Slower but comprehensive (15-45s total)
- 🕷️ **Large repositories**: Limited to 50 files for speed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for RAG framework
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [OpenAI](https://openai.com/) for language models
- [Requests-HTML](https://github.com/psf/requests-html) for JavaScript-enabled web scraping
- [Trafilatura](https://trafilatura.readthedocs.io/) for clean text extraction
- [Newspaper3k](https://newspaper.readthedocs.io/) for article parsing
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) for HTML parsing
- [Selenium](https://selenium-python.readthedocs.io/) for browser automation

---

**Need help?** Check the troubleshooting section or create an issue on GitHub.
