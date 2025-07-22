# RAG Chatbot

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated **Retrieval-Augmented Generation (RAG)** chatbot that combines local document search with corporate portal fallback. The system intelligently retrieves relevant information from your documents and falls back to a corporate portal when local knowledge is insufficient.

## 🌟 Features

- **📄 Multi-format Document Support**: PDF, DOCX, TXT, Markdown, and HTML files
- **🔍 Intelligent Retrieval**: Vector similarity search using ChromaDB and embeddings
- **🤖 Local & External LLMs**: Support for Ollama (local) and external APIs (OpenAI, Anthropic, Google)
- **🔄 Automatic Document Management**: Real-time file watching and smart indexing
- **🗑️ Deletion Handling**: Automatically removes deleted documents from the index
- **🌐 Corporate Portal Integration**: Fallback search with both API and web scraping support
- **🎯 Smart Confidence Scoring**: Determines when to use fallback based on local result quality
- **💬 Multiple Interfaces**: Web UI with management controls, command-line chat, and REST API
- **👀 File System Monitoring**: Automatic detection of document changes using watchdog
- **⚙️ Highly Configurable**: Support for local models, external APIs, and extensive settings
- **📊 Status Reporting**: Real-time indexing status and document synchronization

## 🏗️ Architecture

The system uses a modular architecture with clear separation of concerns:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│ Retrieval Engine│───▶│   Chat Bot      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Vector Store    │    │ Language Model  │
                       │ (ChromaDB)      │    │ (OpenAI GPT)    │
                       └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │Corporate Portal │
                       │   (Fallback)    │
                       └─────────────────┘
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

### Web Interface
The web interface provides a modern chat UI with:
- Real-time messaging with emoji and formatting support
- Source attribution with confidence scores
- System statistics and status monitoring
- **Document management controls**:
  - **"Check Index Status"** - View current indexing state
  - **"Re-index Documents"** - Manual sync with progress feedback
  - Smart sync vs. force rebuild options
- **Automatic file watching** - Real-time detection of document changes
- Responsive design with floating assistant avatar

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

# Get system statistics
curl http://localhost:5001/api/stats
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

### Retrieval Logic

The system uses intelligent retrieval logic:

1. **Primary Search**: Query local vector store
2. **Confidence Evaluation**: Analyze result quality and relevance scores
3. **Fallback Decision**: Use corporate portal if:
   - No high-confidence local results
   - Fewer than 2 relevant local documents
   - Average confidence below threshold
4. **Response Generation**: Combine retrieved content with GPT for final answer

## 🛠️ Development

### Project Structure

```
talos-chatbot/
├── src/
│   ├── __init__.py
│   ├── chatbot.py              # Main RAG orchestrator
│   ├── hybrid_chatbot.py       # Hybrid LLM support (local + external)
│   ├── document_loader.py      # Document processing
│   ├── vector_store.py         # ChromaDB vector operations
│   ├── retrieval.py           # Retrieval logic and fallback
│   ├── corporate_portal.py    # Corporate portal integration
│   ├── ollama_embeddings.py   # Ollama embedding integration
│   ├── auto_indexer.py        # Automatic document indexing
│   └── file_watcher.py        # Real-time file system monitoring
├── templates/
│   └── index.html             # Modern web interface with controls
├── static/
│   └── avatar.svg             # Assistant avatar image
├── config.py                  # Configuration management
├── web_interface.py           # Flask web server with auto-indexing
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

### Chat Endpoint
```
POST /api/chat
Content-Type: application/json

{
  "question": "Your question here",
  "max_sources": 5  // optional
}
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

7. **Python 3.13 setuptools issues**
   ```bash
   # Run the setup script to handle compatibility
   python setup.py
   
   # Or manually upgrade setuptools
   pip install --upgrade pip setuptools wheel
   ```

8. **Package installation failures**
   ```bash
   # Use the installation script (recommended)
   ./install.sh
   
   # Or install dependencies manually
   pip install --upgrade setuptools pip wheel
   pip install langchain langchain-community langchain-ollama
   pip install chromadb sentence-transformers watchdog
   ```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
python main.py --debug web
```

## 📈 Performance Optimization

- **Document Indexing**: Index documents once, query many times
- **Chunk Size**: Balance between context and retrieval speed
- **Similarity Threshold**: Higher thresholds reduce false positives
- **Corporate Portal**: Cache results to reduce external API calls

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

---

**Need help?** Check the troubleshooting section or create an issue on GitHub. 