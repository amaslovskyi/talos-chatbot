#!/bin/bash

# RAG Chatbot Installation Script
# Handles virtual environment setup and dependency installation

set -e  # Exit on any error

echo "🚀 RAG Chatbot Installation"
echo "=============================="

# Check if Python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python $PYTHON_VERSION detected"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment already active: $VIRTUAL_ENV"
    VENV_DIR="$VIRTUAL_ENV"
else
    # Check for existing virtual environments
    if [ -d "./talos" ]; then
        VENV_DIR="./talos"
        echo "📦 Found existing virtual environment: $VENV_DIR"
    elif [ -d "./talos-env" ]; then
        VENV_DIR="./talos-env"
        echo "📦 Found existing virtual environment: $VENV_DIR"
    else
        # Create new virtual environment
        VENV_DIR="./talos-env"
        echo "📦 Creating virtual environment: $VENV_DIR"
        python3 -m venv "$VENV_DIR"
        echo "✅ Virtual environment created"
    fi
    
    # Activate virtual environment
    echo "🔄 Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
fi

# Upgrade pip and setuptools
echo "📦 Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel

# Install requirements
echo "📦 Installing Python packages..."
if [ -f "requirements.txt" ]; then
    python -m pip install -r requirements.txt
    echo "✅ Requirements installed successfully"
else
    echo "⚠️  requirements.txt not found, installing packages individually..."
    
    # Core packages
    python -m pip install langchain langchain-community langchain-openai
    python -m pip install openai chromadb sentence-transformers
    python -m pip install pypdf python-docx beautifulsoup4 markdown
    python -m pip install flask flask-cors requests aiohttp
    python -m pip install python-dotenv pydantic numpy tqdm
    echo "✅ Packages installed successfully"
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p documents vector_db templates
echo "✅ Directories created"

# Create .env template
if [ ! -f ".env" ]; then
    echo "📝 Creating .env template..."
    cat > .env << 'EOF'
# RAG Chatbot Configuration

# =========================
# LLM Configuration
# =========================
# Use local Ollama models by default
USE_LOCAL_LLM=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3n:latest

# OpenAI API (only needed if USE_LOCAL_LLM=false)
OPENAI_API_KEY=your_openai_api_key_here

# =========================
# Embedding Configuration  
# =========================
# Use local Ollama embeddings by default
USE_LOCAL_EMBEDDINGS=true
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest

# Fallback embedding model (sentence-transformers)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# =========================
# External API Fallback
# =========================
# Enable external API when local models fail
ENABLE_EXTERNAL_API_FALLBACK=true
EXTERNAL_API_PROVIDER=openai  # openai, anthropic, google
EXTERNAL_API_KEY=your_external_api_key_here
EXTERNAL_API_MODEL=gpt-3.5-turbo

# =========================
# Corporate Portal Configuration
# =========================
CORPORATE_PORTAL_URL=https://www.andrewng.org/
CORPORATE_PORTAL_API_KEY=your_portal_api_key_here
CORPORATE_PORTAL_USERNAME=your_username
CORPORATE_PORTAL_PASSWORD=your_password

# =========================
# Document Processing
# =========================
DOCS_DIRECTORY=./documents
VECTOR_DB_PATH=./vector_db
SIMILARITY_THRESHOLD=0.7
MAX_DOCS_TO_RETRIEVE=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=100

# =========================
# Flask Web Interface
# =========================
FLASK_HOST=127.0.0.1
FLASK_PORT=5000
FLASK_DEBUG=True
EOF
    echo "✅ .env template created"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your OpenAI API key:"
echo "   nano .env"
echo ""
echo "2. If not in virtual environment, activate it:"
echo "   source ./talos/bin/activate     # or ./talos-env/bin/activate"
echo ""
echo "3. Add documents to ./documents/ directory"
echo ""
echo "4. Start the chatbot:"
echo "   python main.py web      # Web interface"
echo "   python main.py chat     # Command line"
echo "   python main.py config   # Check configuration"
echo ""
echo "💡 For more help, see README.md" 