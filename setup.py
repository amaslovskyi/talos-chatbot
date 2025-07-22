#!/usr/bin/env python3
"""
Setup script for RAG Chatbot.
Alternative to install.sh for Python-based setup.
Helps with Python 3.13 compatibility and installation issues.
"""

import sys
import subprocess
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    elif sys.version_info >= (3, 13):
        print("⚠️  Python 3.13 detected - using compatibility mode")
        return True
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        return False


def install_setuptools():
    """Install or upgrade setuptools for Python 3.13 compatibility."""
    print("📦 Installing/upgrading setuptools...")
    try:
        # Check if we're in a virtual environment
        if hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        ):
            print("✅ Virtual environment detected - installing in venv")
        else:
            print("⚠️  Not in virtual environment - consider using ./talos/bin/activate")

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "setuptools",
                "pip",
                "wheel",
            ]
        )
        print("✅ setuptools upgraded successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to upgrade setuptools: {e}")
        print(
            "💡 Try activating virtual environment first: source ./talos/bin/activate"
        )
        return False
    return True


def install_requirements():
    """Install requirements with compatibility handling."""
    print("📦 Installing requirements...")

    # Try installing requirements
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        print("🔧 Trying alternative installation method...")

        # Try installing packages individually with more permissive versions
        fallback_packages = [
            "langchain",
            "langchain-community",
            "langchain-openai",
            "openai",
            "chromadb",
            "sentence-transformers",
            "pypdf",
            "python-docx",
            "beautifulsoup4",
            "markdown",
            "flask",
            "flask-cors",
            "requests",
            "aiohttp",
            "python-dotenv",
            "pydantic",
            "numpy",
            "tqdm",
        ]

        failed_packages = []
        for package in fallback_packages:
            try:
                print(f"Installing {package}...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package, "--no-deps"]
                )
            except subprocess.CalledProcessError:
                failed_packages.append(package)
                print(f"⚠️  Failed to install {package}")

        if failed_packages:
            print(f"❌ Failed to install: {', '.join(failed_packages)}")
            print("💡 Try installing these manually or use a different Python version")
            return False

        print("✅ Fallback installation completed")
        return True


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")

    directories = ["documents", "vector_db", "templates"]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created: {directory}")


def create_env_template():
    """Create .env template if it doesn't exist."""
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating .env template...")

        env_template = """# RAG Chatbot Configuration

# Required: OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Corporate Portal Configuration  
CORPORATE_PORTAL_URL=https://your-company-portal.com
CORPORATE_PORTAL_API_KEY=your_portal_api_key_here
CORPORATE_PORTAL_USERNAME=your_username
CORPORATE_PORTAL_PASSWORD=your_password

# Optional: Custom paths and settings
DOCS_DIRECTORY=./documents
VECTOR_DB_PATH=./vector_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
SIMILARITY_THRESHOLD=0.7
MAX_DOCS_TO_RETRIEVE=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=100

# Optional: Flask settings
FLASK_HOST=127.0.0.1
FLASK_PORT=5000
FLASK_DEBUG=True
"""

        with open(".env", "w") as f:
            f.write(env_template)

        print("✅ Created .env template")
        print("📝 Please edit .env with your actual configuration")
    else:
        print("✅ .env file already exists")


def main():
    """Main setup function."""
    print("🚀 RAG Chatbot Setup")
    print("=" * 40)

    # Check Python version
    is_python_313 = check_python_version()

    # Install/upgrade setuptools if needed
    if is_python_313:
        if not install_setuptools():
            print("❌ Setup failed - could not install setuptools")
            sys.exit(1)

    # Install requirements
    if not install_requirements():
        print("❌ Setup failed - could not install requirements")
        print("💡 Try running: pip install --upgrade pip setuptools wheel")
        sys.exit(1)

    # Create directories
    create_directories()

    # Create .env template
    create_env_template()

    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Activate virtual environment (if not already active):")
    print("   source ./talos/bin/activate")
    print("2. Edit .env file with your OpenAI API key")
    print("3. Add documents to ./documents/ directory")
    print("4. Run: python main.py web")
    print("\n💡 For help: python main.py config")


if __name__ == "__main__":
    main()
