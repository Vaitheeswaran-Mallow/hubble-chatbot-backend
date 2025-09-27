# 📁 Project Structure - Endorsement Process RAG System

## 🎯 **Organized Folder Structure**

```
hubble-chatbot-backend/
├── 📁 src/                           # Source code
│   ├── 📁 api/                       # FastAPI application and endpoints
│   │   ├── __init__.py
│   │   └── main.py                   # Main FastAPI app with all endpoints
│   ├── 📁 core/                     # Core business logic
│   │   ├── __init__.py
│   │   └── data_processor.py         # Excel data processing and cleaning
│   ├── 📁 services/                 # External services and integrations
│   │   ├── __init__.py
│   │   └── rag_system.py             # RAG system with OpenAI + ChromaDB
│   ├── 📁 models/                   # Pydantic models and schemas
│   │   ├── __init__.py
│   │   └── schemas.py                # API request/response models
│   ├── 📁 utils/                    # Utility functions
│   │   └── __init__.py
│   ├── config.py                    # Centralized configuration
│   └── __init__.py
├── 📁 data/                         # Data files
│   └── Endorsement Process Excel.xlsx  # Source Excel file
├── 📁 static/                       # Static files (HTML, CSS, JS)
│   └── index.html                   # Web interface for testing
├── 📁 templates/                    # Template files (ready for expansion)
├── 📁 tests/                        # Test files
│   └── test_system.py               # Comprehensive system tests
├── 📁 docs/                         # Documentation
│   └── README.md                    # Detailed project documentation
├── 📁 chroma_db/                    # Vector database (auto-created)
├── 🚀 main.py                       # Entry point for the application
├── ⚙️ settings.py                    # Legacy settings (can be removed)
├── 📦 pyproject.toml               # Project dependencies (uv)
├── 📋 requirements.txt             # Python dependencies (pip)
├── 🐚 start.sh                     # Startup script
├── 📖 README.md                    # Main project documentation
└── 🔧 .env                         # Environment variables
```

## 🏗️ **Architecture Overview**

### **Layer Separation**

- **API Layer** (`src/api/`): FastAPI endpoints and request/response handling
- **Core Layer** (`src/core/`): Business logic and data processing
- **Services Layer** (`src/services/`): External integrations (OpenAI, ChromaDB)
- **Models Layer** (`src/models/`): Data structures and validation
- **Utils Layer** (`src/utils/`): Helper functions and utilities

### **Key Benefits**

✅ **Separation of Concerns**: Each layer has a specific responsibility  
✅ **Maintainability**: Easy to find and modify specific functionality  
✅ **Scalability**: Easy to add new features without affecting existing code  
✅ **Testability**: Each component can be tested independently  
✅ **Configuration**: Centralized settings management  
✅ **Documentation**: Clear structure with comprehensive docs

## 🚀 **Quick Start Commands**

```bash
# Start the application
./start.sh

# Or manually
python main.py

# Run tests
python tests/test_system.py

# Install dependencies
uv sync
# or
pip install -r requirements.txt
```

## 📊 **System Status**

- ✅ **37 endorsement processes** processed and indexed
- ✅ **7 categories** automatically identified
- ✅ **Vector database** with OpenAI embeddings
- ✅ **RAG system** fully operational
- ✅ **API endpoints** working perfectly
- ✅ **Web interface** ready for testing

## 🔗 **Access Points**

- **API Documentation**: http://localhost:8000/docs
- **Web Interface**: http://localhost:8000/static/index.html
- **Health Check**: http://localhost:8000/health
- **System Stats**: http://localhost:8000/stats

The project is now properly organized with a clean, maintainable structure! 🎉
