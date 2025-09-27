# Endorsement Process RAG System

A Retrieval-Augmented Generation (RAG) system built with FastAPI, OpenAI, and ChromaDB to answer questions about endorsement processes from an Excel file.

## 📁 Project Structure

```
hubble-chatbot-backend/
├── src/                           # Source code
│   ├── api/                       # FastAPI application and endpoints
│   │   ├── __init__.py
│   │   └── main.py               # Main FastAPI app
│   ├── core/                     # Core business logic
│   │   ├── __init__.py
│   │   └── data_processor.py     # Excel data processing
│   ├── services/                 # External services and integrations
│   │   ├── __init__.py
│   │   └── rag_system.py         # RAG system implementation
│   ├── models/                   # Pydantic models and schemas
│   │   ├── __init__.py
│   │   └── schemas.py            # API request/response models
│   ├── utils/                    # Utility functions
│   │   └── __init__.py
│   ├── config.py                 # Configuration settings
│   └── __init__.py
├── data/                         # Data files
│   └── Endorsement Process Excel.xlsx  # Source Excel file
├── static/                       # Static files (HTML, CSS, JS)
│   └── index.html                # Web interface
├── templates/                    # Template files (if needed)
├── tests/                        # Test files
│   └── test_system.py           # System tests
├── docs/                         # Documentation
├── chroma_db/                   # Vector database (auto-created)
├── main.py                      # Entry point
├── settings.py                  # Legacy settings (can be removed)
├── pyproject.toml              # Project dependencies
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .env                        # Environment variables
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file:

```bash
OPEN_API_KEY=your-openai-api-key-here
```

### 3. Run the Application

```bash
# Using the main entry point
python main.py

# Or directly with uvicorn
uv run uvicorn src.api.main:app --reload
```

### 4. Access the Application

- **API Documentation**: http://localhost:8000/docs
- **Web Interface**: http://localhost:8000/static/index.html
- **Health Check**: http://localhost:8000/health

## 🔧 Configuration

All configuration is managed through `src/config.py`. Key settings:

- **OpenAI API Key**: Set via `OPEN_API_KEY` environment variable
- **Excel File Path**: Configured in `excel_file_path`
- **ChromaDB Path**: Configured in `chroma_db_path`
- **API Settings**: Host, port, title, etc.
- **RAG Settings**: Models, dimensions, temperature, etc.

## 📊 API Endpoints

### Core Endpoints

- `GET /` - System status
- `GET /health` - Health check
- `POST /query` - Query the RAG system (JSON)
- `GET /query?question=...` - Query the RAG system (URL params)

### Data Endpoints

- `GET /processes` - Get all processes
- `GET /process/{id}` - Get specific process by ID
- `GET /categories` - Get all process categories
- `GET /processes/category/{category}` - Get processes by category
- `GET /stats` - Get system statistics

## 🧪 Testing

Run the test suite:

```bash
python tests/test_system.py
```

## 📈 Features

- **37 endorsement processes** automatically processed
- **7 categories** of processes
- **Semantic search** using OpenAI embeddings
- **AI-powered answers** using GPT-3.5-turbo
- **Vector database** for fast similarity search
- **REST API** with comprehensive endpoints
- **Web interface** for easy testing
- **Configurable settings** for all components

## 🔍 Example Queries

- "How do I request access to Bitbucket?"
- "Who approves software installations?"
- "What is the process for AWS account creation?"
- "How do I request email ID for a project?"

## 🛠️ Development

### Adding New Features

1. **New Data Sources**: Modify `src/core/data_processor.py`
2. **New API Endpoints**: Add to `src/api/main.py`
3. **New Models**: Add to `src/models/schemas.py`
4. **New Services**: Add to `src/services/`

### Code Organization

- **API Layer**: FastAPI endpoints and request/response handling
- **Core Layer**: Business logic and data processing
- **Services Layer**: External integrations (OpenAI, ChromaDB)
- **Models Layer**: Data structures and validation
- **Utils Layer**: Helper functions and utilities

## 📝 License

This project is part of the Hubble Chatbot Backend system.
