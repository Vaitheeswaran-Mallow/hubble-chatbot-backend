# Endorsement Process RAG System

A Retrieval-Augmented Generation (RAG) system built with FastAPI, OpenAI, and ChromaDB to answer questions about endorsement processes from an Excel file.

## Features

- 📊 **Data Processing**: Automatically processes Excel files with endorsement processes
- 🔍 **Semantic Search**: Uses OpenAI embeddings for intelligent document retrieval
- 🤖 **AI-Powered Answers**: Generates contextual answers using OpenAI GPT models
- 🌐 **REST API**: FastAPI-based API with comprehensive endpoints
- 🎨 **Web Interface**: Simple HTML frontend for testing queries
- 📈 **Analytics**: System statistics and process categorization

## System Architecture

```
Excel File → Data Processor → Vector Database (ChromaDB) → RAG System → FastAPI → Web Interface
                                    ↓
                              OpenAI Embeddings
                                    ↓
                              OpenAI GPT-3.5
```

## Installation

1. **Clone and navigate to the project**:

   ```bash
   cd /Users/yuvaraj/python/hubble-chatbot-backend
   ```

2. **Install dependencies** (already done):

   ```bash
   uv add pandas openai chromadb openpyxl python-multipart
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```bash
   OPEN_API_KEY=your-openai-api-key-here
   ```

## Usage

### 1. Test the System

Run the test script to verify everything works:

```bash
uv run python test_system.py
```

### 2. Start the API Server

```bash
uv run uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### 3. Access the Web Interface

Open `index.html` in your browser to use the web interface.

### 4. API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

## API Endpoints

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

## Example Queries

Try these example questions:

- "How do I request access to Bitbucket?"
- "Who approves software installations?"
- "What is the process for AWS account creation?"
- "How do I request email ID for a project?"
- "Who can approve network restrictions?"
- "What is the process for team change requests?"

## Data Structure

The system processes Excel files with the following structure:

| Column                      | Description                    |
| --------------------------- | ------------------------------ |
| S. No                       | Process ID                     |
| Management of               | Process category/area          |
| Actions                     | What actions are involved      |
| Initiator                   | Who initiates the request      |
| Approver                    | Who approves the request       |
| Executer                    | Who executes the action        |
| Post Execution Confirmation | Confirmation process           |
| Email Details               | Email templates and recipients |

## Process Categories

The system automatically categorizes processes into:

- **software_management**: Application installations, software upgrades
- **network_access**: Network restrictions, device connections
- **email_management**: Email ID creation, forwarders
- **development_tools**: Bitbucket, AWS, Firebase access
- **finance_certification**: Certifications, purchases
- **hr_management**: Team changes, work schedules
- **general**: Other processes

## Configuration

### Environment Variables

- `OPEN_API_KEY`: Your OpenAI API key (required for RAG functionality)

### File Paths

The system expects the Excel file at:
`data/Endorsement Process Excel.xlsx`

To change this, modify the `excel_file_path` parameter in:

- `src/config.py` (line 17)
- `tests/test_system.py` (line 19)

## Troubleshooting

### Common Issues

1. **"RAG system not initialized"**

   - Check if `OPEN_API_KEY` is set correctly
   - Verify the Excel file path exists
   - Check server logs for initialization errors

2. **"OpenAI API key not configured"**

   - Set the `OPEN_API_KEY` environment variable
   - Restart the server after setting the key

3. **"Excel file not found"**
   - Verify the file path in the configuration
   - Ensure the file exists and is readable

### Logs

Check the server logs for detailed error information:

```bash
uv run uvicorn main:app --reload --log-level debug
```

## Development

### Project Structure

```
hubble-chatbot-backend/
├── main.py              # FastAPI application
├── rag_system.py        # RAG system implementation
├── data_processor.py    # Excel data processing
├── settings.py          # Configuration
├── test_system.py       # Test script
├── index.html           # Web interface
├── pyproject.toml       # Dependencies
└── chroma_db/          # Vector database (created automatically)
```

### Adding New Features

1. **New Data Sources**: Modify `data_processor.py` to handle different file formats
2. **New Embedding Models**: Update `rag_system.py` to use different embedding models
3. **New API Endpoints**: Add endpoints in `main.py`
4. **Enhanced UI**: Modify `index.html` for better user experience

## Performance

- **Vector Database**: ChromaDB with DuckDB backend for fast similarity search
- **Embeddings**: OpenAI text-embedding-3-small for cost-effective embeddings
- **Generation**: GPT-3.5-turbo for balanced performance and cost
- **Caching**: ChromaDB persists embeddings between sessions

## Security

- API keys are stored in environment variables
- CORS is enabled for development (configure appropriately for production)
- Input validation through Pydantic models
- Error handling prevents sensitive information leakage

## License

This project is part of the Hubble Chatbot Backend system.
