#!/bin/bash

# Endorsement Process RAG System Startup Script

echo "🚀 Starting Endorsement Process RAG System..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    uv venv
fi

# Install dependencies
echo "📥 Installing dependencies..."
uv sync

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Creating .env file template..."
    cat > .env << EOF
# OpenAI API Key (required)
OPEN_API_KEY=your-openai-api-key-here

# Optional: Override default Excel file path
# EXCEL_FILE_PATH=/path/to/your/excel/file.xlsx
EOF
    echo "📝 Please edit .env file and add your OpenAI API key"
    echo "   Then run this script again"
    exit 1
fi

# Check if API key is set
if grep -q "your-openai-api-key-here" .env; then
    echo "❌ Please set your OpenAI API key in .env file"
    exit 1
fi

# Start the application
echo "🎯 Starting FastAPI server..."
echo "   API Documentation: http://localhost:8000/docs"
echo "   Web Interface: http://localhost:8000/static/index.html"
echo "   Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python main.py
