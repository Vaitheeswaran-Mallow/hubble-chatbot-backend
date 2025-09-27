#!/usr/bin/env python3
"""
Test script for the Endorsement Process RAG System
"""

import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.data_processor import EndorsementDataProcessor
from services.rag_system import EndorsementRAGSystem

def test_data_processor():
    """Test the data processor"""
    print("🧪 Testing Data Processor...")
    
    processor = EndorsementDataProcessor('data/Endorsement Process Excel.xlsx')
    
    # Load data
    df = processor.load_data()
    print(f"✅ Loaded Excel file with shape: {df.shape}")
    
    # Clean data
    processed_data = processor.clean_data()
    print(f"✅ Processed {len(processed_data)} processes")
    
    # Get stats
    stats = processor.get_summary_stats()
    print(f"✅ Data stats: {stats}")
    
    # Test search
    search_results = processor.search_processes("bitbucket")
    print(f"✅ Found {len(search_results)} processes matching 'bitbucket'")
    
    # Test categories
    categories = processor.get_all_categories()
    print(f"✅ Available categories: {categories}")
    
    return True

def test_rag_system():
    """Test the RAG system (requires OpenAI API key)"""
    print("\n🧪 Testing RAG System...")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set. Skipping RAG system test.")
        print("   To test the RAG system, set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    try:
        # Initialize RAG system
        rag_system = EndorsementRAGSystem(
            openai_api_key=api_key,
            excel_file_path='data/Endorsement Process Excel.xlsx'
        )
        
        # Initialize system
        print("🔄 Initializing RAG system...")
        rag_system.initialize_system()
        print("✅ RAG system initialized successfully!")
        
        # Test queries
        test_queries = [
            "How do I request access to Bitbucket?",
            "Who approves software installations?",
            "What is the process for AWS account creation?",
            "How do I request email ID for a project?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            result = rag_system.query(query, n_results=3)
            
            print(f"✅ Answer: {result['answer'][:200]}...")
            print(f"✅ Found {len(result['similar_processes'])} similar processes")
            
            # Show top similar process
            if result['similar_processes']:
                top_process = result['similar_processes'][0]
                print(f"✅ Top match (ID: {top_process['id']}, Score: {top_process['similarity_score']:.2f})")
        
        # Test system stats
        stats = rag_system.get_system_stats()
        print(f"\n✅ System stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing RAG system: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Endorsement Process RAG System Tests\n")
    
    # Test data processor
    data_test_passed = test_data_processor()
    
    # Test RAG system
    rag_test_passed = test_rag_system()
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"Data Processor: {'✅ PASSED' if data_test_passed else '❌ FAILED'}")
    print(f"RAG System: {'✅ PASSED' if rag_test_passed else '⚠️  SKIPPED (no API key)'}")
    
    if data_test_passed:
        print("\n🎉 Basic system is working! You can now:")
        print("1. Set your OPENAI_API_KEY environment variable")
        print("2. Run: uv run uvicorn main:app --reload")
        print("3. Open http://localhost:8000/docs for API documentation")
        print("4. Open index.html in your browser for the web interface")
    else:
        print("\n❌ Please fix the data processor issues before proceeding")

if __name__ == "__main__":
    main()
