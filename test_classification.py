#!/usr/bin/env python3
"""
Test script to demonstrate the new question classification and database query functionality
"""

import asyncio
import sys
import os

# Add src to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.services.question_classifier import question_classifier
from src.services.answer_service import AnswerService
from src.services.database_service import db_service

async def test_classification():
    """Test the question classification functionality"""
    
    print("=== Testing Question Classification ===\n")
    
    # Test questions
    test_questions = [
        "What is the leave policy for sick days?",
        "Show me all employees in the engineering team",
        "What are the company holidays for 2024?",
        "How do I apply for work from home?",
        "What is John Smith's leave balance?",
        "Show me my timesheet entries for this month",
        "What projects is Sarah Johnson working on?",
        "What is the office etiquette policy?",
        "How many sick leaves does Mike have?",
        "What are the guidelines for meeting rooms?"
    ]
    
    for question in test_questions:
        print(f"Question: {question}")
        classification = question_classifier.classify_question(question)
        print(f"Type: {classification['query_type']}")
        print(f"Confidence: {classification['confidence']}")
        print(f"Reasoning: {classification['reasoning']}")
        print("-" * 50)

async def test_database_service():
    """Test the database service functionality"""
    
    print("\n=== Testing Database Service ===\n")
    
    try:
        # Test schema info
        schema_info = db_service.get_schema_info()
        print("Schema info retrieved successfully")
        print(f"Schema length: {len(schema_info)} characters")
        
        # Test a simple query (this will fail if DB is not connected)
        try:
            result = db_service.execute_query("SELECT 1 as test")
            print(f"Database connection test: {result}")
            
            # Test a more complex query if basic one works
            try:
                result2 = db_service.execute_query("SELECT name FROM sqlite_master WHERE type='table' LIMIT 5")
                print(f"Available tables: {[row['name'] for row in result2]}")
            except Exception as e2:
                print(f"Table listing failed: {e2}")
                
        except Exception as e:
            print(f"Database connection failed (expected if not configured): {e}")
            print("Note: Make sure hubble.db file exists in the project root")
        
    except Exception as e:
        print(f"Database service error: {e}")

async def test_answer_service():
    """Test the answer service with classification"""
    
    print("\n=== Testing Answer Service ===\n")
    
    answer_service = AnswerService()
    
    # Test questions
    test_questions = [
        "What is the leave policy?",
        "Show me employee information",
        "What are the company holidays?"
    ]
    
    for question in test_questions:
        print(f"Question: {question}")
        try:
            # This will use the new classification system
            answer_data = answer_service.classify_and_answer(question)
            print(f"Answer type: {answer_data.get('query_type', 'unknown')}")
            print(f"Confidence: {answer_data.get('confidence', 'unknown')}")
            print(f"Answer preview: {answer_data.get('answer', 'No answer')[:100]}...")
            print("-" * 50)
        except Exception as e:
            print(f"Error processing question: {e}")
            print("-" * 50)

async def main():
    """Main test function"""
    print("Testing Hubble Chatbot Backend - New Classification System")
    print("=" * 60)
    
    await test_classification()
    await test_database_service()
    await test_answer_service()
    
    print("\n=== Test Complete ===")
    print("Note: Database tests will fail if the database is not configured.")
    print("The classification system should work regardless of database connectivity.")

if __name__ == "__main__":
    asyncio.run(main())
