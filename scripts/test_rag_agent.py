#!/usr/bin/env python3
"""Test script for the RAG agent."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.rag_agent import NyayaRAGAgent
from utils.chat_interface import NyayaChatInterface


def test_rag_agent():
    """Test the RAG agent with sample questions."""
    print("🧪 Testing NyayaGPT RAG Agent")
    print("=" * 40)
    
    try:
        # Initialize agent
        print("🔧 Initializing agent...")
        agent = NyayaRAGAgent()
        print("✅ Agent initialized successfully!")
        
        # Test questions
        test_questions = [
            "What are fundamental rights in the Indian Constitution?",
            "What is Article 21 about?",
            "What are the fundamental duties of citizens?",
            "What is the structure of the Indian Constitution?",
            "What is the preamble of the Constitution?"
        ]
        
        print(f"\n📝 Testing {len(test_questions)} questions...")
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*60}")
            print(f"🤔 Question {i}: {question}")
            print("🤖 Answer:")
            
            try:
                answer = agent.ask(question)
                print(answer)
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print("-" * 60)
        
        print(f"\n🎉 Test completed! All {len(test_questions)} questions processed.")
        
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        print("💡 Make sure you have set GOOGLE_API_KEY in your .env file")


def test_chat_interface():
    """Test the chat interface."""
    print("\n💬 Testing Chat Interface")
    print("=" * 30)
    
    try:
        chat = NyayaChatInterface()
        print("✅ Chat interface ready!")
        print("💡 You can now start chatting. Type 'quit' to exit.")
        chat.start_chat()
    except Exception as e:
        print(f"❌ Error with chat interface: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test NyayaGPT RAG Agent")
    parser.add_argument("--mode", choices=["test", "chat"], default="test",
                       help="Mode: 'test' for automated testing, 'chat' for interactive chat")
    
    args = parser.parse_args()
    
    if args.mode == "test":
        test_rag_agent()
    elif args.mode == "chat":
        test_chat_interface()
