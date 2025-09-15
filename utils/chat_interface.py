"""Simple chat interface for NyayaGPT RAG agent."""

import os
import sys
from typing import Optional
from .rag_agent import NyayaRAGAgent

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class NyayaChatInterface:
    """Simple chat interface for NyayaGPT."""
    
    def __init__(self):
        """Initialize the chat interface."""
        try:
            self.agent = NyayaRAGAgent()
            print("✅ NyayaGPT initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing NyayaGPT: {e}")
            print("💡 Make sure you have set GOOGLE_API_KEY in your .env file")
            sys.exit(1)
    
    def start_chat(self):
        """Start the interactive chat session."""
        while True:
            try:
                # Get user input
                question = input("\n🤔 Your question: ").strip()
                
                # Check for exit commands
                if question.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Thank you for using NyayaGPT! Goodbye!")
                    break
                
                # Skip empty questions
                if not question:
                    continue
                
                # Get response
                print("\n🤖 NyayaGPT is thinking...")
                response = self.agent.ask(question)
                
                # Display response
                print(f"\n📖 Answer:\n{response}")
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("💡 Please try again or type 'quit' to exit")
    
    def ask_single_question(self, question: str) -> str:
        """Ask a single question and return the response."""
        try:
            return self.agent.ask(question)
        except Exception as e:
            return f"Error: {e}"
    
    def get_detailed_response(self, question: str) -> dict:
        """Get detailed response with context and sources."""
        try:
            return self.agent.chat(question)
        except Exception as e:
            return {"error": str(e)}


def main():
    """Main function to run the chat interface."""
    chat = NyayaChatInterface()
    chat.start_chat()


if __name__ == "__main__":
    main()
