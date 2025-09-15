from utils.chat_interface import NyayaChatInterface


def main():
    print("🚀 NyayaGPT - Indian Constitution RAG Agent")
    print("=" * 50)
    print("📚 Your AI assistant for Indian Constitution queries")
    print("💡 Type 'quit' or 'exit' to end the conversation")
    print("=" * 50)
    
    # Start the chat interface
    chat = NyayaChatInterface()
    chat.start_chat()


if __name__ == "__main__":
    main()
