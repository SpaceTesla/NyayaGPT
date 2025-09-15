#!/usr/bin/env python3
"""Setup script for Pinecone configuration."""

import os
from dotenv import load_dotenv

def setup_pinecone():
    """Guide user through Pinecone setup."""
    print("🚀 Pinecone Setup Guide")
    print("=" * 50)
    
    # Load current .env
    load_dotenv()
    
    api_key = os.getenv('PINECONE_API_KEY')
    
    if api_key and api_key != 'your_pinecone_api_key_here':
        print("✅ PINECONE_API_KEY is already set!")
        print(f"Current key: {api_key[:10]}...")
        return True
    
    print("❌ PINECONE_API_KEY not found or not set properly.")
    print("\n📋 Setup Steps:")
    print("1. Go to: https://app.pinecone.io/")
    print("2. Sign up for a free account")
    print("3. Go to 'API Keys' section")
    print("4. Copy your API key")
    print("5. Update your .env file:")
    print("   PINECONE_API_KEY=your_actual_api_key_here")
    print("   PINECONE_ENVIRONMENT=us-east-1")
    print("\n6. Then run: uv run main.py")
    
    return False

if __name__ == "__main__":
    setup_pinecone()
