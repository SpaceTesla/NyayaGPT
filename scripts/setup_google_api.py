#!/usr/bin/env python3
"""Setup script for Google Generative AI API key."""

import os
from dotenv import load_dotenv

def setup_google_api():
    """Guide user through Google API setup."""
    print("🔑 Google Generative AI API Setup Guide")
    print("=" * 50)
    
    # Load current .env
    load_dotenv()
    
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if api_key and api_key != 'your_google_api_key_here':
        print("✅ GOOGLE_API_KEY is already set!")
        print(f"Current key: {api_key[:10]}...")
        return True
    
    print("❌ GOOGLE_API_KEY not found or not set properly.")
    print("\n📋 Setup Steps:")
    print("1. Go to: https://aistudio.google.com/app/apikey")
    print("2. Sign in with your Google account")
    print("3. Click 'Create API Key'")
    print("4. Copy the generated API key")
    print("5. Update your .env file:")
    print("   GOOGLE_API_KEY=your_actual_api_key_here")
    print("\n6. Then run: uv run main.py")
    print("\n💡 The API key should look like: AIzaSy...")
    
    return False

if __name__ == "__main__":
    setup_google_api()
