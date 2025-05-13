import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

def llm_factory():
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please make sure you have a .env file with GOOGLE_API_KEY='YOUR_KEY' in it.")
        exit()

    print("GOOGLE_API_KEY successfully loaded from environment.")

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    return llm