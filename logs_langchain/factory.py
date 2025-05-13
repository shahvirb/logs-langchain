import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import logging

def llm_factory(modelstr="gemini-2.0-flash"):
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        # TODO this function should adapt based on what API key has been set
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please make sure you have a .env file with GOOGLE_API_KEY='YOUR_KEY' in it.")
        exit()

    logging.info("GOOGLE_API_KEY successfully loaded from environment.")

    llm = ChatGoogleGenerativeAI(model=modelstr)
    return llm