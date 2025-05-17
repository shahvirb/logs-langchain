import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import logging

class GoogleFactory:
    def __init__(self, modelstr="gemini-2.0-flash", embedding_model="models/embedding-001"):
        load_dotenv()
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            print("Error: GOOGLE_API_KEY not found in environment variables.")
            print(
                "Please make sure you have a .env file with GOOGLE_API_KEY='YOUR_KEY' in it."
            )
            exit()
        logging.info("GOOGLE_API_KEY successfully loaded from environment.")
        self.modelstr = modelstr
        self.embedding_model = embedding_model

    def llm(self):
        return ChatGoogleGenerativeAI(model=self.modelstr)

    def embeddings(self):
        return GoogleGenerativeAIEmbeddings(model=self.embedding_model)
