# embedding_service.py

import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class EmbeddingService:
    def __init__(self, model_name: str = "models/embedding-001"):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        self.embeddings = GoogleGenerativeAIEmbeddings(model=model_name)

    def get_query_embedding(self, query: str):
        """
        Returns the embedding for a given query string.
        """
        if not query:
            raise ValueError("Query must not be empty")
        return self.embeddings.embed_query(query)

    def get_document_embedding(self, text: str):
        """
        Returns the embedding for a document.
        """
        if not text:
            raise ValueError("Text must not be empty")
        return self.embeddings.embed_documents([text])[0]  # Returns a single embedding

