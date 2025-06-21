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
        """Get embedding vector for a query string"""
        if not isinstance(query, str):
            raise TypeError(f"Expected a string for query, got {type(query).__name__}")
        if not query:
            raise ValueError("Query must not be empty")
        return self.embeddings.embed_documents([query])[0]
    
    def get_document_embedding(self, text: str):
        """Get embedding vector for a document text"""
        if not isinstance(text, str):
            raise TypeError(f"Expected a string for text, got {type(text).__name__}")
        if not text:
            raise ValueError("Text must not be empty")
        return self.embeddings.embed_documents([text])[0]
    
    def embed_query(self, query: str):
        """Alias for get_query_embedding to match expected interface"""
        return self.get_query_embedding(query)

    def __init__(self, model_name: str = "models/embedding-001"):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        self.embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
    
    def get_query_embedding(self, query: str):
      if not isinstance(query, str):
        raise TypeError(f"Expected a string for query, got {type(query).__name__}")
      if not query:
        raise ValueError("Query must not be empty")
      return self.embeddings.embed_documents([query])[0]

    def get_document_embedding(self, text: str):
     if not isinstance(text, str):
        raise TypeError(f"Expected a string for text, got {type(text).__name__}")
     if not text:
        raise ValueError("Text must not be empty")
     return self.embeddings.embed_documents([text])[0]

    def embed_query(self, query: str):
        """
        Alias for get_query_embedding to match expected interface.
        """
        return self.get_query_embedding(query)