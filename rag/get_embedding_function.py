from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv("C://Users//user//Desktop//LLM_understands//Agent-Memory-CHI24//config//.env")

def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return embeddings
