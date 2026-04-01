from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embadding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)
documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]
result = embadding.embed_documents(documents)

print(str(result))