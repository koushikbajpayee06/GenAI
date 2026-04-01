from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


load_dotenv()
embadding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)
documents = [
    "Delhi is the capital of India",
    "Paris is the capital of France",
    "Tokyo is the capital of Japan",
    "Cricket is a popular sport in India",
    "Football is widely played in Europe",
    "Python is a programming language",
    "Laravel is a PHP framework",
    "React is used for building user interfaces",
    "Artificial Intelligence is transforming technology",
    "Machine Learning is a subset of AI"
]
query = "Tell me about popular sports in europe"
doc_embeddings = embadding.embed_documents(documents)
query_embedding = embadding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]
index,score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]
print(query)
print(documents[index])
print("Similarity score is: ",score)
