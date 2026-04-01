from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceTB/SmolLM3-3B",
    task="text2text-generation",
    max_new_tokens=10
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("What is capital of india")
print(result.content)
print(result.content)

