from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm, temperature=0, max_output_tokens=200)

result = model.invoke("Write a 5 line poem on cricket")

print(result.content)