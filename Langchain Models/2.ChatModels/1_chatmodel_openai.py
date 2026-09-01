from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model='gpt-4')
result = llm.invoke('what is the date today',temperature=0.5, max_tokens=100) 
#temperature means how creative the model should be in its responses. A higher temperature will result in more creative responses, while a lower temperature will result in more conservative responses
#max_tokens means the maximum number of tokens (words or word pieces) that the model can generate in its response. This can be used to limit the length of the response and prevent it from generating excessively long outputs.
print(result)
print(result.content)
