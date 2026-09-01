from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import YoutubeLoader
from dotenv import load_dotenv

load_dotenv()

print(len("https://www.youtube.com/watch?v="))
#Step 1a- Indexing (Document Ingestion in the vector Store)

yt_url = input("Enter the Youtube URL: ")
#finding the youtube ID
url_sep = CharacterTextSplitter(chunk_size=32,chunk_overlap=0,separator='=').split_text(yt_url)
video_id = url_sep[1]
print(video_id)

'''try:
    # If you don’t care which language, this returns the “best” one
    transcript_list = YouTubeTranscriptApi().fetch(video_id,languages=['en'])
    # Flatten it to plain text
    print(transcript_list)
    tanscript = " ".join(chunk["text"] for chunk in transcript_list)
    print(tanscript)
except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
    print("Transcript not available for this video.")'''


loader = YoutubeLoader.from_youtube_url(yt_url,add_video_info=False,language=['en'])

docs = loader.load()
print(len(docs))
print(docs[0].page_content)


# Text splitting for chunking the document into smaller pieces
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

print(len(chunks))
print(chunks[0].page_content)

## Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks , embeddings)

#vectorstore.save_local("youtube_rag_index")
#vectorstore = FAISS.load_local("youtube_rag_index", embeddings, allow_dangerous_deserialization=True)
vectorstore.index_to_docstore_id

##Step 2 Retireival

retriever = vectorstore.as_retriever(search_type='similarity',search_kwargs ={'k':4})

result = retriever.invoke('what is logistic regression')
print(result[0].page_content)


#Step 3 Augmentation

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm,temperature=0.3)

prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question} 
    """,
    input_variables=['context','question']
)

question = input('what do you wanna ask?')
retrieved_docs = retriever.invoke(question)

print(retrieved_docs)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
context_text

final_prompt = prompt.invoke({"context": context_text, "question": question})

print(final_prompt)


#Step 4 Generation

answer = model.invoke(final_prompt)

print(answer.content)