from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('langchain-document-loaders\\dl-curriculum.pdf')

docs = loader.load()

print(len(docs))

print(docs[20].page_content)
print(docs[1].metadata)