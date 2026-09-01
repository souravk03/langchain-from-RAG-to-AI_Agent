from langchain_community.document_loaders import DirectoryLoader, PythonLoader

loader = DirectoryLoader(
    path='langchain-document-loaders',
    glob='*.py',
    loader_cls=PythonLoader
)

docs = loader.lazy_load()


for document in docs:
    print(document.metadata)
    