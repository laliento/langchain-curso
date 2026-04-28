from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import *
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFDirectoryLoader(r"C:\Users\laptop\git\langchain-curso\dif_memoria\contratos")
documentos = loader.load()

print(f"Se cargaron {len(documentos)} documentos desde el directorio.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=1000
)

docs_split = text_splitter.split_documents(documentos)

print(f"Se crearon {len(docs_split)} chunks de texto.")


# Crear vectorstore
vectorstore = Chroma.from_documents(
    docs_split,
    embedding=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH),
    collection_name="example_contratos",
    persist_directory="C:\\Users\\laptop\\git\\langchain-curso\\dif_memoria\\chroma_db"
)
#vectorstore = Chroma.from_documents(
#    docs_split,
#    embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
#    persist_directory="C:\\Users\\santiago\\curso_langchain\\Tema 3\\chroma_db"
#)

consulta = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos"

resultados = vectorstore.similarity_search(consulta, k=2)

print("Top 3 documentos mas similares a la consulta:\n")
for i, doc in enumerate(resultados, start=1):
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")