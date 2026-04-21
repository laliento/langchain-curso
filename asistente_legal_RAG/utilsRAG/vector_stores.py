from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Cargar documentos PDF
loader = PyPDFDirectoryLoader("C:\\Users\\laptop\\git\\langchain-curso\\asistente_legal_RAG\\contratos")
documentos = loader.load()

print(f"Se cargaron {len(documentos)} documentos desde el directorio.")

# Dividir documentos en chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=1000
)

docs_split = text_splitter.split_documents(documentos)

print(f"Se crearon {len(docs_split)} chunks de texto.")

# Crear embeddings locales con all-MiniLM-L6-v2
embeddings = HuggingFaceEmbeddings(
    model_name="C:\\Users\\laptop\\git\\langchain-curso\\embeddings\\all-MiniLM-L6-v2"
)

# Crear vector store con Chroma
vectorstore = Chroma.from_documents(
    docs_split,
    embedding=embeddings,
    persist_directory="C:\\Users\\laptop\\git\\langchain-curso\\asistente_legal_RAG\\chroma_db"
)

# Realizar búsqueda
consulta = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos"

resultados = vectorstore.similarity_search(consulta, k=2)

print("Top 3 documentos mas similares a la consulta:\n")
for i, doc in enumerate(resultados, start=1):
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")
