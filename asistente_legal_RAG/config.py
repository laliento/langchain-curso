# Configuración de modelos
EMBEDDING_MODEL_PATH = "C:\\Users\\laptop\\git\\langchain-curso\\embeddings\\all-MiniLM-L6-v2"
LLM_MODEL_PATH = "C:\\Users\\laptop\\git\\langchain-curso\\models"

# Configuración del vector store
CHROMA_DB_PATH = "C:\\Users\\laptop\\git\\langchain-curso\\asistente_legal_RAG\\chroma_db"

# Configuración del retriever
SEARCH_TYPE = "mmr"
MMR_DIVERSITY_LAMBDA = 0.7
MMR_FETCH_K = 20
SEARCH_K = 2

# Configuracion alternativa para retriever hibrido
ENABLE_HYBRID_SEARCH = True
SIMILARITY_THRESHOLD = 0.70