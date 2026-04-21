from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from utils.local_gemma import LocalGemma4
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever
import streamlit as st

from config import *
from prompts import *

@st.cache_resource
def initialize_rag_system():
    # Vector Store con embeddings locales
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
    
    vectorestore = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH
    )

    # Modelo local para consultas y generación (mismo modelo para ambos)
    llm = LocalGemma4(model_path=LLM_MODEL_PATH, temperature=0)

    # Retriever MMR (Maximal Margin Relevance)
    base_retriever = vectorestore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={
            "k": SEARCH_K,
            "lambda_mult": MMR_DIVERSITY_LAMBDA,
            "fetch_k": MMR_FETCH_K
        }
    )

    # Retriever adicional con similarity para comparar
    similarity_retriever = vectorestore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": SEARCH_K}
    )

    # Prompt personalizado para MultiQueryRetriever
    multi_query_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)

    # MultiQueryRetriever con prompt personalizado
    print("=== GENERANDO VARIACIONES DE PREGUNTA ===")
    mmr_multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        prompt=multi_query_prompt
    )
    print("=== VARIACIONES DE PREGUNTA GENERADAS ===")

    # Ensemble Retriever que combinar MMR y similarity
    if ENABLE_HYBRID_SEARCH:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[mmr_multi_retriever, similarity_retriever],
            weights=[0.7, 0.3], # mayor peso a MMR
            similarity_threshold=SIMILARITY_THRESHOLD
        )
        final_retriever = ensemble_retriever
    else:
        final_retriever = mmr_multi_retriever

    prompt = PromptTemplate.from_template(RAG_TEMPLATE)

    print("=== PROMPT RAG_TEMPLATE ===")
    print(RAG_TEMPLATE)
    print("=== FIN PROMPT RAG_TEMPLATE ===")

    # Funcion para formatear y preprocesar los documentos recuperados
    def format_docs(docs):
        print(f"=== FORMAT_DOCS - RECIBIDOS {len(docs)} DOCUMENTOS ===")
        formatted = []

        for i, doc in enumerate(docs, 1):
            header = f"[Fragmento {i}]"
            
            if doc.metadata:
                if 'source' in doc.metadata:
                    source = doc.metadata['source'].split("\\")[-1] if '\\' in doc.metadata['source'] else doc.metadata['source']
                    header += f" - Fuente: {source}"
                if 'page' in doc.metadata:
                    header += f" - Pagina: {doc.metadata['page']}"
        
            content = doc.page_content.strip()
            formatted.append(f"{header}\n{content}")
            print(f"Fragmento {i}: {content[:200]}...")
        
        result = "\n\n".join(formatted)
        print(f"=== FORMAT_DOCS - RESULTADO (primeros 500 chars): {result[:500]}... ===")
        return result

    rag_chain = (
        {
            "context": final_retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("=== RAG CHAIN CREADO ===")

    return rag_chain, mmr_multi_retriever


def query_rag(question):
    try:
        print("=== INICIANDO QUERY RAG ===")
        print(f"Pregunta: {question}")
        
        rag_chain, retriever = initialize_rag_system()
        print("=== RAG SYSTEM INICIALIZADO ===")

        # Obtener respuesta
        print("=== OBTENIENDO RESPUESTA ===")
        print("=== INVOCANDO RAG CHAIN ===")
        response = rag_chain.invoke(question)
        print(f"=== RESPUESTA OBTENIDA: {response[:200]}... ===")

        # Obtener documentos para mostrarlos (usar invoke en lugar de get_relevant_documents)
        print("=== OBTENIENDO DOCUMENTOS ===")
        docs = retriever.invoke(question)
        print(f"=== DOCUMENTOS OBTENIDOS: {len(docs)} ===")

        # Formatear los documentos para mostrar
        docs_info = []
        for i, doc in enumerate(docs[:SEARCH_K], 1):
            doc_info = {
                "fragmento": i,
                "contenido": doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content,
                "fuente": doc.metadata.get('source', 'No especificada').split("\\")[-1],
                "pagina": doc.metadata.get('page', 'No especificada')
            }
            docs_info.append(doc_info)
        
        print("=== QUERY RAG COMPLETADO ===")
        return response, docs_info
    
    except Exception as e:
        print(f"=== ERROR EN QUERY RAG: {str(e)} ===")
        error_msg = f"Error al procesar la cosulta: {str(e)}"
        return error_msg, []
    
def get_retriever_info():
    """Obtiene información sobre la configuración del retriever"""
    return {
        "tipo": f"{SEARCH_TYPE.upper()} + MultiQuery" + (" + Hybrid" if ENABLE_HYBRID_SEARCH else ""),
        "documentos": SEARCH_K,
        "diversidad": MMR_DIVERSITY_LAMBDA,
        "candidatos": MMR_FETCH_K,
        "umbral": SIMILARITY_THRESHOLD if ENABLE_HYBRID_SEARCH else "N/A"
    }
