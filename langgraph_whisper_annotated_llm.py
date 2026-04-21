import os
from typing import TypedDict, List, Annotated
from operator import add
from tkinter import Tk, filedialog

# Nuevos namespaces de LangChain/LangGraph 2026
from langgraph.graph import StateGraph, START, END
from utils.local_gemma import LocalGemma4
from langchain_core.messages import BaseMessage

# Importamos tu nuevo módulo local
from utils.local_transcriber import transcribe_local_media

# Configuración del LLM
llm = LocalGemma4(model_path="./models", temperature=0.3)

# Definición del Estado (TypedDict se mantiene igual)
class State(TypedDict):
    notes: str
    participants: List[str]
    topics: List[str]
    action_items: List[str]
    minutes: str
    summary: str
    logs: Annotated[list[str], add]

# ============= NODOS DEL WORKFLOW (Moderna sintaxis) =============

def extract_participants(state: State):
    prompt = f"De las notas: {state['notes']}\nExtrae nombres de participantes separados por comas."
    response = llm.invoke(prompt)
    participants = [p.strip() for p in response.content.split(',') if p.strip()]
    return {"participants": participants, "logs": ["Participantes procesados"]}

def identify_topics(state: State):
    prompt = f"Identifica 3-5 temas de: {state['notes']}\nSepara por punto y coma (;)."
    response = llm.invoke(prompt)
    topics = [t.strip() for t in response.content.split(';') if t.strip()]
    return {"topics": topics, "logs": ["Temas identificados"]}

def extract_actions(state: State):
    prompt = f"Extrae acciones (Acción | Responsable) de: {state['notes']}"
    response = llm.invoke(prompt)
    action_items = [a.strip() for a in response.content.split('|') if a.strip()]
    return {"action_items": action_items, "logs": ["Acciones extraídas"]}

def generate_minutes(state: State):
    prompt = f"Genera minuta formal para: {state['notes']}\nParticipantes: {state['participants']}"
    response = llm.invoke(prompt)
    return {"minutes": response.content, "logs": ["Minuta generada"]}

def create_summary(state: State):
    prompt = f"Resumen ejecutivo de 2 líneas para: {state['minutes']}"
    response = llm.invoke(prompt)
    return {"summary": response.content, "logs": ["Resumen finalizado"]}

# ============= CONSTRUCCIÓN DEL GRAFO =============

def create_workflow():
    workflow = StateGraph(State)
    
    workflow.add_node("extract_participants", extract_participants)
    workflow.add_node("identify_topics", identify_topics)
    workflow.add_node("extract_actions", extract_actions)
    workflow.add_node("generate_minutes", generate_minutes)
    workflow.add_node("create_summary", create_summary)
    
    workflow.add_edge(START, "extract_participants")
    workflow.add_edge("extract_participants", "identify_topics")
    workflow.add_edge("identify_topics", "extract_actions")
    workflow.add_edge("extract_actions", "generate_minutes")
    workflow.add_edge("generate_minutes", "create_summary")
    workflow.add_edge("create_summary", END)
    
    return workflow.compile()

# ============= EJECUCIÓN =============

if __name__ == "__main__":
    app = create_workflow()

    # Selector de archivos
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True) # Asegura que la ventana aparezca al frente
    file_path = filedialog.askopenfilename(title="Selecciona archivo")

    if not file_path:
        print("Salida: No se seleccionó nada.")
    else:
        ext = os.path.splitext(file_path)[1].lower()
        media_exts = {".mp4", ".mov", ".m4a", ".mp3", ".wav", ".mkv", ".webm"}

        # CAMBIO CLAVE: Ahora usa el modelo local
        if ext in media_exts:
            notes = transcribe_local_media(file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                notes = f.read()

        # Ejecución del Grafo
        final_result = app.invoke({"notes": notes, "logs": []})
        print("\n--- participants DE LA REUNIÓN ---")
        print(final_result["participants"])
        print("\n--- topics DE LA REUNIÓN ---")
        print(final_result["topics"])
        print("\n--- action_items DE LA REUNIÓN ---")
        print(final_result["action_items"])
        print("\n--- minutes DE LA REUNIÓN ---")
        print(final_result["minutes"])
        print("\n--- RESUMEN DE LA REUNIÓN ---")
        print(final_result["summary"])
        print("\nLogs del proceso:", final_result["logs"])