from langchain_core.prompts import ChatPromptTemplate

from config import *
# Obtiene la ruta del directorio superior
import sys
import os
from pathlib import Path
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)
from utils.local_gemma import LocalGemma4



llm = LocalGemma4(model_path=LLM_MODEL_PATH, temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil."),
    ("human", "{input}")
])

chain = prompt | llm

print("Chat en terminal (escribe 'salir' para terminar)\n")

while True:
    try:
        user_input = input("Tú: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nHasta luego!")
        break

    if not user_input:
        continue
    if user_input.lower() in {"salir", "exit", "quit"}:
        print("Hasta luego!")
        break

    respuesta = chain.invoke({"input": user_input})
    print("Asistente:", respuesta.content)