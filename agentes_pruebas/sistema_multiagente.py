import sys
import os
from pathlib import Path
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

# Configuración de rutas
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from utils.local_gemma import LocalGemma4
from config import LLM_MODEL_PATH

# 1. Instancia del modelo único
# Usamos temperature=0 para agentes para evitar que "alucinen" con los formatos JSON
model = LocalGemma4(model_path=LLM_MODEL_PATH, temperature=0)

# 2. Definición de herramientas
@tool
def buscar_web(query: str) -> str:
    """Busca información técnica o general en la web sobre un tema específico."""
    return f"Resultados de búsqueda para: {query}. (Simulación: Gemma es un modelo de Google)."

@tool
def calcular(expresion: str) -> str:
    """Resuelve operaciones matemáticas complejas. Ejemplo de entrada: '2 + 2 * 5'."""
    try:
        # Nota: En producción, usa un parser seguro en lugar de eval()
        return f"Resultado: {eval(expresion)}"
    except Exception as e:
        return f"Error en el cálculo: {str(e)}"

# 3. Nodos de Agentes (Especialistas)
agente_investigacion = create_react_agent(
    model=model,
    tools=[buscar_web],
    prompt="Eres un experto investigador. Si te piden un dato, búscalo.",
    name="investigador"
)

agente_matematicas = create_react_agent(
    model=model,
    tools=[calcular],
    prompt="Eres un calculador preciso. Solo resuelves matemáticas.",
    name="matematico"
)

# 4. Orquestador (Supervisor)
# El supervisor usa el mismo modelo para decidir a quién llamar
workflow = create_supervisor(
    [agente_matematicas, agente_investigacion],
    model=model,
    prompt=(
        "Eres el supervisor oficial. Tu ÚNICA función es delegar tareas a los agentes especializados.\n"
        "Si la pregunta es de matemáticas, DEBES delegar al agente 'matematico' usando la herramienta 'transfer_to_matematico'.\n"
        "Si la pregunta es de información/hechos, DEBES delegar al agente 'investigador' usando la herramienta 'transfer_to_investigador'.\n"
        "Si la pregunta contiene ambas cosas, delega a ambos agentes en orden.\n"
        "IMPORTANTE: No respondas con texto normal. Responde ÚNICAMENTE con una llamada a herramienta en formato JSON.\n"
        "Formato correcto: {\"name\": \"transfer_to_matematico\", \"arguments\": {\"input\": \"la pregunta específica\"}}\n"
        "O para múltiples delegaciones: [{\"name\": \"transfer_to_matematico\", \"arguments\": {...}}, {\"name\": \"transfer_to_investigador\", \"arguments\": {...}}]\n"
        "Pregunta del usuario: {messages}"
    ),
    max_iterations=5
)

supervisor = workflow.compile()

# Uso del sistema multi-agente con una tarea real
response = supervisor.invoke({
    "messages": [{
        "role": "user", 
        "content": "¿Cuánto es 1543 multiplicado por 2.5 y quién ganó el mundial de 1986?"
    }]
})

print("\n--- FLUJO DE TRABAJO ---")
for msg in response['messages']:
    # Esto te permitirá ver cómo el supervisor delega
    role = "🤖 Agente" if hasattr(msg, 'tool_calls') else "👤 Usuario/Respuesta"
    print(f"[{role}]: {msg.content}")