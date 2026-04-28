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

from utils.local_gemma_agent import LocalGemma4
from config import LLM_MODEL_PATH
from config_agent import AGENT_INVESTIGATOR, AGENT_MATEMATICO

# 1. Instancia del modelo único
# Usamos temperature=0 para agentes para evitar que "alucinen" con los formatos JSON
model = LocalGemma4(model_path=LLM_MODEL_PATH, temperature=0)

# 2. Definición de herramientas
@tool
def buscar(query: str) -> str:
    """Busca información técnica o general en la web sobre un tema específico."""
    return "Nacio el 28 de febrero de 1989"

@tool
def calcular(expresion: str) -> str:
    """Resuelve operaciones matemáticas complejas. Ejemplo de entrada: '2 + 2 * 5'."""
    try:
        # Nota: En producción, usa un parser seguro en lugar de eval()
        return eval(expresion)
    except Exception as e:
        return f"Error en el cálculo: {str(e)}"

# 3. Nodos de Agentes (Especialistas)
agente_investigacion = create_react_agent(
    model=model,
    tools=[buscar],
    prompt= (
        "Eres un experto investigador a la pregunta proporcionada debes replantearla en modelo cadena, sólo responde la misma pregunta pero replanteada y sumarizada.\n"
        "Sólo responde como plantearías la pregunta, no añadas nada de contexto extra ni citas de la pregunta original ni del formato esperado."),
    name=AGENT_INVESTIGATOR
)

agente_matematicas = create_react_agent(
    model=model,
    tools=[calcular],
    prompt=(
        "Eres un calculador preciso. Del la siguiente pregunta para resolver un problema matemático, sólo responde con la operación matemática para resolverlo NADA extra.\n"
        "Por ejemplo de la pregunta cuanto es 8 por 9 más 2, tu respuesta sólo será la operación matemática como: 9 * 8 + 2"),
    name=AGENT_MATEMATICO
)

# 4. Orquestador (Supervisor)
# El supervisor usa el mismo modelo para decidir a quién llamar
workflow = create_supervisor(
    [agente_matematicas, agente_investigacion],
    model=model,
   prompt = (
    "Eres el Supervisor Orquestador. Tu ÚNICO trabajo es gestionar agentes para obtener una respuesta final. Los agentes disponibles son:\n\n"
    "__TOOLS__\n\n"
    "REGLAS CRÍTICAS:\n"
    "1. Tienes PROHIBIDO responder preguntas tú mismo.\n"
    "2. Si la información no viene de un mensaje [ASSISTANT], entonces NO la conoces.\n"
    "3. Si falta información, DELEGA usando el comando:  DELEGAR_A_AGENTE: ... | INPUT: ...\n"
    "4. Solo cuando TODAS las partes de la pregunta del usuario tengan una respuesta de un [ASSISTANT], redacta la respuesta final combinándolas.\n"
    "5. Puedes usar como base de conocimiento las respuestas de [ASSISTANT] para volver a usar uno de los agentes\n\n"
    "¡NO INVENTES RESPUESTAS! Si no ves un mensaje [ASSISTANT] con el dato."
),
    max_iterations=5
)

supervisor = workflow.compile()

# Uso del sistema multi-agente con una tarea real
response = supervisor.invoke({
    "messages": [{
        "role": "user", 
        #"content": "¿Cuánto es 1543 multiplicado por 2.5 y cuando nació albert einstein?"
        "content": "¿En que año nació albert einstein? el resultado del año multiplicalo por 10"
    }]
})

print("\n--- FLUJO DE TRABAJO ---")
for msg in response['messages']:
    # Esto te permitirá ver cómo el supervisor delega
    role = "🤖 Agente" if hasattr(msg, 'tool_calls') else "👤 Usuario/Respuesta"
    print(f"[{role}]: {msg.content}")