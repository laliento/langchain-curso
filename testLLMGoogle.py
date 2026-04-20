from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

print("hola mundo")
#llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7,api_key="")
pregunta = "¿En qué año llegó el ser humano a la Luna por primera vez?"
print("Pregunta: ", pregunta)

respueta = llm.invoke(pregunta)
print("Respuesta del modelo: ", respueta.content)