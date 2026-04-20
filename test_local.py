from utils.local_gemma import LocalGemma4

# Initialize with your local model path
llm = LocalGemma4(model_path="./models", temperature=0.7)

pregunta = "¿En qué año llegó el ser humano a la Luna por primera vez?"
print("Pregunta:", pregunta)

respuesta = llm.invoke(pregunta)
print("Respuesta del modelo:", respuesta.content)
