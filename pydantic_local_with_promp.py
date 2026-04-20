from pydantic import BaseModel, Field
from utils.local_gemma import LocalGemma4


class AnalisisTexto(BaseModel):
    resumen: str = Field(description="Resumen breve del texto.")
    sentimiento: str = Field(description="Sentimiento del texto (Positivo, neutro o negativo)")


llm = LocalGemma4(model_path="./models", temperature=0)


def format_prompt(texto: str) -> str:
    """Format prompt to force JSON output."""
    return f"""Analiza el siguiente texto y genera una respuesta en formato JSON con las siguientes claves:
- resumen: Un resumen breve del texto
- sentimiento: El sentimiento del texto (Positivo, neutro o negativo)

Texto: {texto}

IMPORTANTE: Genera SOLO el JSON, sin texto adicional, sin markdown, sin explicaciones previas ni posteriores. El JSON debe tener exactamente las dos claves especificadas."""


texto_prueba = "Me encantó la nueva película de acción, tiene muchos efectos especiales y emoción."

# Usar invoke directo con el prompt formateado
prompt = format_prompt(texto_prueba)
response = llm.invoke(prompt)

# Extraer JSON del texto (asumiendo que el modelo generó JSON válido)
print(response.content)