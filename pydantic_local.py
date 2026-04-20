from pydantic import BaseModel, Field
from utils.local_gemma import LocalGemma4


class AnalisisTexto(BaseModel):
    """Análisis de texto."""
    resumen: str = Field(description="Resumen breve del texto.")
    sentimiento: str = Field(description="Sentimiento del texto (Positivo, neutro o negativo)")


llm = LocalGemma4(model_path="./models", temperature=0.0)

texto_prueba = "Me encantó la nueva película de acción, tiene muchos efectos especiales y emoción."

# Crear un prompt que obligue al modelo a generar JSON
prompt = f"""Analiza el siguiente texto y genera una respuesta en formato JSON con las siguientes claves:
- resumen: Un resumen breve del texto
- sentimiento: El sentimiento del texto (Positivo, neutro o negativo)

Texto: {texto_prueba}

IMPORTANTE: Genera SOLO el JSON, sin texto adicional, sin markdown, sin explicaciones previas ni posteriores. El JSON debe tener exactamente las dos claves especificadas."""

# Probar invoke directo para ver qué genera el modelo
print("=== Respuesta directa del modelo ===")
response = llm.invoke(prompt)
print(response.content)
print("=== Fin de respuesta directa ===")

# Intentar extraer JSON manualmente
import re
import json

json_match = re.search(r'\{[\s\S]*\}', response.content)
if json_match:
    try:
        data = json.loads(json_match.group())
        resultado = AnalisisTexto(**data)
        print("=== Resultado estructurado ===")
        print(resultado.model_dump_json())
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
else:
    print("No se encontró JSON en la respuesta")