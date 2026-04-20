from utils.local_gemma import LocalGemma4
from models.cv_model import AnalisisCV
from prompts.cv_prompts import crear_sistema_prompts
import re
import json

def crear_evaluador_cv():
    modelo_base = LocalGemma4(
        model_path="C:\\Users\\laptop\\git\\langchain-curso\\models",
        temperature=0.0
    )
    
    chat_prompt = crear_sistema_prompts()
    
    # Crear una cadena que combine el prompt con el modelo
    def evaluar(texto_cv: str, descripcion_puesto: str) -> dict:
        """Evaluar candidato y devolver resultados en formato JSON."""
        print("=== INICIANDO EVALUACIÓN ===")
        print(f"Texto CV (primeros 500 chars): {texto_cv[:500]}...")
        print(f"Descripción puesto (primeros 500 chars): {descripcion_puesto[:500]}...")
        
        prompt = chat_prompt.invoke({
            "texto_cv": texto_cv,
            "descripcion_puesto": descripcion_puesto
        })
        
        print("=== PROMPT GENERADO ===")
        # Obtener todos los mensajes del prompt
        if hasattr(prompt, 'to_messages'):
            messages = prompt.to_messages()
            print(f"Número de mensajes: {len(messages)}")
            for i, msg in enumerate(messages):
                print(f"Mensaje {i} ({msg.type}): {msg.content[:500]}...")
            prompt_text = "\n\n".join([msg.content for msg in messages])
        else:
            prompt_text = str(prompt)
        
        print(f"Prompt text completo (primeros 1500 chars): {prompt_text[:1500]}...")
        
        # Agregar instrucción explícita para generar JSON
        prompt_final = f"""{prompt_text}

Por favor, responde SOLO con un objeto JSON con el siguiente formato exacto:
{{
  "nombre_candidato": "Nombre completo del candidato",
  "experiencia_años": 5,
  "habilidades_clave": ["Python", "JavaScript", "React", "SQL", "Git"],
  "education": "Ingeniería en Sistemas - Universidad X",
  "experiencia_relevante": "Desarrollador Senior en empresa Y con 3 años de experiencia...",
  "fortalezas": ["Experiencia en desarrollo full-stack", "Habilidades de liderazgo", "Resolución de problemas"],
  "areas_mejora": ["Gestión de proyectos", "Inglés avanzado"],
  "porcentaje_ajuste": 85
}}

IMPORTANTE: 
- Genera SOLO el JSON, sin texto adicional, sin markdown, sin explicaciones previas ni posteriores
- Usa los campos exactos especificados
- Si no encuentras alguna información, usa una cadena vacía "" o 0 para números
- No inventes información"""
        
        print("=== INVOCANDO MODELO ===")
        response = modelo_base.invoke(prompt_final)
        print("=== RESPUESTA DEL MODELO ===")
        print("Response raw: " + response.content)
        
        # Extraer JSON de la respuesta
        json_match = re.search(r'\{[\s\S]*\}', response.content)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                pass
        
        print("=== NO SE ENCONTRÓ JSON VÁLIDO ===")
        return {}
    
    return evaluar

def evaluar_candidato(texto_cv: str, descripcion_puesto: str) -> AnalisisCV:
    try:
        print ("evaluar_candidato texto_cv: ", texto_cv )
        print ("evaluar_candidato textodescripcion_puesto_cv: ", descripcion_puesto )
        evaluar = crear_evaluador_cv()
        resultado_dict = evaluar(texto_cv, descripcion_puesto)
        
        print ("evaluar_candidato resultado_dict: ", resultado_dict )
        if resultado_dict:
            return AnalisisCV(**resultado_dict)
        else:
            return AnalisisCV(
                nombre_candidato="Error en procesamiento.",
                experiencia_años=0,
                habilidades_clave=["Error al procesar CV"],
                education="No se puede determinar.",
                experiencia_relevante="Error durante el análisis.",
                fortalezas=["Requiere revisión manual del CV"],
                areas_mejora=["Verificar formato y legibilidad del PDF"],
                porcentaje_ajuste=0
            )
    
    except Exception as e:
        print(f"Ocurrió un error: {e}")
        return AnalisisCV(
            nombre_candidato="Error en procesamiento.",
            experiencia_años=0,
            habilidades_clave=["Error al procesar CV"],
            education="No se puede determinar.",
            experiencia_relevante="Error durante el análisis.",
            fortalezas=["Requiere revisión manual del CV"],
            areas_mejora=["Verificar formato y legibilidad del PDF"],
            porcentaje_ajuste=0
        )
