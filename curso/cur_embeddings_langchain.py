from langchain_openai import OpenAIEmbeddings
import numpy as np

########### The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

texto1 = "La capital de Francia es París."
texto2 = "París es un nombre común para mascotas."

vec1 = embeddings.embed_query(texto1)
vec2 = embeddings.embed_query(texto2)

print(f"Dimensión de los vectores: {len(vec1)}")

cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

print(f"Similitud coseno entre vec1 y vec2: {cos_sim:.3f}")