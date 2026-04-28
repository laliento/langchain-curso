from langchain_huggingface import HuggingFaceEmbeddings
from config import *
import numpy as np

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

texto1 = "La capital de Francia es París."
texto2 = "En paris hay muchos perfumes."

vec1 = embeddings.embed_query(texto1)
vec2 = embeddings.embed_query(texto2)

print(f"Dimensión de los vectores: {len(vec1)}")

cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

print(f"Similitud coseno entre vec1 y vec2: {cos_sim:.3f}")