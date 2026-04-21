"""
Módulo para descargar embeddings de HuggingFace.
"""

from huggingface_hub import snapshot_download
from typing import Optional
import os


class EmbeddingDownloader:
    """
    Clase para descargar embeddings de HuggingFace de forma dinámica.
    
    Uso:
        1. Llamar a download_embedding() con el ID del embedding
    """
    
    def __init__(self):
        """Inicializa el descargador de embeddings."""
        print("EmbeddingDownloader inicializado")
    
    def download_embedding(self, model_id: str, local_dir: str = "./embeddings") -> str:
        """
        Descarga un embedding de HuggingFace.
        
        Args:
            model_id: ID del embedding en HuggingFace (ej: "sentence-transformers/all-MiniLM-L6-v2")
            local_dir: Directorio local donde se guardará el embedding
            
        Returns:
            Ruta al directorio donde se descargó el embedding
        """
        print(f"Iniciando descarga del embedding: {model_id}")
        
        # Crear directorio local si no existe
        os.makedirs(local_dir, exist_ok=True)
        
        try:
            # Descargar el embedding completo
            local_path = snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                max_workers=4  # Ajustar según ancho de banda
            )
            
            print(f"Embedding descargado exitosamente en: {local_path}")
            return local_path
            
        except Exception as e:
            print(f"Error al descargar el embedding: {str(e)}")
            raise


# Uso de ejemplo
if __name__ == "__main__":
    downloader = EmbeddingDownloader()
    
    # Descargar all-MiniLM-L6-v2
    embedding_path = downloader.download_embedding(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        local_dir="./embeddings/all-MiniLM-L6-v2"
    )
    
    print(f"Embedding listo para usar en: {embedding_path}")
