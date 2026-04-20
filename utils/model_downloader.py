"""
Módulo para descargar modelos de HuggingFace.
Requiere token de autorización para modelos privados o restringidos.
"""

from huggingface_hub import snapshot_download
from typing import Optional
import os


class ModelDownloader:
    """
    Clase para descargar modelos de HuggingFace de forma dinámica.
    
    Uso:
        1. Obtener token de: https://huggingface.co/settings/tokens
        2. Configurar en el constructor o como variable de entorno HF_TOKEN
        3. Llamar a download_model() con el ID del modelo
    """
    
    def __init__(self, token: Optional[str] = None):
        """
        Inicializa el descargador de modelos.
        
        Args:
            token: Token de HuggingFace (opcional). Si no se proporciona,
                   se buscará en la variable de entorno HF_TOKEN.
                   Si no hay token, solo se podrán descargar modelos públicos.
        """
        self.token = token or os.getenv("HF_TOKEN")
        if self.token:
            print("Token de HuggingFace configurado")
        else:
            print("Advertencia: No se ha configurado token. Solo se podrán descargar modelos públicos.")
    
    def download_model(self, model_id: str, local_dir: str = "./models") -> str:
        """
        Descarga un modelo de HuggingFace.
        
        Args:
            model_id: ID del modelo en HuggingFace (ej: "meta-llama/Llama-3.2-3B-Instruct")
            local_dir: Directorio local donde se guardará el modelo
            
        Returns:
            Ruta al directorio donde se descargó el modelo
        """
        print(f"Iniciando descarga del modelo: {model_id}")
        
        # Crear directorio local si no existe
        os.makedirs(local_dir, exist_ok=True)
        
        try:
            # Descargar el modelo completo
            local_path = snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                token=self.token,
                max_workers=4  # Ajustar según ancho de banda
            )
            
            print(f"Modelo descargado exitosamente en: {local_path}")
            return local_path
            
        except Exception as e:
            print(f"Error al descargar el modelo: {str(e)}")
            raise


# Uso de ejemplo
if __name__ == "__main__":
    # Configurar token aquí (o como variable de entorno HF_TOKEN)
    token = ""  # <-- COLOCAR TU TOKEN DE HUGGINGFACE AQUÍ
    downloader = ModelDownloader(token=token)
    
    # Descargar Llama-3.2-3B-Instruct
    model_path = downloader.download_model(
        model_id="google/gemma-4-E2B-it",
        local_dir="./models"
    )
    
    print(f"Modelo listo para usar en: {model_path}")
