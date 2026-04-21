"""
Módulo para descargar Whisper-Small de HuggingFace.
"""

from huggingface_hub import snapshot_download
from typing import Optional
import os


class WhisperDownloader:
    """
    Clase para descargar Whisper-Small en el directorio local especificado.
    """
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("HF_TOKEN")
        if self.token:
            print("Token de HuggingFace configurado")
        else:
            # Whisper es público, no dará error si el token está vacío
            print("Nota: Descargando como modelo público (Whisper no requiere token).")
    
    def download_whisper(self, model_id: str = "openai/whisper-small", local_dir: str = "./models/whisper-small") -> str:
        """
        Descarga el modelo Whisper.
        
        Args:
            model_id: ID del modelo (por defecto whisper-small)
            local_dir: Directorio específico solicitado
        """
        print(f"Iniciando descarga de Whisper: {model_id}")
        
        # Crear la ruta completa de carpetas
        os.makedirs(local_dir, exist_ok=True)
        
        try:
            # snapshot_download descarga todos los archivos necesarios (.bin, config.json, etc.)
            local_path = snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                token=self.token,
                max_workers=4 
            )
            
            print(f"\n✅ Whisper listo en: {local_path}")
            return local_path
            
        except Exception as e:
            print(f"Error al descargar Whisper: {str(e)}")
            raise


if __name__ == "__main__":
    # Para Whisper no es estrictamente necesario el token
    downloader = WhisperDownloader()
    
    # Ejecutamos la descarga
    # Esto creará la carpeta /models/whisper-small/ dentro de tu proyecto
    path = downloader.download_whisper(
        model_id="openai/whisper-small",
        local_dir="./models/whisper-small"
    )