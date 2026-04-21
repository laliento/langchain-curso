#model_path = "C:\\Users\\laptop\\git\\langchain-curso\\models\\whisper-small"
import os
import librosa
import warnings
from transformers import pipeline
import logging
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

# Silenciamos advertencias para una consola limpia
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def transcribe_local_media(file_path: str) -> str:
    # Asegúrate de usar la ruta absoluta o relativa correcta
    model_path = os.path.abspath("C:\\Users\\laptop\\git\\langchain-curso\\models\\whisper-small")
    
    if not os.path.exists(model_path):
        return f"Error: No se encontró el modelo en {model_path}"

    try:
        # 1. Cargar el audio con Librosa
        print(f"🎬 Procesando audio de {os.path.basename(file_path)}...")
        audio_array, sampling_rate = librosa.load(file_path, sr=16000)

        # 2. Configurar el pipeline 
        # Quitamos local_files_only para evitar el conflicto de model_kwargs
        print(f"🎙️ Cargando Whisper Local...")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_path,
            device="cpu"
        )

        print(f"⌛ Transcribiendo (esto puede tardar según la duración)...")
        
        # 3. Ejecución de la transcripción
        result = pipe(
            audio_array, 
            batch_size=8, 
            return_timestamps=True,
            generate_kwargs={
                "language": "es", 
                "task": "transcribe"
            }
        )
        
        print(f"✓ Transcripción completada con éxito.")
        return result["text"]
        
    except Exception as e:
        print(f"❌ Error en transcripción: {e}")
        return f"Error: {str(e)}"