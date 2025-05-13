import os
import sys
import requests
from dotenv import load_dotenv

# Cargar API key de Groq
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def transcribe_audio(audio_path):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"No se encontró el archivo: {audio_path}")

    print(f"🎙 Transcribiendo archivo: {audio_path}")
    
    try:
        # Leer el archivo de audio
        with open(audio_path, "rb") as audio_file:
            files = {
                "file": ("audio.wav", audio_file, "audio/wav")
            }
            
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}"
            }
            
            # Hacer la petición a la API de Groq
            response = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers=headers,
                files=files,
                data={"model": "whisper-large-v3"}
            )
            
            # Verificar si la respuesta fue exitosa
            response.raise_for_status()
            
            # Extraer el texto transcrito
            text = response.json()["text"]
            
            # Crear carpeta transcripts si no existe
            os.makedirs("transcripts", exist_ok=True)
            
            # Guardar la transcripción
            filename = os.path.basename(audio_path).rsplit(".", 1)[0] + ".txt"
            output_path = os.path.join("transcripts", filename)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
                
            print(f"✅ Transcripción guardada en: {output_path}")
            return text
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Error al transcribir el audio: {str(e)}"
        if hasattr(e.response, 'json'):
            try:
                error_detail = e.response.json()
                error_msg += f"\nDetalle: {error_detail}"
            except:
                pass
        raise Exception(error_msg)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️  Uso: python scripts/transcribe_audio.py <ruta_audio>")
        sys.exit(1)
    
    transcribe_audio(sys.argv[1])
