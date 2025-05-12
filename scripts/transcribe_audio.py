import whisper
import os
import sys

def transcribe_audio(audio_path, model_size="base"):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"No se encontró el archivo: {audio_path}")

    print(f"🔁 Cargando modelo Whisper ({model_size})...")
    model = whisper.load_model(model_size)

    print(f"🎙 Transcribiendo archivo: {audio_path}")
    result = model.transcribe(audio_path)
    text = result["text"]

    # Crear carpeta transcripts si no existe
    os.makedirs("transcripts", exist_ok=True)

    filename = os.path.basename(audio_path).rsplit(".", 1)[0] + ".txt"
    output_path = os.path.join("transcripts", filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"✅ Transcripción guardada en: {output_path}")
    return text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️  Uso: python scripts/transcribe_audio.py <ruta_audio>")
        sys.exit(1)
    
    transcribe_audio(sys.argv[1])
