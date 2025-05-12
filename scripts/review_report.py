import os
from dotenv import load_dotenv
import requests

# Cargar API Key desde .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def cargar_texto(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""

def revisar_informe_como_especialista(informe, tematica):
    endpoint = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
Sos un revisor técnico especializado en {tematica}.

Te entregaron el siguiente informe técnico, que resume una reunión o clase. Tu tarea es revisar su calidad profesional, evaluando:

1. Si el contenido es coherente y técnicamente correcto.
2. Si hay afirmaciones sin sustento o inexactas.
3. Si falta mencionar algo importante que normalmente debería incluirse.
4. Si es claro, directo y útil como documento profesional.

No reescribas el informe. Solo devolvé observaciones y sugerencias concretas.

Informe:
\"\"\"{informe}\"\"\"
"""

    data = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1024
    }

    response = requests.post(endpoint, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# === FLUJO PRINCIPAL ===
if __name__ == "__main__":
    informe = cargar_texto("outputs/informe_tecnico.txt")
    tematica = cargar_texto("outputs/tematica_detectada.txt")

    if not informe:
        print("❌ No se encontró el informe técnico.")
        exit()

    if not tematica:
        tematica = "tecnología"
        print("⚠️ No se detectó temática. Se usará 'tecnología' por defecto.")

    print(f"📌 Temática detectada: {tematica}")
    print("\n📝 Informe original (primeros 500 caracteres):")
    print(informe[:500] + "...\n")

    print("🔎 Revisando el informe como especialista...\n")
    revision = revisar_informe_como_especialista(informe, tematica)

    print("\n🛠️ REVISIÓN DEL INFORME:\n")
    print(revision)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/revision_informe.txt", "w", encoding="utf-8") as f:
        f.write(revision)

    print("\n✅ Revisión guardada en outputs/revision_informe.txt")
