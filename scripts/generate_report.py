import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests

# Cargar API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def cargar_base_vectorial(path="data/faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def recuperar_contexto(vectorstore, k=10):
    docs = vectorstore.similarity_search("", k=k)
    return "\n".join([doc.page_content for doc in docs])

def cargar_tematica(path="outputs/tematica_detectada.txt"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip().lower()
    return "tecnología"  # fallback por defecto

def generar_informe(contexto, tematica):
    prompt = f"""
Sos un especialista en {tematica}. Acabás de escuchar una conversación técnica grabada en formato audio.

Tu tarea es redactar un INFORME PROFESIONAL que cumpla los siguientes objetivos:

1. Resumir los temas tratados en forma técnica y clara.
2. Identificar puntos clave, propuestas o problemas discutidos.
3. Redactar recomendaciones técnicas si se mencionaron.
4. Mantener un tono formal y profesional (no conversacional).
5. No inventes datos que no aparezcan en el texto.

⚠️ No menciones que es un audio ni uses frases como "A continuación..." o "Se transcribió lo siguiente".

Texto fuente:
\"\"\"{contexto}\"\"\"
"""

    endpoint = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }

    response = requests.post(endpoint, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# === FLUJO PRINCIPAL ===
if __name__ == "__main__":
    print("📦 Cargando base FAISS...")
    vectorstore = cargar_base_vectorial()

    print("🔍 Recuperando contexto técnico...")
    contexto = recuperar_contexto(vectorstore)

    print("📁 Cargando temática detectada...")
    tematica = cargar_tematica()
    print(f"➡️ Actuando como especialista en: {tematica}")

    print("🧠 Generando informe técnico...\n")
    informe = generar_informe(contexto, tematica)
    print(informe)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/informe_tecnico.txt", "w", encoding="utf-8") as f:
        f.write(informe)

    print("\n✅ Informe generado y guardado en outputs/informe_tecnico.txt")
