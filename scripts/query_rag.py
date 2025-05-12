import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests

# === CONFIG ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def cargar_vectorstore(path="data/faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def cargar_tematica(path="outputs/tematica_detectada.txt"):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "el tema tratado en la reunión"

def consultar_transcripcion(query, tematica, k=5):
    vectorstore = cargar_vectorstore()
    documentos = vectorstore.similarity_search(query, k=k)
    contexto = "\n---\n".join([doc.page_content for doc in documentos])

    prompt = f"""
Sos un especialista en {tematica}.

A continuación tenés fragmentos de una clase o reunión transcripta, junto con una pregunta realizada por un alumno o participante.

Tu tarea es:
- Responder con precisión técnica y claridad.
- Usar únicamente la información de los fragmentos.
- Indicar si algo no fue mencionado, en lugar de inventar.

❓ Pregunta:
{query}

📚 Fragmentos de contexto:
\"\"\"{contexto}\"\"\"

🧠 Respondé de forma clara, profesional y concisa.
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 512
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# === FLUJO PRINCIPAL ===
if __name__ == "__main__":
    print("🧠 Sistema de consulta sobre la clase o reunión transcripta")
    tematica = cargar_tematica()
    print(f"📌 Temática detectada: {tematica}")

    while True:
        pregunta = input("\n🔎 Ingresá tu pregunta (o escribí 'salir' para terminar):\n> ")
        if pregunta.lower() in ["salir", "exit", "quit"]:
            break
        respuesta = consultar_transcripcion(pregunta, tematica)
        print("\n🧠 RESPUESTA:\n")
        print(respuesta)
