import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests

# Cargar clave API
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def cargar_chunks_desde_faiss(path="data/faiss_index"):
    embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(path, embeddings=embeddings_model, allow_dangerous_deserialization=True)
    documentos = [doc.page_content for doc in vectorstore.docstore._dict.values()]
    return documentos, embeddings_model

def detectar_texto_representativo(chunks, embeddings_model, num_clusters=6, top_n=15, min_len=100):
    embeddings = embeddings_model.encode(chunks)
    kmeans = KMeans(n_clusters=min(num_clusters, len(chunks)), random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Seleccionar el cluster más frecuente (dominante)
    dominant_cluster = np.argmax(np.bincount(labels))
    centroid = kmeans.cluster_centers_[dominant_cluster]

    # Ordenar los chunks más similares al centroide del cluster dominante
    similarities = cosine_similarity([centroid], embeddings)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]

    # Filtrar por longitud mínima y diversidad léxica
    top_chunks = []
    for i in top_indices:
        chunk = chunks[i]
        if len(chunk) >= min_len and len(set(chunk.split())) >= 8:
            top_chunks.append(chunk)

    # Eliminar duplicados o repetidos
    chunk_unicos = []
    for chunk in top_chunks:
        if all(chunk not in c for c in chunk_unicos):
            chunk_unicos.append(chunk)

    texto_final = "\n\n".join(chunk_unicos)
    return texto_final[:10000]  # Corte defensivo por límite de Groq

def etiquetar_con_llm(texto, model="llama3-70b-8192"):
    prompt = f"""
Estás analizando el contenido de una clase, reunión o conversación técnica.

Tu tarea es identificar la **temática general predominante** del siguiente texto.

Texto:
\"\"\"{texto}\"\"\"

Respondé SOLO con una palabra exacta de esta lista, y NADA MÁS:

- educación
- medicina
- psicología
- derecho
- ingeniería
- tecnología
- negocios
- otra

Ejemplo válido: tecnología
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 5
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip().lower()

if __name__ == "__main__":
    print("📦 Cargando chunks desde FAISS...")
    chunks, model = cargar_chunks_desde_faiss()

    print("🧠 Detectando texto representativo (top 15 chunks únicos)...")
    texto_dominante = detectar_texto_representativo(chunks, model)

    print("\n📌 Texto representativo:")
    print(texto_dominante[:1000] + "...\n")  # Vista extendida

    print("🔖 Etiquetando temática...")
    etiqueta = etiquetar_con_llm(texto_dominante)

    print(f"\n✅ Temática detectada: {etiqueta}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/tematica_detectada.txt", "w", encoding="utf-8") as f:
        f.write(etiqueta)
