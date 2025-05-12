import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests

# === CONFIG ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def cargar(path, default=""):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return default

def cargar_base_vectorial(path="data/faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def recuperar_contexto(vectorstore, k=10):
    docs = vectorstore.similarity_search("", k=k)
    return "\n".join([doc.page_content for doc in docs])

def call_llm(prompt, temperature=0.2, max_tokens=1500):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def generar_informe(contexto, tematica):
    return call_llm(f"""
Sos un especialista en {tematica}. Recibiste una conversación técnica grabada.

Tu tarea es redactar un INFORME TÉCNICO profesional que cumpla:

1. Resumir los puntos clave.
2. Identificar problemas, ideas o decisiones.
3. Incluir recomendaciones si fueron mencionadas.

⚠️ Escribí de forma técnica y profesional. No digas que fue un audio ni repitas frases como "a continuación".

Texto fuente:
\"\"\"{contexto}\"\"\"
""")

def revisar_informe(informe, tematica):
    return call_llm(f"""
Sos un revisor técnico especializado en {tematica}.

Revisá el siguiente informe y devolvé observaciones claras:

1. ¿Es correcto técnicamente?
2. ¿Faltan cosas que deberían estar?
3. ¿Tiene afirmaciones inexactas o poco claras?
4. ¿Está bien redactado como informe profesional?

Solo devolvé una revisión crítica. No lo reescribas.

Informe:
\"\"\"{informe}\"\"\"
""", max_tokens=1024)

def generar_informe_final(informe, revision, tematica):
    return call_llm(f"""
Sos un redactor técnico especializado en {tematica}.

Redactá un nuevo INFORME FINAL, aplicando todas las mejoras señaladas en la revisión. No menciones que hubo revisión. No repitas errores anteriores. Reescribilo como si siempre hubiera estado perfecto.

--- Informe original ---
{informe}

--- Revisión técnica ---
{revision}
""", temperature=0.3)

# === FLUJO PRINCIPAL ===
if __name__ == "__main__":
    print("📦 Cargando base FAISS...")
    vectorstore = cargar_base_vectorial()

    print("🔍 Recuperando contexto...")
    contexto = recuperar_contexto(vectorstore)

    print("📁 Cargando temática detectada...")
    tematica = cargar("outputs/tematica_detectada.txt", default="tecnología")
    print(f"➡️ Temática: {tematica}")

    # Paso 1: Informe técnico inicial
    print("\n🧠 Generando informe técnico inicial...")
    informe = generar_informe(contexto, tematica)
    print(informe)
    with open("outputs/informe_tecnico.txt", "w", encoding="utf-8") as f:
        f.write(informe)

    # Paso 2: Revisión técnica
    print("\n🔎 Revisando informe...")
    revision = revisar_informe(informe, tematica)
    print(revision)
    with open("outputs/revision_informe.txt", "w", encoding="utf-8") as f:
        f.write(revision)

    # Paso 3: Informe final validado
    print("\n📋 Generando informe final validado...")
    informe_final = generar_informe_final(informe, revision, tematica)
    print(informe_final)
    with open("outputs/informe_final.txt", "w", encoding="utf-8") as f:
        f.write(informe_final)

    print("\n✅ Informe final generado y guardado en outputs/informe_final.txt")
