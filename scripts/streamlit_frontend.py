import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import requests

# === CONFIG ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === UTILIDADES ===
def guardar_archivo_subido(uploaded_file, save_path):
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

def construir_chunks_y_guardar(path_txt, path_faiss="data/faiss_index"):
    with open(path_txt, encoding="utf-8") as f:
        texto = f.read()

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documentos = splitter.create_documents([texto])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documentos, embeddings)
    os.makedirs("data", exist_ok=True)
    vectorstore.save_local(path_faiss)
    return vectorstore, documentos

def detectar_tematica(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [doc.page_content for doc in chunks]
    embeddings = model.encode(texts)
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    dominant_cluster = max(set(labels), key=list(labels).count)
    centroid = kmeans.cluster_centers_[dominant_cluster]
    similarities = cosine_similarity([centroid], embeddings)[0]
    top_indices = similarities.argsort()[-3:][::-1]
    chunk_representativo = "\n\n".join([texts[i] for i in top_indices])[:1500]

    prompt = f"""Tu tarea es analizar este texto y determinar su temática general principal.

Texto:
\"\"\"{chunk_representativo}\"\"\"

Respondé SOLO con una palabra entre las siguientes:
- educación
- medicina
- psicología
- derecho
- ingeniería
- tecnología
- negocios
- otra
"""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 10
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

def call_llm(prompt, temperature=0.2, max_tokens=1500):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "llama3-70b-8192", "messages": [{"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": max_tokens}
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def generar_informe_final_directo(contexto, tematica):
    prompt = f"""Sos un redactor técnico especializado en {tematica}.

Te presento una transcripción de una clase o reunión. Tu tarea es generar directamente un INFORME FINAL profesional.

Debés:
- Identificar los temas tratados
- Resumir las decisiones, propuestas o problemas discutidos
- Redactar recomendaciones técnicas si fueron mencionadas

No digas que esto fue una transcripción ni menciones que proviene de audio.

Texto fuente:
\"\"\"{contexto}\"\"\"
"""
    return call_llm(prompt, temperature=0.3, max_tokens=1500)

def responder_query(query, tematica, vectorstore, k=5):
    documentos = vectorstore.similarity_search(query, k=k)
    contexto = "\n---\n".join([doc.page_content for doc in documentos])
    prompt = f"""Sos un especialista en {tematica}.

A continuación tenés fragmentos de una clase o reunión transcripta, junto con una pregunta de un participante.

Tu tarea es:
- Responder solo en base a los fragmentos
- Ser claro, profesional y preciso
- Indicar si no se mencionó el tema

❓ Pregunta:
{query}

📚 Fragmentos:
\"\"\"{contexto}\"\"\"
"""
    return call_llm(prompt, temperature=0.2, max_tokens=512)

# === INTERFAZ STREAMLIT ===
st.set_page_config(page_title="Informe + Chatbot", layout="wide")
st.title("🧠 Informe Técnico + Chatbot RAG")

# === CARGA DE TRANSCRIPCIÓN ===
with st.sidebar:
    st.header("📤 Subir transcripción TXT")
    uploaded = st.file_uploader("Elegí un archivo .txt con la transcripción", type="txt")

    if uploaded:
        st.success("Archivo recibido. Procesando...")
        os.makedirs("transcripts", exist_ok=True)
        path_txt = os.path.join("transcripts", uploaded.name)
        guardar_archivo_subido(uploaded, path_txt)

        with st.spinner("Construyendo chunks y embeddings..."):
            vectorstore, chunks = construir_chunks_y_guardar(path_txt)
            tematica = detectar_tematica(chunks)
            os.makedirs("outputs", exist_ok=True)
            with open("outputs/tematica_detectada.txt", "w", encoding="utf-8") as f:
                f.write(tematica)
            st.session_state["vectorstore"] = vectorstore
            st.session_state["tematica"] = tematica
        st.success(f"Proceso completado. Temática detectada: {tematica}")

# === CARGA AUTOMÁTICA DE FAISS + TEMÁTICA ===
if "vectorstore" not in st.session_state:
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state["vectorstore"] = FAISS.load_local("data/faiss_index", embeddings, allow_dangerous_deserialization=True)
        with open("outputs/tematica_detectada.txt", "r", encoding="utf-8") as f:
            st.session_state["tematica"] = f.read().strip()
    except:
        st.warning("Subí primero un archivo de transcripción para comenzar.")
        st.stop()

vectorstore = st.session_state["vectorstore"]
tematica = st.session_state["tematica"]

# === INTERFAZ PRINCIPAL ===
tab1, tab2 = st.tabs(["📄 Informe Final", "💬 Chatbot de Consulta"])

with tab1:
    st.subheader("Generar informe técnico final")
    if st.button("📝 Generar Informe"):
        with st.spinner("Generando informe final como especialista..."):
            chunks = vectorstore.similarity_search("", k=10)
            contexto = "\n".join([doc.page_content for doc in chunks])
            resultado = generar_informe_final_directo(contexto, tematica)
        st.success("Informe generado correctamente")
        st.text_area("📋 Informe Final", resultado, height=400)

with tab2:
    st.subheader("Consultar sobre la reunión")
    st.markdown(f"**Temática detectada:** `{tematica}`")
    pregunta = st.text_input("❓ Escribí tu pregunta sobre la reunión")
    if st.button("🤖 Consultar") and pregunta:
        with st.spinner("Buscando respuesta..."):
            respuesta = responder_query(pregunta, tematica, vectorstore)
        st.markdown("### 🧠 Respuesta")
        st.write(respuesta)
