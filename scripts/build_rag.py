import os
import re
import glob
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

def detectar_ultimo_txt(carpeta="transcripts"):
    archivos_txt = glob.glob(f"{carpeta}/*.txt")
    if not archivos_txt:
        raise FileNotFoundError("❌ No se encontraron archivos .txt en la carpeta transcripts/")
    archivo_mas_reciente = max(archivos_txt, key=os.path.getmtime)
    print(f"📄 Archivo detectado automáticamente: {archivo_mas_reciente}")
    return archivo_mas_reciente

def cargar_frases_utiles(ruta_archivo):
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        texto = f.read()

    print(f"🔍 Texto original tiene {len(texto)} caracteres")

    # Limpieza ligera (sin lematización)
    texto = texto.lower()
    texto = re.sub(r"\s+", " ", texto)
    texto = re.sub(r"(?<!\w)([a-zA-Z])\1{2,}(?!\w)", "", texto)
    texto = texto.replace("..", ".").replace("...", ".").strip()

    # Dividir en frases útiles
    frases_crudas = re.split(r"[.?!]\s+|\n+", texto)
    frases_crudas = [f.strip() for f in frases_crudas if len(f.strip()) > 15 and len(f.strip().split()) > 2]

    print(f"🔍 Se detectaron {len(frases_crudas)} frases útiles")
    return frases_crudas

def construir_chunks(frases, max_frases_por_chunk=2):
    chunks = []
    for i in range(0, len(frases), max_frases_por_chunk):
        bloque = " ".join(frases[i:i+max_frases_por_chunk])
        if len(bloque) > 60:
            chunks.append(Document(page_content=bloque))

    print(f"✅ {len(chunks)} chunks generados.")
    return chunks

def vectorizar_y_guardar(documentos, ruta_salida="data/faiss_index"):
    os.makedirs("data", exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documentos, embeddings)
    vectorstore.save_local(ruta_salida)
    print(f"✅ Base FAISS guardada en: {ruta_salida}")

if __name__ == "__main__":
    ruta_txt = detectar_ultimo_txt()

    frases_utiles = cargar_frases_utiles(ruta_txt)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/contexto_utilizado.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(frases_utiles))

    print("📚 Construyendo chunks...")
    documentos = construir_chunks(frases_utiles)

    print(f"🔢 Vectorizando {len(documentos)} chunks y guardando FAISS...")
    vectorizar_y_guardar(documentos)
