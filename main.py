from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import tempfile
from scripts.transcribe_audio import transcribe_audio
from scripts.build_rag import cargar_frases_utiles, construir_chunks, vectorizar_y_guardar
from scripts.detectar_tematica import detectar_texto_representativo
from scripts.generate_report import generar_informe, cargar_base_vectorial, recuperar_contexto
from scripts.review_report import revisar_informe_como_especialista
from scripts.query_rag import consultar_transcripcion
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Optional

# Cargar variables de entorno
load_dotenv()

app = FastAPI()

# CORS para aceptar requests desde frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    try:
        # Crear directorios necesarios
        os.makedirs("temp", exist_ok=True)
        os.makedirs("transcripts", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        # Guardar el archivo de audio temporalmente
        temp_audio_path = os.path.join("temp", file.filename)
        with open(temp_audio_path, "wb") as buffer:
            audio_bytes = await file.read()
            buffer.write(audio_bytes)
        
        # Transcribir el audio
        texto_transcrito = transcribe_audio(temp_audio_path)
        
        # Procesar el texto y construir RAG
        transcript_path = os.path.join("transcripts", os.path.basename(temp_audio_path).rsplit(".", 1)[0] + ".txt")
        frases_utiles = cargar_frases_utiles(transcript_path)
        documentos = construir_chunks(frases_utiles)
        vectorizar_y_guardar(documentos)
        
        # Detectar temática
        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        texto_representativo = detectar_texto_representativo(frases_utiles, embeddings_model)
        
        # Guardar temática detectada
        with open("outputs/tematica_detectada.txt", "w", encoding="utf-8") as f:
            f.write(texto_representativo)
        
        # Generar informe inicial
        vectorstore = cargar_base_vectorial()
        contexto = recuperar_contexto(vectorstore)
        informe = generar_informe(contexto, texto_representativo)
        
        # Guardar informe
        with open("outputs/informe_tecnico.txt", "w", encoding="utf-8") as f:
            f.write(informe)
        
        # Revisar informe
        revision = revisar_informe_como_especialista(informe, texto_representativo)
        
        # Guardar revisión
        with open("outputs/revision_informe.txt", "w", encoding="utf-8") as f:
            f.write(revision)
        
        # Limpiar archivo temporal
        os.remove(temp_audio_path)
        
        return {
            "message": "Procesamiento completado exitosamente",
            "transcripcion": texto_transcrito,
            "informe": informe,
            "revision": revision,
            "tematica": texto_representativo
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_transcription(query: str = Body(..., embed=True)):
    try:
        # Cargar temática detectada
        tematica = "tecnología"  # valor por defecto
        if os.path.exists("outputs/tematica_detectada.txt"):
            with open("outputs/tematica_detectada.txt", "r", encoding="utf-8") as f:
                tematica = f.read().strip()
        
        respuesta = consultar_transcripcion(query, tematica)
        return {"respuesta": respuesta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report")
async def get_report():
    try:
        informe = ""
        revision = ""
        tematica = "tecnología"
        
        if os.path.exists("outputs/informe_tecnico.txt"):
            with open("outputs/informe_tecnico.txt", "r", encoding="utf-8") as f:
                informe = f.read()
        
        if os.path.exists("outputs/revision_informe.txt"):
            with open("outputs/revision_informe.txt", "r", encoding="utf-8") as f:
                revision = f.read()
                
        if os.path.exists("outputs/tematica_detectada.txt"):
            with open("outputs/tematica_detectada.txt", "r", encoding="utf-8") as f:
                tematica = f.read().strip()
        
        return {
            "informe": informe,
            "revision": revision,
            "tematica": tematica
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/regenerate-report")
async def regenerate_report():
    try:
        vectorstore = cargar_base_vectorial()
        contexto = recuperar_contexto(vectorstore)
        
        # Cargar temática
        tematica = "tecnología"
        if os.path.exists("outputs/tematica_detectada.txt"):
            with open("outputs/tematica_detectada.txt", "r", encoding="utf-8") as f:
                tematica = f.read().strip()
        
        # Generar nuevo informe
        informe = generar_informe(contexto, tematica)
        with open("outputs/informe_tecnico.txt", "w", encoding="utf-8") as f:
            f.write(informe)
        
        # Revisar nuevo informe
        revision = revisar_informe_como_especialista(informe, tematica)
        with open("outputs/revision_informe.txt", "w", encoding="utf-8") as f:
            f.write(revision)
        
        return {
            "informe": informe,
            "revision": revision,
            "tematica": tematica
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
