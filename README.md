# Proyecto RAG Audio Assistant
# 🧠 RAG Audio Assistant – Agente de IA para Reuniones y Clases

Este proyecto es un MVP de un **agente inteligente** que transforma clases o reuniones grabadas en:

✅ Informes técnicos especializados  
✅ Un chatbot que responde consultas con contexto, mediante RAG

---

## 🔍 ¿Qué hace este agente?

1. **Transcribe o analiza el contenido** (.wav o .txt)  
2. **Detecta la temática principal** usando embeddings y clustering  
3. **Genera un informe profesional** con un LLM especializado  
4. **Valida automáticamente el informe generado**  
5. **Responde preguntas** sobre el contenido mediante búsqueda semántica

---

## ⚙️ Tecnologías utilizadas

| Etapa                          | Tecnología                          |
|-------------------------------|-------------------------------------|
| Transcripción de audio        | Whisper                             |
| Preprocesamiento + Chunking   | LangChain TextSplitter              |
| Embeddings semánticos         | Sentence-Transformers (MiniLM)      |
| Base de datos vectorial       | FAISS                               |
| Detección de temática         | KMeans + LLaMA 3 (Groq API)         |
| Generación + revisión         | Prompt Engineering + LLaMA 3        |
| Interfaz                      | Streamlit                           |

---

## 🚀 Cómo usarlo

1. Cloná el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/rag-audio-assistant.git
   cd rag-audio-assistant

Instalá las dependencias:

bash
Copiar
Editar
pip install -r requirements.txt
Ejecutá la app con Streamlit:

bash
Copiar
Editar
streamlit run app.py