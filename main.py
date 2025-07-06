import streamlit as st
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from io import BytesIO
import base64
from gtts import gTTS
import os

# --- Page Config ---
st.set_page_config(page_title="RAG Q&A + Podcast Generator", layout="wide")
st.title("📄 RAG Q&A + Podcast Generator with OpenAI")

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if "collection" not in st.session_state:
    st.session_state.collection = None
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

# --- Manual OpenAI API Key Input ---
st.subheader("🔐 Enter OpenAI API Key")
user_input_key = st.text_input("OpenAI API Key", type="password")

if st.button("Validate"):
    st.session_state["openai_api_key"] = user_input_key

openai_api_key = st.session_state.get("openai_api_key", "")
client = None
api_key_status = {"success": False, "message": ""}

if openai_api_key:
    try:
        client = OpenAI(api_key=openai_api_key)
        client.models.list()
        api_key_status["success"] = True
        api_key_status["message"] = "✅ API Key is valid!"
    except Exception as e:
        api_key_status["message"] = f"❌ Invalid OpenAI API key: {e}"
else:
    api_key_status["message"] = "⚠️ Please enter your OpenAI API key above."

# --- Layout Columns ---
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.subheader("🔐 API Key")
    if api_key_status["success"]:
        st.success(api_key_status["message"])
    else:
        st.error(api_key_status["message"])

    def chunking_string(text, chunk_size):
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def get_embedding(text):
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    st.subheader("📂 Documents")
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

    if uploaded_file:
        doc_text = ""
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                doc_text += page.get_text()

        if not doc_text.strip():
            st.error("❌ The uploaded PDF appears to be empty or contains no extractable text.")
        else:
            st.session_state.doc_text = doc_text
            st.success("✅ Uploaded document is read and stored")
            chunks = chunking_string(doc_text, 30)
            st.success("✅ Uploaded document is chunked")
            persist_dir = "./chromadb_store"
            chroma_client = chromadb.Client(Settings(persist_directory=persist_dir))
            collection = chroma_client.get_or_create_collection(name="rag_db")
            for i, chunk in enumerate(chunks):
                collection.add(
                    ids=[f"chunk-{i+1}"],
                    documents=[chunk],
                    embeddings=[get_embedding(chunk)]
                )
            st.session_state.collection = collection
            st.success("✅ Uploaded document is embedded into ChromaDB")

    if uploaded_file:
        st.markdown("### 📚 Sources")
        st.write(f"- {uploaded_file.name}")

with col2:
    st.subheader("💬 Q&A Chat")
    user_query = st.text_input("Ask a question about the document")

    if st.button("🔍 Get Answer"):
        if user_query == "":
            st.error("❌ No question entered, please re-enter a valid question")
        elif not st.session_state.get("collection"):
            st.error("❌ Please process a PDF first.")
        else:
            with st.spinner("Processing your question..."):
                query_embedding = get_embedding(user_query)
                top_chunks = st.session_state.collection.query(query_embeddings=[query_embedding], n_results=6)
                context = " ".join(top_chunks["documents"][0])
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "user", "content": f"Use the following context to answer the query: {user_query}\nContext: {context}"}
                    ]
                )
                answer = response.choices[0].message.content
            import datetime
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            st.session_state.chat_history.append({"question": user_query, "answer": answer, "timestamp": timestamp})

    if st.session_state.chat_history:
        st.markdown("### 💂️ Chat History")
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"""
                <div style='display: flex; justify-content: flex-end;'>
                    <div style='background-color: #d4edda; padding: 10px; border-radius: 10px; max-width: 80%;'>
                        <strong>YOU-Q{i+1}:</strong> {chat['question']}<br><small>{chat['timestamp']}</small>
                    </div>
                </div>
                <div style='display: flex; justify-content: flex-start; margin-top: 5px;'>
                    <div style='background-color: #f1f1f1; padding: 10px; border-radius: 10px; max-width: 80%;'>
                        <strong>GPT-Answer:</strong> {chat['answer']}<br><small>{chat['timestamp']}</small>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("\n\n")

        chat_text = "\n\n".join([
            f"Q: {c['question']}\nA: {c['answer']}\nT: {c['timestamp']}" for c in st.session_state.chat_history])
        st.download_button("📅 Download Chat History", data=chat_text, file_name="chat_history.txt", mime="text/plain")

        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.utils import simpleSplit

        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        y = 750
        font_name = "Helvetica"
        font_size = 12
        c.setFont(font_name, font_size)
        max_width = 500

        for entry in st.session_state.chat_history:
            lines = [
                f"Q: {entry['question']}",
                f"A: {entry['answer']}",
                f"Time: {entry['timestamp']}"
            ]
            for line in lines:
                wrapped_lines = simpleSplit(line, font_name, font_size, max_width)
                for wrapped_line in wrapped_lines:
                    c.drawString(40, y, wrapped_line)
                    y -= 20
                    if y < 100:
                        c.showPage()
                        c.setFont(font_name, font_size)
                        y = 750
            y -= 10
        c.save()
        pdf_buffer.seek(0)
        st.download_button("📄 Download Chat as PDF", data=pdf_buffer, file_name="chat_history.pdf", mime="application/pdf")

with col3:
    st.subheader("🎷 Podcast Summary")
    if client and st.session_state.doc_text:
        if st.button("🎧 Generate Podcast Summary"):
            with st.spinner("Summarizing document and generating podcast..."):
                summary_response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "user", "content": f"Please summarize this document in a calm, engaging podcast script style:\n{st.session_state.doc_text}"}
                    ]
                )
                summary_text = summary_response.choices[0].message.content
                tts = gTTS(summary_text)
                mp3_fp = BytesIO()
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)

                st.audio(mp3_fp.read(), format="audio/mp3")

                b64 = base64.b64encode(mp3_fp.getvalue()).decode()
                href = f'<a href="data:audio/mp3;base64,{b64}" download="podcast_summary.mp3">📅 Download Podcast MP3</a>'
                st.markdown(href, unsafe_allow_html=True)
