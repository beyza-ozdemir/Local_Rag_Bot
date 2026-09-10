import os
from flask import Flask, render_template_string, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

app = Flask(__name__)

# Geçici dosya yükleme klasörü
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global değişkenler
vector_store = None
rag_chain = None

# Arayüz HTML Kodu
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Local RAG Assistant (Foundry Style)</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-gray-100 font-sans min-h-screen flex flex-col items-center justify-center p-4">
    <div class="w-full max-w-4xl bg-gray-800 rounded-xl shadow-2xl p-6 border border-gray-700">
        <header class="border-b border-gray-700 pb-4 mb-6 flex justify-between items-center">
            <div>
                <h1 class="text-2xl font-bold text-cyan-400">Local RAG Application</h1>
                <p class="text-sm text-gray-400">Azure AI Foundry & Local LLM Mimarisi</p>
            </div>
            <span class="bg-cyan-900 text-cyan-300 text-xs px-3 py-1 rounded-full font-semibold">Aktif Model: Local LLM</span>
        </header>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
            <!-- Sol Panel: Doküman Yükleme -->
            <div class="bg-gray-900 p-4 rounded-lg border border-gray-700 flex flex-col justify-between">
                <div>
                    <h2 class="text-lg font-semibold mb-3 text-cyan-300">1. Bilgi Tabanı (PDF)</h2>
                    <p class="text-xs text-gray-400 mb-4">Modelinize kaynak olacak bir PDF dökümanı yükleyin.</p>
                    <input type="file" id="pdfFile" accept=".pdf" class="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-cyan-600 file:text-white hover:file:bg-cyan-700 mb-4">
                </div>
                <button onclick="uploadPDF()" id="uploadBtn" class="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-medium py-2 px-4 rounded-md transition duration-200">Dökümanı İndexle</button>
                <div id="uploadStatus" class="text-xs mt-3 text-center text-gray-400"></div>
            </div>

            <!-- Sağ Panel: Sohbet ve Sorgu Ekranı -->
            <div class="md:col-span-2 bg-gray-900 p-4 rounded-lg border border-gray-700 flex flex-col h-[450px]">
                <h2 class="text-lg font-semibold mb-3 text-cyan-300">2. Akıllı Asistan ile Sohbet</h2>
                <div id="chatContainer" class="flex-1 overflow-y-auto bg-gray-950 p-3 rounded-md border border-gray-800 mb-3 space-y-3 text-sm">
                    <div class="text-gray-500 text-center italic">Henüz bir döküman yüklenmedi veya soru sorulmadı.</div>
                </div>
                <div class="flex gap-2">
                    <input type="text" id="userInput" placeholder="Döküman hakkında bir şeyler sorun..." class="flex-1 bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-sm focus:outline-none focus:border-cyan-500">
                    <button onclick="sendMessage()" class="bg-cyan-600 hover:bg-cyan-500 text-white px-4 py-2 rounded-md font-medium text-sm transition duration-200">Gönder</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function uploadPDF() {
            const fileInput = document.getElementById('pdfFile');
            const statusDiv = document.getElementById('uploadStatus');
            if (fileInput.files.length === 0) {
                alert('Lütfen bir PDF dosyası seçin!');
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            statusDiv.innerText = 'Yükleniyor ve vektör veritabanına işleniyor...';

            const response = await fetch('/upload', { method: 'POST', body: formData });
            const data = await response.json();
            statusDiv.innerText = data.message;
            statusDiv.className = "text-xs mt-3 text-center text-green-400 font-medium";
        }

        async function sendMessage() {
            const inputField = document.getElementById('userInput');
            const chatContainer = document.getElementById('chatContainer');
            const query = inputField.value.trim();
            if (!query) return;

            chatContainer.innerHTML += `<div class="text-right"><span class="inline-block bg-cyan-900 text-cyan-100 p-2 rounded-lg max-w-[80%] text-left">${query}</span></div>`;
            inputField.value = '';
            chatContainer.scrollTop = chatContainer.scrollHeight;

            const loadingId = 'loading-' + Date.now();
            chatContainer.innerHTML += `<div id="${loadingId}" class="text-left"><span class="inline-block bg-gray-800 text-gray-400 p-2 rounded-lg italic">Düşünüyor...</span></div>`;
            chatContainer.scrollTop = chatContainer.scrollHeight;

            const response = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query })
            });
            const data = await response.json();

            document.getElementById(loadingId).remove();
            chatContainer.innerHTML += `<div class="text-left"><span class="inline-block bg-gray-800 text-gray-100 p-2 rounded-lg max-w-[80%]">${data.answer}</span></div>`;
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/upload", methods=["POST"])
def upload_file():
    global vector_store, rag_chain
    if "file" not in request.files:
        return jsonify({"message": "Dosya bulunamadı!"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message": "Dosya seçilmedi!"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # PDF'i yükle ve parçala
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Yerel Embeddings ve Vektör Veritabanı
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(docs, embeddings)

    # Saf LCEL RAG Zinciri (Dış zincir modülü gerektirmez)
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template("""Aşağıdaki bağlamı kullanarak soruyu yanıtla:
{context}

Soru: {input}""")

    rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return jsonify({"message": "Başarıyla yüklendi ve indekslendi!"})


@app.route("/ask", methods=["POST"])
def ask_question():
    global rag_chain
    data = request.get_json()
    query = data.get("query", "")

    if not rag_chain:
        return jsonify({"answer": "Lütfen önce sol taraftan bir PDF dökümanı yükleyin ve indeksleyin."})

    try:
        response = rag_chain.invoke(query)
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"answer": f"Bir hata oluştu: {str(e)}"})


if __name__ == "__main__":
    app.run(debug=True, port=5001)