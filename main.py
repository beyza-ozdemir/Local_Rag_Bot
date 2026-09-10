import os

from flask import Flask, render_template_string, request, jsonify

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ============================================================
# FLASK UYGULAMASI
# ============================================================

app = Flask(__name__)


# ============================================================
# PDF YÜKLEME KLASÖRÜ
# ============================================================

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# GLOBAL DEĞİŞKENLER
# ============================================================

vector_store = None
rag_chain = None


# ============================================================
# HTML ARAYÜZ
# ============================================================

HTML_TEMPLATE = """
<!DOCTYPE html>

<html lang="tr">

<head>

    <meta charset="UTF-8">

    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Local RAG Assistant</title>

    <script src="https://cdn.tailwindcss.com"></script>

</head>


<body class="bg-gray-900 text-gray-100 font-sans min-h-screen flex flex-col items-center justify-center p-4">


<div class="w-full max-w-4xl bg-gray-800 rounded-xl shadow-2xl p-6 border border-gray-700">


    <!-- ================================================= -->
    <!-- BAŞLIK -->
    <!-- ================================================= -->

    <header class="border-b border-gray-700 pb-4 mb-6 flex justify-between items-center">

        <div>

            <h1 class="text-2xl font-bold text-cyan-400">
                Local RAG Application
            </h1>

            <p class="text-sm text-gray-400">
                PDF Bilgi Tabanlı Akıllı Asistan
            </p>

        </div>


        <span class="bg-cyan-900 text-cyan-300 text-xs px-3 py-1 rounded-full font-semibold">

            RAG Aktif

        </span>

    </header>



    <!-- ================================================= -->
    <!-- ANA ALAN -->
    <!-- ================================================= -->

    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">


        <!-- ================================================= -->
        <!-- SOL PANEL -->
        <!-- ================================================= -->

        <div class="bg-gray-900 p-4 rounded-lg border border-gray-700 flex flex-col justify-between">


            <div>

                <h2 class="text-lg font-semibold mb-3 text-cyan-300">

                    1. Bilgi Tabanı (PDF)

                </h2>


                <p class="text-xs text-gray-400 mb-4">

                    Modelinize kaynak olacak bir PDF dokümanı yükleyin.

                </p>


                <input
                    type="file"
                    id="pdfFile"
                    accept=".pdf"
                    class="block w-full text-sm text-gray-400
                    file:mr-4 file:py-2 file:px-4
                    file:rounded-md file:border-0
                    file:text-sm file:font-semibold
                    file:bg-cyan-600 file:text-white
                    hover:file:bg-cyan-700 mb-4"
                >

            </div>


            <button
                onclick="uploadPDF()"
                id="uploadBtn"
                class="w-full bg-cyan-600 hover:bg-cyan-500
                text-white font-medium py-2 px-4 rounded-md
                transition duration-200"
            >

                Dökümanı İndexle

            </button>


            <div
                id="uploadStatus"
                class="text-xs mt-3 text-center text-gray-400"
            ></div>


        </div>



        <!-- ================================================= -->
        <!-- SAĞ PANEL -->
        <!-- ================================================= -->

        <div
            class="md:col-span-2 bg-gray-900 p-4 rounded-lg
            border border-gray-700 flex flex-col h-[450px]"
        >


            <h2 class="text-lg font-semibold mb-3 text-cyan-300">

                2. Akıllı Asistan ile Sohbet

            </h2>


            <div
                id="chatContainer"
                class="flex-1 overflow-y-auto bg-gray-950
                p-3 rounded-md border border-gray-800
                mb-3 space-y-3 text-sm"
            >

                <div class="text-gray-500 text-center italic">

                    Henüz bir doküman yüklenmedi veya soru sorulmadı.

                </div>

            </div>



            <div class="flex gap-2">


                <input
                    type="text"
                    id="userInput"
                    placeholder="Doküman hakkında bir şeyler sorun..."
                    class="flex-1 bg-gray-800 border border-gray-700
                    rounded-md px-3 py-2 text-sm
                    focus:outline-none focus:border-cyan-500"
                >


                <button
                    onclick="sendMessage()"
                    class="bg-cyan-600 hover:bg-cyan-500
                    text-white px-4 py-2 rounded-md
                    font-medium text-sm transition duration-200"
                >

                    Gönder

                </button>


            </div>


        </div>


    </div>


</div>



<script>


// ============================================================
// PDF YÜKLEME
// ============================================================

async function uploadPDF() {


    const fileInput = document.getElementById('pdfFile');

    const statusDiv = document.getElementById('uploadStatus');

    const uploadBtn = document.getElementById('uploadBtn');


    if (fileInput.files.length === 0) {

        alert('Lütfen bir PDF dosyası seçin!');

        return;

    }


    const formData = new FormData();

    formData.append('file', fileInput.files[0]);


    statusDiv.innerText =
        'PDF işleniyor... Terminalde işlem adımlarını görebilirsiniz.';

    statusDiv.className =
        "text-xs mt-3 text-center text-yellow-400 font-medium";


    uploadBtn.disabled = true;

    uploadBtn.innerText = "İşleniyor...";


    try {

        const response = await fetch('/upload', {

            method: 'POST',

            body: formData

        });


        const data = await response.json();


        if (response.ok) {

            statusDiv.innerText = data.message;

            statusDiv.className =
                "text-xs mt-3 text-center text-green-400 font-medium";

        }

        else {

            statusDiv.innerText =
                data.message || "PDF yüklenirken hata oluştu.";

            statusDiv.className =
                "text-xs mt-3 text-center text-red-400 font-medium";

        }


    }


    catch (error) {

        statusDiv.innerText =
            "Sunucuya bağlanırken hata oluştu: " + error;

        statusDiv.className =
            "text-xs mt-3 text-center text-red-400 font-medium";

    }


    finally {

        uploadBtn.disabled = false;

        uploadBtn.innerText = "Dökümanı İndexle";

    }

}



// ============================================================
// SORU SORMA
// ============================================================

async function sendMessage() {


    const inputField =
        document.getElementById('userInput');


    const chatContainer =
        document.getElementById('chatContainer');


    const query =
        inputField.value.trim();


    if (!query) {

        return;

    }


    chatContainer.innerHTML += `

        <div class="text-right">

            <span
                class="inline-block bg-cyan-900
                text-cyan-100 p-2 rounded-lg
                max-w-[80%] text-left"
            >

                ${query}

            </span>

        </div>

    `;


    inputField.value = '';


    chatContainer.scrollTop =
        chatContainer.scrollHeight;


    const loadingId =
        'loading-' + Date.now();


    chatContainer.innerHTML += `

        <div id="${loadingId}" class="text-left">

            <span
                class="inline-block bg-gray-800
                text-gray-400 p-2 rounded-lg italic"
            >

                Düşünüyor...

            </span>

        </div>

    `;


    chatContainer.scrollTop =
        chatContainer.scrollHeight;


    try {

        const response = await fetch('/ask', {

            method: 'POST',

            headers: {

                'Content-Type': 'application/json'

            },

            body: JSON.stringify({

                query: query

            })

        });


        const data =
            await response.json();


        const loadingElement =
            document.getElementById(loadingId);


        if (loadingElement) {

            loadingElement.remove();

        }


        chatContainer.innerHTML += `

            <div class="text-left">

                <span
                    class="inline-block bg-gray-800
                    text-gray-100 p-2 rounded-lg
                    max-w-[80%]"
                >

                    ${data.answer}

                </span>

            </div>

        `;


        chatContainer.scrollTop =
            chatContainer.scrollHeight;


    }


    catch (error) {


        const loadingElement =
            document.getElementById(loadingId);


        if (loadingElement) {

            loadingElement.remove();

        }


        chatContainer.innerHTML += `

            <div class="text-left">

                <span
                    class="inline-block bg-red-900
                    text-red-100 p-2 rounded-lg"
                >

                    Hata oluştu: ${error}

                </span>

            </div>

        `;

    }

}



// ============================================================
// ENTER TUŞU
// ============================================================

document
    .getElementById('userInput')
    .addEventListener('keydown', function(event) {

        if (event.key === 'Enter') {

            sendMessage();

        }

    });


</script>


</body>

</html>
"""


# ============================================================
# ANA SAYFA
# ============================================================

@app.route("/")
def index():

    return render_template_string(HTML_TEMPLATE)


# ============================================================
# PDF YÜKLEME VE INDEXLEME
# ============================================================

@app.route("/upload", methods=["POST"])
def upload_file():

    global vector_store
    global rag_chain


    print("\n")
    print("=" * 60)
    print("PDF INDEXLEME BAŞLADI")
    print("=" * 60)


    # ========================================================
    # 1. DOSYA KONTROLÜ
    # ========================================================

    if "file" not in request.files:

        print("HATA: Dosya bulunamadı!")

        return jsonify({
            "message": "Dosya bulunamadı!"
        }), 400


    file = request.files["file"]


    if file.filename == "":

        print("HATA: Dosya seçilmedi!")

        return jsonify({
            "message": "Dosya seçilmedi!"
        }), 400


    # ========================================================
    # 2. DOSYAYI KAYDET
    # ========================================================

    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )


    print("\n[1] PDF kaydediliyor...")

    file.save(file_path)


    print("[OK] PDF kaydedildi:")
    print(file_path)


    # ========================================================
    # 3. PDF OKUMA
    # ========================================================

    print("\n[2] PDF okunuyor...")


    loader = PyPDFLoader(file_path)

    documents = loader.load()


    print("[OK] PDF okundu.")

    print("Sayfa sayısı:", len(documents))


    # ========================================================
    # 4. PDF'İ CHUNK'LARA AYIR
    # ========================================================

    print("\n[3] PDF metni parçalara ayrılıyor...")


    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=500,

        chunk_overlap=50

    )


    docs = text_splitter.split_documents(
        documents
    )


    print("[OK] PDF parçalandı.")

    print("Chunk sayısı:", len(docs))


    # ========================================================
    # 5. EMBEDDING MODELİ
    # ========================================================

    print("\n[4] Embedding modeli yükleniyor...")

    print("Model: all-MiniLM-L6-v2")

    print("İlk çalıştırmada model indirilebilir.")


    embeddings = HuggingFaceEmbeddings(

        model_name="all-MiniLM-L6-v2"

    )


    print("[OK] Embedding modeli hazır.")


    # ========================================================
    # 6. CHROMA VECTOR DATABASE
    # ========================================================

    print("\n[5] Chroma vector database oluşturuluyor...")

    print("Chunk'ların embedding'leri oluşturuluyor...")

    print("Bu işlem PDF boyutuna göre zaman alabilir.")


    vector_store = Chroma.from_documents(

        docs,

        embeddings

    )


    print("[OK] Chroma vector database hazır.")


    # ========================================================
    # 7. LOCAL LLM - OLLAMA
    # ========================================================

    print("\n[6] Local LLM hazırlanıyor...")

    print("Model: llama3.2")

    print("LLM Ollama üzerinden bilgisayarda çalışıyor.")


    llm = ChatOllama(

        model="llama3.2",

        temperature=0

    )


    print("[OK] Local LLM hazır.")


    # ========================================================
    # 8. RETRIEVER
    # ========================================================

    print("\n[7] Retriever oluşturuluyor...")


    retriever = vector_store.as_retriever(

        search_kwargs={
            "k": 2
        }

    )


    print("[OK] Retriever hazır.")


    # ========================================================
    # 9. DOKÜMAN FORMATLAMA
    # ========================================================

    def format_docs(docs):

        return "\n\n".join(

            doc.page_content

            for doc in docs

        )


    # ========================================================
    # 10. PROMPT
    # ========================================================

    print("\n[8] Prompt oluşturuluyor...")


    prompt = ChatPromptTemplate.from_template(

        """
        Sen PDF dokümanına dayalı çalışan bir RAG asistanısın.

        SADECE aşağıdaki bağlamda verilen bilgileri kullanarak
        soruyu cevapla.

        Eğer sorunun cevabı bağlamda bulunmuyorsa:

        "Bu bilgi verilen dokümanda bulunmamaktadır."

        şeklinde cevap ver.

        Cevabı Türkçe ve anlaşılır şekilde ver.

        Bağlam:
        {context}

        Soru:
        {input}

        Cevap:
        """

    )


    # ========================================================
    # 11. RAG CHAIN
    # ========================================================

    print("[9] RAG zinciri oluşturuluyor...")


    rag_chain = (

        {

            "context":
                retriever | format_docs,

            "input":
                RunnablePassthrough()

        }

        | prompt

        | llm

        | StrOutputParser()

    )


    print("[OK] RAG zinciri hazır.")


    print("\n")
    print("=" * 60)
    print("PDF BAŞARIYLA INDEXLENDİ!")
    print("=" * 60)
    print("\n")


    return jsonify({

        "message":
            "PDF başarıyla yüklendi ve indekslendi!"

    })


# ============================================================
# SORU SORMA
# ============================================================

@app.route("/ask", methods=["POST"])
def ask_question():

    global rag_chain


    data = request.get_json()


    if not data:

        return jsonify({

            "answer":
                "Geçerli bir istek gönderilmedi."

        }), 400


    query = data.get(
        "query",
        ""
    )


    print("\n")
    print("=" * 60)
    print("YENİ SORU")
    print("=" * 60)

    print("Soru:", query)


    if not query:

        return jsonify({

            "answer":
                "Lütfen bir soru yazın."

        }), 400


    # ========================================================
    # PDF YÜKLENMİŞ Mİ?
    # ========================================================

    if not rag_chain:

        print("HATA: Henüz RAG zinciri yok.")


        return jsonify({

            "answer":
                "Lütfen önce bir PDF yükleyin ve indeksleyin."

        })


    try:

        print("\n[1] Retriever ilgili bilgileri arıyor...")


        # ====================================================
        # RAG SORGUSU
        # ====================================================

        response = rag_chain.invoke(
            query
        )


        print("[OK] Local LLM cevap oluşturdu.")

        print("Cevap:", response)


        return jsonify({

            "answer":
                response

        })


    except Exception as e:


        print("\nHATA!")

        print(str(e))


        return jsonify({

            "answer":
                f"Bir hata oluştu: {str(e)}"

        }), 500


# ============================================================
# UYGULAMAYI BAŞLAT
# ============================================================

if __name__ == "__main__":


    print("\n")
    print("=" * 60)
    print("LOCAL RAG APPLICATION BAŞLATILIYOR")
    print("=" * 60)

    print("Adres: http://127.0.0.1:5001")

    print("LLM: Ollama / llama3.2")

    print("OpenAI API: KULLANILMIYOR")

    print("=" * 60)
    print("\n")


    app.run(

        debug=True,

        port=5001

    )
