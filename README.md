

https://github.com/user-attachments/assets/8d79a09f-e7c4-475b-85bf-dd770eed1fd1



<img width="1482" height="887" alt="Ekran görüntüsü 2026-09-10 211442" src="https://github.com/user-attachments/assets/d77a61a2-6ce3-42a1-b998-974498b2d440" />

Tabii. Bu proje için GitHub'a koyabileceğin **hazır, profesyonel ama öğrenci projesine uygun bir `README.md`** hazırladım. Projenin gerçekten kullandığı teknolojilere göre yazdım; özellikle **“local LLM”** ifadesini düzelttim çünkü kodunda embedding yerel olsa da `ChatOpenAI` ile OpenAI API kullanılıyor.

# Local RAG Bot

PDF dokümanları üzerinden soru-cevap yapabilen, **Retrieval-Augmented Generation (RAG)** mimarisi ile geliştirilmiş yapay zekâ destekli bir web uygulamasıdır.

Kullanıcı sisteme bir PDF yükler. Uygulama PDF içerisindeki metni parçalara ayırır, embedding oluşturur ve bu bilgileri **ChromaDB** üzerinde vektör olarak saklar. Kullanıcı bir soru sorduğunda, sistem soruyla en alakalı doküman parçalarını bulur ve bu bilgileri **OpenAI LLM** modeline göndererek cevap oluşturur.

---

## 🚀 Özellikler

* PDF dosyası yükleme
* PDF içerisindeki metinleri otomatik olarak okuma
* Metinleri küçük parçalara ayırma
* HuggingFace ile yerel embedding oluşturma
* ChromaDB ile vector database kullanımı
* Kullanıcı sorusuna göre ilgili doküman parçalarını bulma
* OpenAI LLM ile cevap üretme
* Basit ve kullanıcı dostu web arayüzü
* Flask backend
* Chat tabanlı soru-cevap sistemi

---

## 🧠 RAG Mimarisi

Uygulamanın temel çalışma mantığı:

```text
              PDF
               │
               ▼
        PyPDFLoader
               │
               ▼
       Text Splitting
               │
               ▼
       HuggingFace
        Embeddings
               │
               ▼
          ChromaDB
       Vector Database
               │
               │
       User Question
               │
               ▼
          Retriever
               │
               ▼
     Relevant PDF Chunks
               │
               ▼
            Prompt
               │
               ▼
        OpenAI LLM
               │
               ▼
            Answer
```

### RAG Nasıl Çalışır?

1. Kullanıcı PDF dosyasını yükler.
2. `PyPDFLoader` PDF içerisindeki metni okur.
3. `RecursiveCharacterTextSplitter` metni küçük parçalara ayırır.
4. `HuggingFaceEmbeddings` bu parçaları sayısal vektörlere dönüştürür.
5. Vektörler ChromaDB içerisinde saklanır.
6. Kullanıcı bir soru sorar.
7. Retriever, soruyla en alakalı doküman parçalarını bulur.
8. Bulunan parçalar prompt içerisine eklenir.
9. OpenAI LLM bu bilgileri kullanarak cevap oluşturur.
10. Cevap web arayüzünde kullanıcıya gösterilir.

---

## 🛠️ Kullanılan Teknolojiler

| Teknoloji                      | Kullanım Amacı                  |
| ------------------------------ | ------------------------------- |
| Python                         | Backend ve uygulama geliştirme  |
| Flask                          | Web uygulaması ve API           |
| LangChain                      | RAG pipeline                    |
| PyPDFLoader                    | PDF dosyalarını okuma           |
| RecursiveCharacterTextSplitter | Metin parçalama                 |
| HuggingFace Embeddings         | Metinleri vektörlere dönüştürme |
| ChromaDB                       | Vector database                 |
| OpenAI                         | LLM ve cevap üretimi            |
| HTML                           | Web arayüzü                     |
| JavaScript                     | Kullanıcı etkileşimleri         |
| Tailwind CSS                   | Arayüz tasarımı                 |

---

## 📁 Proje Yapısı

```text
Local_Rag_Bot/
│
├── app.py
├── uploads/
│   └── uploaded_pdf_files
│
└── README.md
```

### `app.py`

Uygulamanın ana Python dosyasıdır.

Flask uygulamasını, PDF işleme sürecini, embedding oluşturmayı, ChromaDB'yi, Retriever'ı ve RAG chain'i içerir.

### `uploads/`

Kullanıcı tarafından yüklenen PDF dosyalarının geçici olarak saklandığı klasördür.

### `README.md`

Proje hakkında açıklamalar ve kurulum bilgilerini içerir.

---

## ⚙️ Kurulum

### 1. Projeyi klonlayın

```bash
git clone https://github.com/beyza-ozdemir/Local_Rag_Bot.git
```

Proje klasörüne girin:

```bash
cd Local_Rag_Bot
```

---

### 2. Sanal ortam oluşturun

Windows:

```bash
python -m venv venv
```

Sanal ortamı aktif edin:

```bash
venv\Scripts\activate
```

---

### 3. Gerekli paketleri yükleyin

```bash
pip install flask
pip install langchain
pip install langchain-community
pip install langchain-openai
pip install langchain-text-splitters
pip install chromadb
pip install sentence-transformers
pip install pypdf
```

Alternatif olarak bir `requirements.txt` dosyası oluşturularak bütün bağımlılıklar tek seferde kurulabilir:

```bash
pip install -r requirements.txt
```

---

## 🔑 OpenAI API Key

Projede cevap üretmek için OpenAI API kullanılmaktadır.

Windows PowerShell üzerinde:

```powershell
$env:OPENAI_API_KEY="YOUR_API_KEY"
```

Linux / macOS:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```

API anahtarınızı doğrudan Python kodunun içerisine yazmak yerine environment variable kullanılması önerilir.

> **Not:** API anahtarınızı GitHub'a kesinlikle yüklemeyin.

---

## ▶️ Uygulamayı Çalıştırma

Projeyi çalıştırmak için:

```bash
python app.py
```

Flask uygulaması başlatıldıktan sonra tarayıcıdan:

```text
http://127.0.0.1:5001
```

adresine gidilebilir.

---

## 💬 Kullanım

### 1. PDF yükleyin

Arayüzdeki:

**"Dökümanı İndexle"**

butonuna tıklayın.

Uygulama:

* PDF'i okur
* Metni parçalara ayırır
* Embedding oluşturur
* ChromaDB'ye kaydeder
* RAG chain'i oluşturur

---

### 2. Soru sorun

PDF başarıyla indekslendikten sonra sohbet alanına bir soru yazabilirsiniz.

Örneğin:

```text
Docker nedir?
```

veya:

```text
Bu dokümanda hangi teknolojilerden bahsediliyor?
```

Sistem PDF içerisinden soruyla ilişkili bilgileri bulur ve OpenAI modeli kullanarak cevap oluşturur.

---

## 🔌 API Endpoints

Uygulamada Flask tarafından sağlanan temel endpoint'ler:

### `GET /`

Ana web arayüzünü görüntüler.

### `POST /upload`

PDF dosyasını yükler ve RAG bilgi tabanını oluşturur.

Örnek:

```text
POST /upload
Content-Type: multipart/form-data
```

Dosya parametresi:

```text
file
```

### `POST /ask`

Kullanıcının sorusunu RAG sistemine gönderir.

Request:

```json
{
    "query": "Docker nedir?"
}
```

Response:

```json
{
    "answer": "Docker ..."
}
```

---

## 🔍 RAG Chain

Projede LangChain Expression Language (LCEL) kullanılarak RAG pipeline oluşturulmuştur.

```python
rag_chain = (
    {"context": retriever | format_docs,
     "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

Pipeline şu şekilde çalışmaktadır:

```text
User Query
    ↓
Retriever
    ↓
Relevant Documents
    ↓
Context
    ↓
Prompt
    ↓
OpenAI LLM
    ↓
String Output
```

---

## 📌 Embedding Model

Projede:

```text
all-MiniLM-L6-v2
```

modeli kullanılmaktadır.

Embedding işlemi yerel olarak gerçekleştirildiği için PDF metinlerinin vektörleştirilmesi için ayrı bir embedding API'sine ihtiyaç duyulmaz.

---

## 🗄️ Vector Database

Proje **ChromaDB** kullanmaktadır.

ChromaDB'nin görevi, PDF içerisindeki metin parçalarının embedding'lerini saklamak ve kullanıcı sorusuna en yakın parçaların bulunmasını sağlamaktır.

Retriever yapılandırması:

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)
```

Buradaki `k=2`, kullanıcı sorusuyla en alakalı iki doküman parçasının getirilmesini sağlar.

---

## 🖥️ Arayüz

Uygulamanın frontend kısmında:

* HTML
* JavaScript
* Tailwind CSS

kullanılmıştır.

Arayüz iki ana bölümden oluşmaktadır:

### Bilgi Tabanı

PDF yükleme ve indeksleme işlemleri.

### Akıllı Asistan

Kullanıcının PDF hakkında soru sorduğu sohbet alanı.

---

## 🔐 Güvenlik Notu

OpenAI API anahtarının kaynak kod içerisinde tutulmaması gerekir.

Örneğin:

```python
# ÖNERİLMEZ
api_key = "sk-..."
```

yerine environment variable kullanılması önerilir:

```text
OPENAI_API_KEY
```

Ayrıca `.env` veya API anahtarlarını içeren dosyalar `.gitignore` içerisine eklenmelidir.

Örnek:

```text
.env
venv/
__pycache__/
uploads/
*.pyc
```

---

## 🚧 Gelecekte Geliştirilebilecek Özellikler

Projeye ilerleyen aşamalarda aşağıdaki özellikler eklenebilir:

* Birden fazla PDF yükleme
* Kalıcı ChromaDB kullanımı
* PDF sayfa numarası ile kaynak gösterme
* Cevaplarda kaynak doküman parçalarını gösterme
* Chat geçmişinin saklanması
* Kullanıcı authentication sistemi
* Daha gelişmiş LLM modelleri
* Streaming response
* Docker container desteği
* Production deployment
* Dosya yönetim sistemi

---

## 🎯 Projenin Amacı

Bu proje ile **Retrieval-Augmented Generation (RAG)** mimarisinin gerçek bir web uygulaması üzerinde uygulanması amaçlanmıştır.

Proje kapsamında:

* PDF processing
* Text chunking
* Embedding
* Vector database
* Semantic retrieval
* Prompt engineering
* LLM integration
* Flask API
* Web interface

gibi farklı teknolojiler bir araya getirilmiştir.

---

## 👩‍💻 Author

**Beyza Özdemir**

Computer Engineering Student

GitHub:
[https://github.com/beyza-ozdemir](https://github.com/beyza-ozdemir)

---

## 📄 License

This project was developed for educational and demonstration purposes.

**Bir önemli nokta:** Senin kodunda `ChatOpenAI(model="gpt-3.5-turbo")` olduğu için README'de **“Local LLM”** dememek daha doğru. `all-MiniLM-L6-v2` embedding modeli lokal çalışıyor, fakat cevap üretme kısmında OpenAI API kullanıyorsun. Bu ayrımı README'de özellikle doğru bıraktım.
