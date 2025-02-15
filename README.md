# document-ai

# 📄 Document AI

🚀 **Document AI** is a **serverless AI-powered document processing pipeline** that:
- Extracts text from uploaded **PDF, DOCX, and TXT** files.
- Converts text into **vector embeddings** using **OpenAI GPT-4** or **DeepSeek**.
- Stores embeddings in **Pinecone (Vector Database)** for **semantic search**.
- Supports **AI-powered document retrieval**, allowing users to **query documents** and get AI-generated answers.

---

## 🛠️ **Tech Stack**
- **Cloud Services**: AWS Lambda, S3, Pinecone
- **AI/NLP**: OpenAI GPT-4, DeepSeek, text-embedding-ada-002
- **Vector Search**: Pinecone for semantic search
- **File Processing**: pdfplumber (PDFs), python-docx (DOCX)
- **CI/CD**: GitHub Actions for automated deployment

---

## 🚀 **Features**
✅ **Multi-Format Support**: PDF, DOCX, TXT file processing  
✅ **Automatic Embeddings Generation**: Converts text into vector embeddings  
✅ **Efficient Retrieval**: Searches related documents using Pinecone  
✅ **AI-Powered Q&A**: Uses OpenAI GPT-4/DeepSeek to generate intelligent answers  
✅ **Serverless Architecture**: Uses AWS Lambda for seamless automation  
✅ **CI/CD Deployment**: GitHub Actions for automated deployment  

---

## 🔧 **Setup & Deployment**
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/YY35n/document-ai.git
cd document-ai
