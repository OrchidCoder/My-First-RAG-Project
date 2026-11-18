# 🚀 My First RAG Project

## 🚀 Live Application
**Try the application here:**

[Open My First RAG Application](https://my-first-rag-project-7dvsunuwyaypxlvbxjpvb3.streamlit.app/)


### Retrieval-Augmented Generation using LangChain, Qdrant, FastEmbed and NVIDIA Nemotron

This is my first hands-on **Retrieval-Augmented Generation (RAG)** project. It allows users to ask questions about PDF documents and receive context-aware answers using an AI language model.

I built this project to understand how document processing, vector databases, embeddings, semantic search, and Large Language Models work together in a real-world AI application.


## Project Description

A PDF-based Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions using NVIDIA Nemotron.

## Features

- Upload custom PDF documents.
- Use the default STATS.pdf document.
- Extract and process PDF text.
- Generate embeddings using FastEmbed.
- Store and retrieve document vectors using Qdrant.
- Generate PDF-grounded answers using NVIDIA Nemotron.
- Display detailed answers and simple summaries.

## Technology Stack

Python | Streamlit | LangChain | Qdrant | FastEmbed | NVIDIA Nemotron


## 🎯 Project Overview

The application follows a PDF-based RAG pipeline:

1. Loads and extracts text from PDF documents.
2. Splits documents into smaller text chunks.
3. Generates vector embeddings using FastEmbed.
4. Stores embeddings in an in-memory Qdrant vector database.
5. Retrieves relevant document chunks based on a user question.
6. Sends the retrieved context to NVIDIA Nemotron.
7. Generates answers based on the provided PDF content.

## 🛠️ Technologies Used

- Python
- LangChain
- LangChain Community
- LangChain NVIDIA AI Endpoints
- Qdrant Vector Database
- FastEmbed
- NVIDIA Nemotron LLM
- PyPDF
- Python Dotenv

## 🏗️ Project Architecture

```text
PDF Document
     |
     v
Document Loading (PyPDFLoader)
     |
     v
Text Splitting
     |
     v
FastEmbed Embeddings
     |
     v
Qdrant Vector Database
     |
     v
Semantic Retrieval
     |
     v
Retrieved Context
     |
     v
NVIDIA Nemotron LLM
     |
     v
AI Generated Answer
```

## 📁 Project Structure

```text
My-First-RAG-Project/
│
├── document_loader/
│   └── STATS.pdf
│
├── main.py
├── requirements.txt
├── .env
└── README.md
```

The `.venv` environment is created locally and is not included in GitHub.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/My-First-RAG-Project.git
cd My-First-RAG-Project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Configure API key

Create a `.env` file in the project root:

```env
NVIDIA_API_KEY=your_nvidia_api_key
```

Get your API key from [NVIDIA Build](https://build.nvidia.com/).

### 6. Add your PDF

Place your PDF document inside:

```text
document_loader/STATS.pdf
```

### 7. Run the application

```bash
python main.py
```

## 🧠 Key Concepts Learned

- Understanding Retrieval-Augmented Generation (RAG).
- PDF document loading and processing.
- Text chunking and overlapping.
- Vector embeddings.
- Semantic similarity search.
- Qdrant vector storage.
- Connecting LLMs using LangChain.
- Building a basic document question-answering system.

## 🔮 Future Improvements

- Build a Streamlit-based interactive chatbot.
- Support multiple PDF uploads.
- Add persistent Qdrant storage.
- Implement chat history and conversational memory.
- Improve document retrieval accuracy.
- Add source citations and page references.

## 👨‍💻 Developer

**Kartikey Kashyap**

B.Tech | Computer Science and Engineering (AI & ML)

This is my first step into building practical Generative AI and RAG applications.

⭐ If you find this project useful, consider giving the repository a star!

---

**#Python #GenerativeAI #RAG #LangChain #Qdrant #NVIDIA #MachineLearning #ArtificialIntelligence**
