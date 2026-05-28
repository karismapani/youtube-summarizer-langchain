# YouTube Video Summarizer using LangChain

An AI tool that summarizes YouTube videos using LangChain and RAG-based retrieval.

## 🚀 Features
- Enter any YouTube video URL
- Automatically extracts transcript
- Generates concise summary using LLM
- Powered by LangChain and Google Gemini

## 🛠️ Tech Stack
- Python
- LangChain
- YouTube Transcript API
- Google Gemini API
- FAISS (Vector Store)

## ⚙️ How It Works
1. YouTube transcript is extracted from the video
2. Transcript is split into chunks
3. Chunks are stored in FAISS vector store
4. LLM generates a concise summary

## 📦 Installation
pip install langchain-google-genai faiss-cpu youtube-transcript-api langchain-text-splitters langchain-core langchain-community

## 🔑 Setup
Add your Gemini API key:
export GOOGLE_API_KEY="your-api-key"

## 📌 Status
🚧 Under Development# youtube-summarizer-langchain
Summarizes YouTube videos using LangChain and RAG-based retrieval
