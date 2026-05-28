import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.schema import Document

# Set your API key
os.environ["GOOGLE_API_KEY"] = "your-api-key-here"

# Extract transcript from YouTube video
def get_transcript(video_url):
    video_id = video_url.split("v=")[-1]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([t["text"] for t in transcript])
    print(f"✅ Transcript extracted!")
    return text

# Split transcript into chunks
def create_vectorstore(text):
    docs = [Document(page_content=text)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✅ Vector store created!")
    return vectorstore

# Summarize video
def summarize(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = ChatPromptTemplate.from_template("""
    Based on the following context from a YouTube video transcript,
    provide a clear and concise summary.
    Context: {context}
    Question: {question}
    """)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    summary = chain.invoke("Summarize this video in detail.")
    return summary

# Main
if __name__ == "__main__":
    video_url = input("Enter YouTube video URL: ")
    text = get_transcript(video_url)
    vectorstore = create_vectorstore(text)
    
    print("\n🤖 Generating Summary...\n")
    summary = summarize(vectorstore)
    print(f"Summary:\n{summary}")
