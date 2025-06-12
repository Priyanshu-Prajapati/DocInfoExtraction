import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

# Streamlit UI config
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("📄 Chat with your PDF (Personal PDF Assistant)")

uploaded_file = st.file_uploader("📥 Upload your PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing your PDF..."):

        # Save PDF temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Load PDF
        loader = PyPDFLoader("temp.pdf")
        try:
            pages = loader.load_and_split()
        except Exception as e:
            st.error(f"PDF loading error: {e}")
            st.stop()

        if not pages or len(pages) == 0:
            st.warning("⚠️ PDF seems empty or unreadable.")
            st.stop()

        # Split documents
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = splitter.split_documents(pages)

        if not documents:
            st.warning("⚠️ Document could not be split.")
            st.stop()

        # Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(documents, embeddings)
        retriever = vectordb.as_retriever()

        # LLM
        llm = Ollama(model="llama2")  # Make sure this is pulled: `ollama pull llama2`

        # Memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
        )

        # Session state for chat
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

    st.success("✅ PDF is ready! Ask anything below 👇")

    user_question = st.text_input("💬 Your question:")

    if user_question:
        with st.spinner("Thinking..."):
            try:
                result = qa_chain({
                    "question": user_question,
                    "chat_history": st.session_state.chat_history
                })
                answer = result["answer"]
                st.session_state.chat_history.append((user_question, answer))
            except Exception as e:
                st.error(f"❌ Error while answering: {e}")
                st.stop()

        # Show full chat history
        st.markdown("---")
        for q, a in st.session_state.chat_history:
            st.markdown(f"**🧑 You:** {q}")
            st.markdown(f"**🤖 Bot:** {a}")
