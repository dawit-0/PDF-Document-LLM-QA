from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
from streamlit_chat import message

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def create_embeddings(pdf):
    """
    Returns: VectorStore with embeddings that could be loaded or newly created
    """
    if pdf is None:
        return None
    pdf_reader = PdfReader(pdf)

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)

    # Embeddings
    store_name = pdf.name[:-4]
    if os.path.exists(f"{store_name}.pk1"):
        with open(f"{store_name}.pk1", "rb") as f:
            VectorStore = pickle.load(f)
        st.sidebar.write('Document Embeddings Loaded from Disk')
    else:
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pk1", "wb") as f:
            pickle.dump(VectorStore, f)
        st.sidebar.write('Document Embeddings Created and written to Disk')

    return VectorStore


def ask_llm(llm, vector_store, query):
    """
    Returns: response, callback 
    """
    print("sending query to LLM")
    print(query)
    docs = vector_store.similarity_search(query=query, k=3)
    chain = load_qa_chain(llm=llm, chain_type='stuff')

    # Log price and output price of query
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=query)
        return response, cb


def main():
    st.header('Chat w PDF')
    st.sidebar.header('PDF Q/A')
    pdf = st.sidebar.file_uploader('Upload your PDF Document here', type='pdf')
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = []

    if pdf:
        VectorStore = create_embeddings(pdf=pdf)
        openai = OpenAI(
            api_key=OPENAI_API_KEY,
        )

        user_input = st.sidebar.text_input(
            "What's your question about your document?: ")

        if user_input:
            st.session_state.conversation.append(
                {"role": "user", "message": user_input})
            output, _ = ask_llm(
                llm=openai, vector_store=VectorStore, query=user_input)
            st.session_state.conversation.append(
                {"role": "bot", "message": output})

        for message in st.session_state.conversation:
            if message["role"] == "user":
                st.markdown(f'<div style="display:flex; align-items:center; margin-bottom:10px;">'
                            f'<div style="font-size:24px;">🙂</div>'
                            f'<div style="padding:10px; border-radius:10px; display:inline-block;'
                            f'position:relative; margin-bottom:10px; margin-left:10px; border: 1px solid #DCF8C6;">'
                            f'{message["message"]}'
                            f'</div>'
                            f'</div>', unsafe_allow_html=True)
            elif message["role"] == "bot":
                st.markdown(f'<div style="display:flex; align-items:center; margin-bottom:10px;">'
                            f'<div style="font-size:24px;">🤖</div>'
                            f'<div style="padding:10px; border-radius:10px; display:inline-block;'
                            f'position:relative; margin-bottom:10px; margin-left:10px; border: 1px solid #DCF8C6;">'
                            f'{message["message"]}'
                            f'</div>'
                            f'</div>', unsafe_allow_html=True)

        # if query:
        #     print(query)

        #     with st.spinner("Waiting for LLM to generate response ..."):
        #         response, callback = ask_llm(
        #             llm=openai, vector_store=VectorStore, query=query)
        #         st.write(response)
        #         st.write(f"cost of that query: ${callback.total_cost}")


if __name__ == "__main__":
    main()
