from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
import streamlit
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


def create_home_page():
    with streamlit.sidebar:
        streamlit.title('Chat with your PDF 🤖🤪')
        streamlit.markdown(
            '''
            ## About
            Document chatbot powered by LLMs

            Upload your pdf document, and ask any questions you have about it!
            '''
        )
        add_vertical_space(5)
        streamlit.write("Made by Dawit 😈")

    streamlit.header('Chat w PDF')


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
        streamlit.write('Embeddings Loaded from Disk')
    else:
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pk1", "wb") as f:
            pickle.dump(VectorStore, f)
        streamlit.write('Embeddings Created and written to Disk')

    return VectorStore


def ask_llm(llm, vector_store, query):
    """
    Returns: response, callback 
    """
    docs = vector_store.similarity_search(query=query, k=3)
    chain = load_qa_chain(llm=llm, chain_type='stuff')

    # Log price and output price of query
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=query)
        return response, cb


def main():
    create_home_page()
    pdf = streamlit.file_uploader('Upload your PDF Document', type='pdf')

    if pdf:
        VectorStore = create_embeddings(pdf=pdf)
        openai = OpenAI()

        query = streamlit.text_input(
            "What's your question about your document?: ")

        if query:
            streamlit.write(query)
            with streamlit.spinner("Waiting for LLM to generate response ..."):
                response, callback = ask_llm(
                    llm=openai, vector_store=VectorStore, query=query)
            streamlit.write(response)
            streamlit.write(f"cost of that query: ${callback.total_cost}")


if __name__ == "__main__":
    main()
