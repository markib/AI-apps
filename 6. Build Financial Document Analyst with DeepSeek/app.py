import os
import streamlit as st
from pathlib import Path
from PyPDF2 import PdfReader
from PIL import Image
import tempfile
import shutil
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers import VectorStoreRetriever
from langchain.llms import Ollama
import chromadb

VECTOR_DB_FOLDER = "vector_databases"
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)


def display_pdf_in_sidebar(pdf_path):
    pdf_folder = Path(pdf_path).stem
    pdf_images_folder = Path("pdf_images") / pdf_folder
    pdf_images_folder.mkdir(parents=True, exist_ok=True)
    if not any(pdf_images_folder.iterdir()):
        try:
            pdf = PdfReader(pdf_path)
            for page_num in range(len(pdf.pages)):
                page = pdf.pages[page_num]
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".png"
                ) as temp_image_file:
                    temp_image_path = temp_image_file.name
                    Image.frombytes(
                        "RGB", (page.mediaBox[2], page.mediaBox[3]), page.extract_text()
                    ).save(temp_image_path)
                    shutil.move(
                        temp_image_path, pdf_images_folder / f"page_{page_num + 1}.png"
                    )
        except Exception as e:
            st.sidebar.error(f"Error processing PDF: {e}")
    for image_file in sorted(pdf_images_folder.iterdir()):
        st.sidebar.image(str(image_file))


def process_pdf_to_vector_store(pdf_path, vector_store_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    embeddings = OllamaEmbeddings()
    vector_store = Chroma.from_documents(docs, embeddings)
    vector_store.save(vector_store_path)


def load_vector_store(vector_store_path):
    embeddings = OllamaEmbeddings()
    vector_store = Chroma.load(vector_store_path, embeddings)
    return vector_store


st.title("Financial Document Analyst")

vector_db_options = ["Upload a new document"] + [
    f.stem for f in Path(VECTOR_DB_FOLDER).glob("*.chroma")
]
selected_vector_db = st.selectbox("Select a vector database", vector_db_options)

if selected_vector_db == "Upload a new document":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
            temp_pdf_path = temp_pdf_file.name
            temp_pdf_file.write(uploaded_file.read())
        display_pdf_in_sidebar(temp_pdf_path)
        if st.button("Process Document"):
            vector_store_path = (
                Path(VECTOR_DB_FOLDER) / f"{Path(temp_pdf_path).stem}.chroma"
            )
            process_pdf_to_vector_store(temp_pdf_path, vector_store_path)
            shutil.move(temp_pdf_path, vector_store_path.with_suffix(".pdf"))
            st.success("Document processed and vector store created.")
else:
    vector_store_path = Path(VECTOR_DB_FOLDER) / f"{selected_vector_db}.chroma"
    pdf_path = vector_store_path.with_suffix(".pdf")
    if pdf_path.exists():
        display_pdf_in_sidebar(pdf_path)
    else:
        st.warning("PDF file not found.")
    vector_store = load_vector_store(vector_store_path)

    question = st.text_input("Enter your question:")
    if st.button("Submit Question"):
        retriever = VectorStoreRetriever(vector_store)
        rag_chain = RetrievalQAWithSourcesChain(retriever=retriever, llm=Ollama())
        response = rag_chain.run(question)
        st.write(response)
