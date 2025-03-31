import streamlit as st
import tempfile
import os
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#.sidebar.header("Resume Summarizer")
pdf_file = st.sidebar.file_uploader("Upload Resume (PDF)", type=["pdf"])
#um_points = st.sidebar.number_input("Number of Summary Points", min_value=1, max_value=10, value=5)

#st.set_page_config(layout="wide")
#st.title("Reliance Jio - Recruiter Feedback Performance System")
#t.markdown("<h1 style='text-align: center;'> Reliance Jio - Recruiter Feedback Performance System</h1>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: center;">
        <img src="https://e7.pngegg.com/pngimages/33/16/png-clipart-jio-logo-jio-reliance-digital-business-logo-mobile-phones-business-blue-text.png" width="100">
        <h1>Reliance Jio - Resume Summarizer</h1>
    </div>
    """, 
    unsafe_allow_html=True
)

if pdf_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.read())
        temp_pdf_path = temp_pdf.name
    
    with st.spinner("Extracting text from resume..."):
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
    
    with st.spinner("Generating summary..."):
# Prepare a custom prompt to instruct the model on how to summarize the resume
        prompt = """
        You are an expert summarizer. Please read the resume and summarize it in a bullet-point format. 
        Each bullet point should be a concise and clear statement, focusing on key achievements, skills, experience, and education.
        Here is the content of the resume:
        
        {text}
        
        Please return your summary in bullet-point format.
        """

        # Combine the document chunks into a single text input for the model
        #full_text = " ".join([doc.page_content for doc in docs])
        llm = ChatGroq(model_name="mistral-saba-24b", groq_api_key=GROQ_API_KEY, temperature=0)
        #formatted_prompt = prompt.format(text=full_text)
        #response = llm.invo(formatted_prompt)
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)
    
    st.success("Summary Generated:")
    formatted_summary = "\n- ".join([sentence.strip() for sentence in summary.split(". ") if sentence.strip()])
    formatted_summary = "- " + formatted_summary  # Add a bullet point at the start
    #st.write(response["text"])
    st.write(formatted_summary)
