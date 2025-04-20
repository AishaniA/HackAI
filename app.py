import streamlit as st
import os
from tempfile import NamedTemporaryFile
from your_backend_module import parse_pdf, setup_retriever, setup_agent  # Assuming you refactor logic into this

st.set_page_config(page_title="📊 AI Report Analyst", layout="wide")
st.title("📊 AI-Powered Annual Report Analyzer")

uploaded_file = st.file_uploader("Upload an Annual Report (PDF)", type=["pdf"])

if uploaded_file is not None:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        temp_pdf_path = temp_pdf.name

    st.success("PDF uploaded successfully!")

    if st.button("🔍 Parse Report"):
        with st.spinner("Extracting text, tables, and images..."):
            parse_result = parse_pdf(temp_pdf_path)
        st.success("Parsing complete!")
        st.session_state.qa_chain = setup_retriever(temp_pdf_path)
        st.session_state.agent = setup_agent(st.session_state.qa_chain)

if "agent" in st.session_state:
    st.markdown("### 💬 Ask a question about the uploaded report")
    user_query = st.text_input("Type your question below:")

    if user_query:
        with st.spinner("Thinking..."):
            answer = st.session_state.agent.run(user_query)
        st.markdown("#### 🧠 Answer:")
        st.markdown(answer)
else:
    st.info("Please upload and parse a PDF to begin querying.")
