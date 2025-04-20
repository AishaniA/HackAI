import streamlit as st
import os
from tempfile import NamedTemporaryFile
from backend import parse_pdf, load_csv_to_db, setup_retriever, setup_agent  # Make sure you import these

# === Streamlit App Setup ===
st.set_page_config(page_title="📊 AI Report Analyst", layout="wide")
st.title("📊 AI-Powered Data Analyzer")

# === File Upload ===
uploaded_file = st.file_uploader("Upload a Report (PDF or CSV)", type=["pdf", "csv"])

if uploaded_file is not None:
    file_suffix = os.path.splitext(uploaded_file.name)[-1].lower()

    # Save uploaded file temporarily
    with NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    st.success(f"{file_suffix.upper()} uploaded successfully!")

    # === Parse PDF or CSV ===
    if st.button("🔍 Parse Report"):
        with st.spinner("Processing..."):
            if file_suffix == ".pdf":
                parse_result = parse_pdf(temp_file_path)
                qa_chain = setup_retriever(temp_file_path)
            elif file_suffix == ".csv":
                from backend import DB_PATH  # ensure DB_PATH is accessible
                load_csv_to_db(temp_file_path, DB_PATH)
                qa_chain = None  # no retriever for CSV
            else:
                st.error("Unsupported file type.")
                st.stop()

            st.session_state.agent = setup_agent(qa_chain)
        st.success("Parsing complete!")

# === Query Interface ===
if "agent" in st.session_state:
    st.markdown("### 💬 Ask a question about the uploaded report")
    user_query = st.text_input("Type your question below:")

    if user_query:
        with st.spinner("Thinking..."):
            answer = st.session_state.agent.run(user_query)
        st.markdown("#### 🧠 Answer:")
        st.markdown(answer)
else:
    st.info("Please upload and parse a PDF or CSV to begin querying.")
