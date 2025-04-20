import os
import pymupdf  # PyMuPDF
import pdfplumber
import pandas as pd
import sqlite3
from PIL import Image
from io import BytesIO
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from openai import OpenAI
import re
from unstructured.partition.pdf import partition_pdf
import camelot


# === Set your OpenAI key ===
os.environ["OPENAI_API_KEY"] = ""  # Replace with your key
# === Setup paths ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Or hardcoded

OUTPUT_DIR = "/Users/succhaygadhar/Downloads/output1"
TEXT_PATH = os.path.join(OUTPUT_DIR, "text_chunks.txt")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
DB_PATH = os.path.join(OUTPUT_DIR, "financials.db")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# === Extract text from PDF ===
# === Extract text from PDF ===

def extract_text_chunks(pdf_path):
    doc = pymupdf.open(pdf_path)
    chunks = []
    
    with open(TEXT_PATH, "w", encoding="utf-8") as f:
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            lines = []

            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            line_text += span["text"]
                        if line_text.strip():
                            lines.append(line_text.strip())

            full_page_text = "\n".join(lines)
            paragraphs = re.split(r'\n\s*\n', full_page_text)

            for para in paragraphs:
                para = para.strip()
                if len(para) > 100:
                    chunk = {"page": page_num + 1, "text": para}
                    chunks.append(chunk)
                    f.write(f"[Page {chunk['page']}]\n{chunk['text']}\n\n")

    doc.close()
    return chunks
# === Extract tables and load into SQLite ===
def extract_tables_to_db(pdf_path, db_path):
    conn = sqlite3.connect(db_path)
    tables = []
    table_counter = 1

    print("🔍 Scanning for tables with Camelot...")

    # Try both lattice (with lines) and stream (without lines)
    all_tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')

    print(f"✅ Found {all_tables.n} tables")

    for i, table in enumerate(all_tables):
        try:
            df = table.df

            # First row might be header
            if df.shape[0] < 2:
                continue

            df.columns = [f"col_{j}" for j in range(df.shape[1])] if df.columns.isnull().any() else df.iloc[0]
            df = df[1:] if df.columns.equals(df.iloc[0]) else df

            table_name = f"table_{table_counter}"
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            df.to_csv(os.path.join(TABLES_DIR, f"{table_name}.csv"), index=False)
            tables.append((table_name, df))
            table_counter += 1

        except Exception as e:
            print(f"⚠️ Error parsing table {i}: {e}")
            continue

    conn.commit()
    conn.close()
    return tables


# === Extract images ===
def extract_images(pdf_path):
    doc = pymupdf.open(pdf_path)
    img_count = 0
    for i in range(len(doc)):
        for img_index, img in enumerate(doc.get_page_images(i)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(BytesIO(image_bytes))
            img_path = os.path.join(IMAGES_DIR, f"page{i+1}_img{img_index+1}.png")
            image.save(img_path)
            img_count += 1
    doc.close()
    return img_count

# === Parse PDF and prepare everything ===
def parse_pdf(pdf_path):
    print("Extracting text...")
    text_chunks = extract_text_chunks(pdf_path)

    print("Extracting tables...")
    tables = extract_tables_to_db(pdf_path, DB_PATH)

    print("Extracting images...")
    num_images = extract_images(pdf_path)

    return {
        "text_chunks": text_chunks,
        "tables": tables,
        "num_images": num_images,
        "db_path": DB_PATH
    }

# === Setup LangChain Retrieval QA ===
def setup_retriever(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # slightly bigger chunks
        chunk_overlap=150  # more overlap = more context safety
    )
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})  # 🔥 higher recall
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4"),
        retriever=retriever,
        return_source_documents=True
    )

# === SQL Execution Tool ===
def run_sql_query(query):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.execute(query)
        rows = cursor.fetchall()
        return str(rows)
    except Exception as e:
        return f"SQL Error: {e}"

# === Main Agent Setup ===
def setup_agent(qa_chain):
    doc_qa_tool = Tool(
    name="PDF_QA_Tool",
    func=lambda q: qa_chain.invoke({"query": q})["result"],
    description=(
        "Use this to answer **any questions about the uploaded PDF**. It searches the entire document thoroughly "
        "and uses GPT-4 to reason across multiple sections. Ideal for summarizing, comparing sections, or finding detailed information."
    )
    )


    sql_tool = Tool(
        name="SQL_Tool",
        func=run_sql_query,
        description="Useful for querying tabular data from the financials database using SQL."
    )
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    return initialize_agent(tools=[doc_qa_tool, sql_tool], llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# === Example Run ===
if __name__ == "__main__":
    pdf_path = "/Users/succhaygadhar/Downloads/Untitled document (2).pdf"  # Replace with your file
    parse_pdf(pdf_path)
    qa_chain = setup_retriever(pdf_path)
    agent = setup_agent(qa_chain)

    while True:
        query = input("\n💬 Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        response = agent.run(query)
        print("\n🧠 Response:", response)