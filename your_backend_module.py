import os
import pymupdf  # PyMuPDF
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
import matplotlib.pyplot as plt
import json
import openai
import streamlit as st



from unstructured.partition.pdf import partition_pdf
import camelot
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI


# === Set your OpenAI key ===
os.environ["OPENAI_API_KEY"] = st.secrets["OpenAI_key"]
# === Setup paths ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Or hardcoded

OUTPUT_DIR = "/Users/succhaygadhar/Downloads/output6"
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
import pymupdf  # PyMuPDF

def extract_tables_to_db(pdf_path, db_path):
    import sqlite3
    conn = sqlite3.connect(db_path)
    tables = []
    table_counter = 1

    doc = pymupdf.open(pdf_path)

    for page_num, page in enumerate(doc):
        tabs = page.find_tables()
        if not tabs or not tabs.tables:
            continue

        for tnum, tab in enumerate(tabs.tables):
            try:
                df = tab.to_pandas()

                # Fallback header fix
                df.columns = [
                    col if col and str(col).strip() else f"col_{i+1}"
                    for i, col in enumerate(df.columns)
                ]
                df.columns = make_unique_columns(df.columns)

                table_name = f"table_page{page_num+1}_{table_counter}"
                csv_path = os.path.join(TABLES_DIR, f"{table_name}.csv")
                df.to_csv(csv_path, index=False)
                df.to_sql(table_name, conn, if_exists="replace", index=False)

                tables.append((table_name, df))
                print(f"✅ Extracted (MuPDF) table on page {page_num + 1} → {table_name}")
                table_counter += 1
            except Exception as e:
                print(f"❌ Failed to extract table on page {page_num+1}: {e}")

    conn.commit()
    conn.close()
    return tables

def load_csv_to_db(csv_path, db_path):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_csv(csv_path)

        # Ensure valid column names
        df.columns = [
            col if col and str(col).strip() else f"col_{i+1}"
            for i, col in enumerate(df.columns)
        ]
        df.columns = make_unique_columns(df.columns)

        table_name = os.path.splitext(os.path.basename(csv_path))[0]
        df.to_sql(table_name, conn, if_exists="replace", index=False)

        print(f"✅ CSV loaded into table: {table_name}")
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
    finally:
        conn.commit()
        conn.close()


# Helper: ensures valid + unique column names
def make_unique_columns(columns):
    seen = {}
    unique = []
    for col in columns:
        col = col if col and str(col).strip() else "column"
        if col in seen:
            seen[col] += 1
            unique.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            unique.append(col)
    return unique

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

from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# === BLIP caption generator setup (load once) ===
def setup_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# === Generate captions for extracted images ===
def caption_images(image_dir):
    processor, model = setup_blip_model()
    captions = {}
    all_caption_lines = []

    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path).convert("RGB")

            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)

            captions[filename] = caption
            print(f"🖼 {filename} → 📝 {caption}")

            # Save individual caption file (optional)
            with open(os.path.join(image_dir, f"{filename}.txt"), "w") as f:
                f.write(caption)

            # Add to list for combined text file
            all_caption_lines.append(f"{filename}:\n{caption}\n")

    # Save all captions to one text file
    all_caption_path = os.path.join(OUTPUT_DIR, "image_captions.txt")
    with open(all_caption_path, "w", encoding="utf-8") as f:
        f.writelines(all_caption_lines)

    return captions


# === Parse PDF and prepare everything ===
def parse_pdf(pdf_path):
    print("Extracting text...")
    text_chunks = extract_text_chunks(pdf_path)

    print("Extracting tables...")
    tables = extract_tables_to_db(pdf_path, DB_PATH)

    print("Extracting images...")
    num_images = extract_images(pdf_path)

    print("Generating image captions...")
    image_captions = caption_images(IMAGES_DIR)

    return {
        "text_chunks": text_chunks,
        "tables": tables,
        "num_images": num_images,
        "image_captions": image_captions,
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
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import os
import re

def ask_ai_for_chart_plan_from_db(db_path, query):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    schema = {}

    for table in tables:
        cursor.execute(f"PRAGMA table_info('{table}')")
        columns = [col[1] for col in cursor.fetchall()]
        cursor.execute(f"SELECT * FROM '{table}' LIMIT 5")
        rows = cursor.fetchall()
        schema[table] = {
            "columns": columns,
            "sample": [dict(zip(columns, row)) for row in rows]
        }

    prompt = f"""
You are a data analysis assistant.

A user asks: "{query}"

Here is the schema and sample data from the SQLite database:
{json.dumps(schema, indent=2)}

Your task is to return a JSON object with:
- x_column: (string) the x-axis column
- y_column: (string) the y-axis column (can be null for pie charts)
- chart_type: (string) "bar", "line", or "pie"
- aggregation: (string) "mean", "sum", or "count"
- explanation: (string) why you chose these columns
"""

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)

def generate_chart(query):
    try:
        conn = sqlite3.connect(DB_PATH)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)['name'].tolist()

        for table in tables:
            df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)

            plan = ask_ai_for_chart_plan_from_db(DB_PATH, query)
            print("🧠 AI plan:", plan)


            # Normalize AI column names to match df.columns
            x = plan["x_column"]
            y = plan.get("y_column")
            

            chart_type = plan["chart_type"]
            agg_func = plan["aggregation"]
            explanation = plan.get("explanation", "")
            df[y] = pd.to_numeric(df[y], errors="coerce")

                # Drop missing values after coercion



            plt.figure(figsize=(10, 6))

            if chart_type == "pie":
                if y:
                    grouped = df.groupby(x)[y].agg(agg_func).sort_values(ascending=False).head(10)
                else:
                    grouped = df[x].value_counts().head(10)
                grouped.plot(kind="pie", autopct='%1.1f%%', startangle=90)
                plt.ylabel("")
            else:
                grouped = df.groupby(x)[y].agg(agg_func).sort_values(ascending=False).head(20)
                plot_args = {"kind": chart_type, "color": "skyblue", "linewidth": 2}
                if chart_type == "line":
                    plot_args["marker"] = "o"
                grouped.plot(**plot_args)
                plt.xlabel(x.replace("_", " ").title())
                plt.ylabel(f"{agg_func.title()} of {y.replace('_', ' ').title()}" if y else "")
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, linestyle="--", alpha=0.5)

            title = f"{chart_type.title()} Chart of {y} by {x}" if y else f"{chart_type.title()} Chart of {x}"
            plt.title(title)
            plt.tight_layout()

            chart_path = os.path.join(OUTPUT_DIR, f"{table}_{x}_vs_{y or 'count'}_{chart_type}.png")
            plt.savefig(chart_path)
            plt.show()
            plt.close()

            return f"📊 {explanation}\nChart saved to: {chart_path}"

        return "⚠️ No tables found in the database."
    except Exception as e:
        return f"Chart Generation Error: {e}"
# === Load image captions from file ===
def load_image_captions():
    captions_path = os.path.join(OUTPUT_DIR, "image_captions.txt")
    if not os.path.exists(captions_path):
        return "No image captions were found."
    with open(captions_path, "r", encoding="utf-8") as f:
        return f.read()

def image_caption_tool_fn(query):
    captions = load_image_captions()
    return f"""Use the following image captions to answer the question.

Image Captions:
{captions}

Question: {query}
"""

# === Setup full LangChain agent ===
def setup_agent(qa_chain):
    doc_qa_tool = Tool(
        name="PDF_QA_Tool",
        func=lambda q: qa_chain.invoke({"query": q})["result"] if qa_chain else "No QA retriever available.",
        description=(
            "Use this to answer questions about the uploaded PDF. "
            "It searches the full document and uses GPT-4 to reason across multiple sections."
        )
    )

    sql_tool = Tool(
        name="SQL_Tool",
        func=run_sql_query,
        description=(
            "Use this tool to query structured tabular data (from PDF tables or CSV) using SQLite syntax."
        )
    )

    image_tool = Tool(
        name="Image_Caption_Tool",
        func=image_caption_tool_fn,
        description=(
            "Use this to answer questions about images or charts in the PDF. "
            "It uses BLIP-generated captions. Ask things like 'What does the chart on page 5 show?'."
        )
    )

    chart_tool = Tool(
    name="Chart_Generator_Tool",
    func=generate_chart,
    description=(
        "Use this tool to generate visual charts (bar, line, pie) from CSV or PDF tables. "
        "The tool automatically infers the best x and y columns from the query. "
        "Ask things like: 'Show average order value by month', 'Line chart of profit by year', or 'Pie chart of category distribution'."
        )
    )


    llm = ChatOpenAI(temperature=0, model="gpt-4")

    return initialize_agent(
        tools=[doc_qa_tool, sql_tool, image_tool, chart_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
# === Example Run ===
if __name__ == "__main__":
    input_path = input("📄 Enter path to PDF or CSV file: ").strip()

    if input_path.lower().endswith(".pdf"):
        parse_pdf(input_path)
        qa_chain = setup_retriever(input_path)
    elif input_path.lower().endswith(".csv"):
        load_csv_to_db(input_path, DB_PATH)
        qa_chain = None  # No retriever needed for CSV-only mode
    else:
        print("❌ Unsupported file type. Please provide a .pdf or .csv file.")
        exit()

    agent = setup_agent(qa_chain)

    while True:
        query = input("\n💬 Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        # 🔍 Basic keyword trigger to directly call the chart generator
        if "chart" in query.lower() or "graph" in query.lower() or "plot" in query.lower():
            print("\n📊 Generating chart...")
            response = generate_chart(query)
        else:
            response = agent.run(query)

        print("\n🧠 Response:", response)
