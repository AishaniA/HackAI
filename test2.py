from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

# === Set your OpenAI key ===
os.environ["OPENAI_API_KEY"] = ""  # Replace with your key

# === Load and parse the PDF ===
pdf_path = "/Users/succhaygadhar/Downloads/ltimindtree_annual_report.pdf"
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages from PDF")

# === Split the text into chunks ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
print(f"🧩 Split into {len(chunks)} text chunks")

# === Embed and create a vector store ===
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("/Users/succhaygadhar/Downloads/ltimindtree_index")
print("📦 Saved vector store to 'ltimindtree_index/'")

# === Load retriever for QA ===
retriever = vectorstore.as_retriever()

# === Setup the retrieval-augmented QA chain ===
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=retriever,
    return_source_documents=True
)

# === Example query ===
while True:
    query = input("\n💬 Ask a question about the annual report (or type 'exit'): ")
    if query.lower() == "exit":
        break

    result = qa_chain(query)
    print("\n🧠 Answer:")
    print(result['result'])

    print("\n📄 Source pages:")
    for doc in result['source_documents']:
        print(f"Page {doc.metadata.get('page', 'N/A')} → {doc.page_content[:200]}...\n")
