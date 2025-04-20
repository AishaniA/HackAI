# FinSyte
FinSyte is an AI-powered tool that helps users easily understand complex documents by answering questions, analyzing tables, and generating insights directly from uploaded PDFs or CSV files.

## Inspiration
We’ve spent countless hours navigating long and cluttered reports. FinSyte was created to remove that friction. By using agentic AI, we give users a faster and easier way to get the information they need, without wasting time or effort.

## What it does
- Supports both PDF and CSV files
- Extracts and analyzes tables and charts
- Answers natural-language questions using only the uploaded file
- Provides a confidence score with each answer

## How we built it
We used a combination of tools and frameworks:
- PyMuPDF for extracting text and tables
- LangChain with OpenAI’s GPT-4 for question-answering
- FAISS for vector-based document retrieval
- Streamlit for the web interface
- BLIP for generating captions from images and graphs

## How to run it
1. Download project and open root project directory
2. Go to backend.py and change output directory to local dessired location
3. Substitute OPEN AI API Key in for place holder
4. In terminal under project root folder run, " streamlit run app.py "

## Challenges we ran into
Reading tables from PDFs was especially difficult due to inconsistent layouts. Many tools didn’t return usable results, which cost us a lot of time. We also had limitations with styling the frontend since Streamlit is lightweight and simple by design.

## Accomplishments we're proud of
We successfully extracted tables using PyMuPDF after a lot of trial and error. We also connected LangChain’s retrieval pipeline to GPT-4, allowing for accurate and dynamic answers from any uploaded file.

## What we learned
We learned how to use LangChain to create structured workflows for AI, and how agentic AI goes beyond simple prompt-based tools. We also discovered the value of combining multiple tools—language models, image models, vector search—to solve real-world problems.

## What's next for FinSyte
We found that many document tools still struggle with inconsistent layouts, embedded visuals, and complex tables. With further work, FinSyte could become a practical tool for students, companies, and researchers who need insights from dense reports.  

We plan to expand support for other formats like Excel and HTML, and continue improving FinSyte’s ability to interpret and summarize all kinds of structured and unstructured data.


