import pdfplumber
import pandas as pd

PDF_PATH = "/Users/succhaygadhar/Downloads/Untitled document (3).pdf"
CSV_OUTPUT = "/Users/succhaygadhar/Downloads/output1/tables/plumber_table_page1.csv"  # Update if needed

with pdfplumber.open(PDF_PATH) as pdf:
    page = pdf.pages[0]

    # Extract the table using text-based alignment strategies
    table = page.extract_table({
        "vertical_strategy": "text", 
        "horizontal_strategy": "text"
    })

    if table:
        df = pd.DataFrame(table[1:], columns=table[0])  # First row = header
        df.to_csv(CSV_OUTPUT, index=False)
        print(f"✅ Table saved to: {CSV_OUTPUT}")
    else:
        print("⚠️ No table found on page 1.")
