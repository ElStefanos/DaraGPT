import os
import glob
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    """Ekstraktuje čisti tekst iz PDF fajla."""
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Greška pri čitanju {pdf_path}: {e}")
    return text

def batch_extract_pdfs(pdf_folder="./PDF", out_folder="./TXT"):
    """Pronađe sve PDF fajlove i sačuva njihov tekst u TXT folder."""
    os.makedirs(out_folder, exist_ok=True)
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

    if not pdf_files:
        print("Nema PDF fajlova u direktorijumu:", pdf_folder)
        return

    print(f"Pronađeno {len(pdf_files)} PDF fajlova.\n")

    for pdf_path in pdf_files:
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        txt_path = os.path.join(out_folder, f"{base_name}.txt")

        text = extract_text_from_pdf(pdf_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text.strip())

        print(f"Sačuvan tekst u: {txt_path}")

    print("\nZavršeno! Svi tekstovi su uspešno sačuvani.")

if __name__ == "__main__":
    batch_extract_pdfs()
