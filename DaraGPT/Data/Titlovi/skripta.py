import re
import glob
import os

# Mapa za popravku lošeg encodinga (č, ć, đ, š, ž)
REPLACEMENTS = {
    "æ": "ć", "è": "č", "ê": "š", "ð": "đ", "ò": "š", "ø": "ž", "ý": "ž", "û": "đ",
    "ã": "ć", "ñ": "ń", "œ": "đ", "Å¡": "š", "Å¾": "ž", "Ä‡": "ć", "Ä�": "đ",
    "Ä�": "č", "Ã³": "ó", "Ã¨": "č", "Ã¦": "ć", "Ã": "č", "Â": "",
}

def clean_text(text: str) -> str:
    """Uklanja oznake, HTML tagove, višak praznina i popravlja slova."""
    # Ukloni redne brojeve i vremenske oznake
    text = re.sub(r'\d+\s*\n\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}', '', text)

    # Ukloni HTML tagove (npr. <i>, <b>, <font> itd.)
    text = re.sub(r'<[^>]+>', '', text)

    # Zameni loše enkodirane karaktere
    for bad, good in REPLACEMENTS.items():
        text = text.replace(bad, good)

    # Spoji sve linije u jedan tok rečenica
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = ' '.join(lines)

    # Očisti višestruke razmake
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_srt_file(input_path: str, output_path: str):
    """Čisti pojedinačni .srt fajl."""
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()

    cleaned = clean_text(raw)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)

    print(f"Očišćen fajl: {os.path.basename(output_path)}")


def batch_clean_srt(folder_pattern="./SRT/*.srt"):
    """Pronađe sve .srt fajlove i obradi ih redom."""
    files = glob.glob(folder_pattern)
    if not files:
        print("Nema .srt fajlova u datom direktorijumu.")
        return

    os.makedirs("./TXT", exist_ok=True) 

    print(f"Pronađeno {len(files)} fajlova za čišćenje...\n")

    for path in files:
        base_name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join("./TXT", f"{base_name}_clean.txt")
        clean_srt_file(path, out_path)

    print("\nSvi fajlovi su uspešno očišćeni! Rezultati su u ./TXT/")


if __name__ == "__main__":
    batch_clean_srt("./SRT/*.srt")
