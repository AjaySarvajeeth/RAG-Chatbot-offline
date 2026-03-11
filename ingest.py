import os, glob, json
from pathlib import Path
from dotenv import load_dotenv
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz  # PyMuPDF
import docx
from bs4 import BeautifulSoup
import pandas as pd
import pytesseract
from PIL import Image
from utils import smart_chunk_text  # make sure this is your chunking function

load_dotenv()

# ---------------- CONFIG ---------------- #
DATA_DIR = Path(os.getenv('DATA_DIR', './data')).expanduser().resolve()
INDEX_DIR = Path(os.getenv('INDEX_DIR', './index')).expanduser().resolve()
EMBEDDING_MODEL = Path(os.getenv("EMBEDDING_MODEL")).expanduser().resolve()
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 800))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 120))
OCR_LANG = os.getenv('OCR_LANG', 'eng')
USE_TABLE_EXTRACTION = os.getenv('USE_TABLE_EXTRACTION', 'true').lower() in ('1','true','yes')

# Offline mode
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# Device
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ---------------- EMBEDDER ---------------- #
embedder = SentenceTransformer(str(EMBEDDING_MODEL), device=DEVICE, trust_remote_code=False)

# ---------------- UTILITIES ---------------- #
def ocr_image_from_pix(pix):
    mode = "RGBA" if pix.alpha else "RGB"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(img, lang=OCR_LANG).strip()

def extract_tables_pdf(path):
    try:
        import pdfplumber
    except Exception:
        print("⚠️ pdfplumber not installed, skipping table extraction.")
        return []
    tables_texts = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables() or []
                for t_idx, table in enumerate(tables):
                    rows = [", ".join([str(c) if c is not None else '' for c in row]) for row in table]
                    txt = "\n".join(rows)
                    if txt.strip():
                        tables_texts.append({'page': i, 'text': txt, 'table_index': t_idx})
            except Exception as e:
                print(f"⚠️ Error extracting table from page {i}: {e}")
                continue
    return tables_texts

# ---------------- READERS ---------------- #
def read_pdf(path):
    out = []
    doc = fitz.open(path)
    for i, page in enumerate(doc, start=1):
        text = (page.get_text() or "").strip()
        if not text:
            try:
                pix = page.get_pixmap()
                text = ocr_image_from_pix(pix)
            except Exception:
                text = ""
        if text.strip():
            out.append((i, text))
    return out

def read_docx(path):
    d = docx.Document(path)
    text = "\n".join([p.text for p in d.paragraphs if p.text.strip()])
    return [(None, text)] if text else []

def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    return [(None, text)] if text else []

def read_csv(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, encoding='latin-1')
    text = df.to_string().strip()
    return [(None, text)] if text else []

def read_json(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    text = json.dumps(data, indent=2, ensure_ascii=False).strip()
    return [(None, text)] if text else []

def read_html_or_md(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n").strip()
    return [(None, text)] if text else []

READERS = {
    '.pdf': read_pdf,
    '.docx': read_docx,
    '.txt': read_txt,
    '.csv': read_csv,
    '.json': read_json,
    '.html': read_html_or_md,
    '.htm': read_html_or_md,
    '.md': read_html_or_md
}

# ---------------- LOAD DOCUMENTS ---------------- #
def load_documents():
    docs = []
    for path in glob.glob(str(DATA_DIR / '*')):
        ext = os.path.splitext(path)[1].lower()
        name = os.path.basename(path)
        if ext not in READERS:
            print(f"Skipping unsupported file: {name}")
            continue
        print(f"📄 Processing {name}...")
        try:
            # Table extraction for PDFs
            if ext == '.pdf' and USE_TABLE_EXTRACTION:
                tables = extract_tables_pdf(path)
                for t in tables:
                    for chunk in smart_chunk_text(t['text'], CHUNK_SIZE, CHUNK_OVERLAP, source=name, lang="en"):
                        if chunk.get('text','').strip():
                            chunk['page'] = t['page']
                            chunk['type'] = 'table'
                            docs.append(chunk)
            # Regular text
            pages = READERS[ext](path)
            for page_num, text in pages:
                for chunk in smart_chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP, source=name, lang="en"):
                    if chunk.get('text','').strip():   # filter out empty chunks
                        chunk['page'] = page_num
                        chunk['type'] = 'text'
                        docs.append(chunk)
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")
    return docs

# ---------------- WRITE FAISS INDEX ---------------- #
def write_index(docs):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    texts = [d['text'] for d in docs if d.get('text','').strip()]
    if not texts:
        print("⚠️ No valid text to index.")
        return
    emb = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.asarray(emb, dtype='float32'))
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "meta.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"✅ Indexed {len(texts)} chunks.")

# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true', help='Rebuild index from scratch')
    args = parser.parse_args()

    docs = load_documents()
    print(f"Loaded {len(docs)} chunks from files.")

    if args.rebuild or not (INDEX_DIR / "faiss.index").exists():
        write_index(docs)
    else:
        # Add new chunks to existing index
        idx_path = INDEX_DIR / "faiss.index"
        idx = faiss.read_index(str(idx_path))
        new_texts = [d['text'] for d in docs if d.get('text','').strip()]
        if new_texts:
            emb_new = embedder.encode(new_texts, convert_to_numpy=True, show_progress_bar=True)
            idx.add(np.asarray(emb_new, dtype='float32'))
            faiss.write_index(idx, str(idx_path))
            with open(INDEX_DIR / "meta.jsonl", "a", encoding='utf-8') as f:
                for d in docs:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"✅ Added {len(new_texts)} new chunks to existing index.")
