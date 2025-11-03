import os
import yaml
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyPDF2 import PdfReader
from PIL import Image
from typing import List
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ========== OCR HELPERS ==========
def try_init_deepseek_pipeline(ocr_model_name):
    try:
        from transformers import pipeline
        logging.info("Initializing DeepSeek OCR pipeline...")
        ocr = pipeline("image-to-text", model=ocr_model_name, trust_remote_code=True)
        return ocr
    except Exception as e:
        logging.warning(f"DeepSeek init failed: {e}")
        return None


def ocr_with_pytesseract(pil_image):
    import pytesseract
    return pytesseract.image_to_string(pil_image)


def ocr_with_easyocr(pil_image, reader=None):
    import easyocr, numpy as np
    if reader is None:
        reader = easyocr.Reader(['en'], gpu=False)
    arr = np.array(pil_image)
    res = reader.readtext(arr, detail=0)
    return "\n".join(res)


# ========== TEXT EXTRACTION ==========
def extract_text_any(path, hf_pipeline=None, easyocr_reader=None):
    ext = os.path.splitext(path)[1].lower()

    # Handle PDFs
    if ext == ".pdf":
        try:
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if text.strip():
                logging.info(f"Extracted text from PDF: {path}")
                return text.strip()

            logging.warning(f"No text in {path}, running OCR...")
            from pdf2image import convert_from_path
            pages = convert_from_path(path, dpi=200)
            if pages:
                return ocr_with_pytesseract(pages[0])
        except Exception as e:
            logging.error(f"PDF extraction failed: {e}")
        return ""

    # Handle image files
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        logging.warning(f"Failed to open image {path}: {e}")
        return ""

    # Try OCR methods in order
    for method_name, method in [
        ("DeepSeek", lambda: hf_pipeline(img) if hf_pipeline else None),
        ("pytesseract", lambda: ocr_with_pytesseract(img)),
        ("easyocr", lambda: ocr_with_easyocr(img, reader=easyocr_reader))
    ]:
        try:
            result = method()
            if result:
                if isinstance(result, list) and 'generated_text' in result[0]:
                    return result[0]['generated_text']
                return str(result)
        except Exception as e:
            logging.warning(f"{method_name} failed: {e}")
    logging.error(f"All OCR methods failed for {path}")
    return ""


# ========== UTILS ==========
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def chunk_text(text: str, max_chars=1000, overlap=100) -> List[str]:
    return [
        text[i:i + max_chars]
        for i in range(0, len(text), max_chars - overlap)
    ]


# ========== MAIN PIPELINE ==========
def process_file(path, embedder, q, coll, hf_pipeline, easyocr_reader, chunk_chars, overlap, point_id_start):
    fname = os.path.basename(path)
    text = extract_text_any(path, hf_pipeline=hf_pipeline, easyocr_reader=easyocr_reader)
    if not text:
        logging.warning(f"No text extracted from {fname}, skipping.")
        return 0

    chunks = chunk_text(text, chunk_chars, overlap)
    embeddings = embedder.encode(chunks, batch_size=8, show_progress_bar=False)
    points = [
        qdrant_models.PointStruct(
            id=point_id_start + i,
            vector=embeddings[i].tolist(),
            payload={"source_file": fname, "chunk_index": i, "text": chunks[i][:5000]}
        )
        for i in range(len(chunks))
    ]
    q.upsert(collection_name=coll, points=points)
    logging.info(f"Ingested {len(chunks)} chunks from {fname}")
    return len(chunks)


def main():
    cfg = load_config()
    qcfg, coll = cfg["qdrant"], cfg["qdrant"]["collection"]
    embed_model_name, ocr_model_name = cfg["models"]["embedding"], cfg["models"]["ocr"]
    chunk_chars, overlap = cfg["ingest"]["chunk_chars"], cfg["ingest"]["chunk_overlap"]
    docs_dir = cfg["docs_dir"]["docs_to_ingest"]

    # Initialize models
    logging.info(f"Loading embedding model: {embed_model_name}")
    embedder = SentenceTransformer(embed_model_name)
    q = QdrantClient(url=qcfg["url"], api_key=qcfg.get("api_key"))
    hf_pipeline = try_init_deepseek_pipeline(ocr_model_name)
    import easyocr
    easyocr_reader = easyocr.Reader(['en'], gpu=False)

    # Create collection if not exists
    vector_size = embedder.get_sentence_embedding_dimension()
    try:
        q.get_collection(coll)
    except Exception:
        q.recreate_collection(
            collection_name=coll,
            vectors_config=qdrant_models.VectorParams(size=vector_size, distance=qdrant_models.Distance.COSINE)
        )
        logging.info(f"Created collection '{coll}'")

    # Process files concurrently
    paths = [os.path.join(docs_dir, f) for f in os.listdir(docs_dir) if os.path.isfile(os.path.join(docs_dir, f))]
    os.makedirs(docs_dir, exist_ok=True)
    point_id = 0

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                process_file, path, embedder, q, coll, hf_pipeline, easyocr_reader, chunk_chars, overlap, point_id + idx * 10000
            ): path
            for idx, path in enumerate(paths)
        }

        for future in as_completed(futures):
            path = futures[future]
            try:
                future.result()
            except Exception as e:
                logging.error(f"Failed to process {path}: {e}")

    logging.info("✅ Ingestion complete.")


if __name__ == "__main__":
    main()
