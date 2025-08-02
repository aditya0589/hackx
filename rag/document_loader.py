import os
import requests
from PyPDF2 import PdfReader
from docx import Document
from urllib.parse import urlparse, unquote

# Utility to download file from a URL (cloud link)
def download_file(url, dest_path):
    response = requests.get(url)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        f.write(response.content)
    return dest_path

# Extract text from PDF
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = "\n".join(page.extract_text() or '' for page in reader.pages)
    return text

# Extract text from DOCX
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Main loader function
def load_document(link):
    """
    Accepts a local file path or a URL to a PDF/DOCX document.
    Returns extracted text and the local file path.
    """
    if link.startswith('http://') or link.startswith('https://'):
        parsed = urlparse(link)
        filename = os.path.basename(parsed.path)
        filename = unquote(filename)

        os.makedirs('tmp', exist_ok=True)
        local_path = os.path.join('tmp', filename)
        download_file(link, local_path)
    else:
        local_path = link

    ext = os.path.splitext(local_path)[1].lower()
    print(f"[DEBUG] Resolved file path: {local_path}, extension: {ext}")

    if ext == '.pdf':
        text = extract_text_from_pdf(local_path)
    elif ext == '.docx':
        text = extract_text_from_docx(local_path)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Only PDF and DOCX are supported. "
            f"Received file: {local_path}"
        )
    return text, local_path
