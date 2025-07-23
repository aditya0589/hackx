def split_text(text, chunk_size=500, overlap=50):
    """
    Splits text into chunks of chunk_size with overlap.
    Returns a list of text chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks 