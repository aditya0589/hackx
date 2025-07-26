import re
from typing import List

def split_text(text, chunk_size=1000, overlap=200, strategy='words'):
    """
    Splits text into chunks with overlap using different strategies.
    
    Args:
        text: Input text to split
        chunk_size: Target size for each chunk (words or characters)
        overlap: Overlap between consecutive chunks
        strategy: 'words' for word-based splitting, 'sentences' for sentence-based, 'paragraphs' for paragraph-based
    
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    text = text.strip()
    
    if strategy == 'words':
        return _split_by_words(text, chunk_size, overlap)
    elif strategy == 'sentences':
        return _split_by_sentences(text, chunk_size, overlap)
    elif strategy == 'paragraphs':
        return _split_by_paragraphs(text, chunk_size, overlap)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'words', 'sentences', or 'paragraphs'")

def _split_by_words(text, chunk_size, overlap):
    """
    Split text by words with overlap.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks

def _split_by_sentences(text, chunk_size, overlap):
    """
    Split text by sentences, trying to keep chunks around chunk_size words.
    """
    # Split by sentence endings (., !, ?) followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_size = len(sentence_words)
        
        # If adding this sentence would exceed chunk_size and we have content
        if current_size + sentence_size > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            overlap_words = []
            overlap_size = 0
            for word in reversed(current_chunk):
                if overlap_size + 1 <= overlap:
                    overlap_words.insert(0, word)
                    overlap_size += 1
                else:
                    break
            
            current_chunk = overlap_words
            current_size = overlap_size
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def _split_by_paragraphs(text, chunk_size, overlap):
    """
    Split text by paragraphs, trying to keep chunks around chunk_size words.
    """
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        paragraph_words = paragraph.split()
        paragraph_size = len(paragraph_words)
        
        # If adding this paragraph would exceed chunk_size and we have content
        if current_size + paragraph_size > chunk_size and current_chunk:
            # Save current chunk
            chunks.append('\n\n'.join(current_chunk))
            
            # Start new chunk with overlap
            overlap_paragraphs = []
            overlap_size = 0
            for para in reversed(current_chunk):
                para_words = para.split()
                if overlap_size + len(para_words) <= overlap:
                    overlap_paragraphs.insert(0, para)
                    overlap_size += len(para_words)
                else:
                    break
            
            current_chunk = overlap_paragraphs
            current_size = overlap_size
        
        current_chunk.append(paragraph)
        current_size += paragraph_size
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def optimize_chunk_size(text_length, target_chunks=50):
    """
    Automatically determine optimal chunk size based on text length and target number of chunks.
    
    Args:
        text_length: Length of the text in characters
        target_chunks: Target number of chunks
    
    Returns:
        Optimal chunk size in characters
    """
    if target_chunks <= 0:
        return 1000  # Default chunk size
    
    optimal_size = max(500, text_length // target_chunks)
    return min(optimal_size, 2000)  # Cap at 2000 characters 