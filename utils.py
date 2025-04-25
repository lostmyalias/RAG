from typing import List
from pathlib import Path
from config import CHUNK_SIZE, OVERLAP

def chunk_text(text: str) -> List[str]:
    """Split text into overlapping chunks of words."""
    words, chunks = text.split(), []
    step = CHUNK_SIZE - OVERLAP
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i : i + CHUNK_SIZE]))
    return chunks

def get_txt_files(folder: Path) -> List[Path]:
    """Recursively find all .txt files in a directory."""
    return sorted(folder.rglob("*.txt"))