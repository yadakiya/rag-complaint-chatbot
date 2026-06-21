import re
from typing import List

class ComplaintChunker:
    @staticmethod
    def chunk_by_character(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """
        Splits text by character size with a specified sliding overlap.
        Great for standard fixed-size window allocations.
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += (chunk_size - chunk_overlap)
        return chunks

    @staticmethod
    def chunk_by_sentence(text: str, max_sentences_per_chunk: int = 3) -> List[str]:
        """
        Groups text cleanly by sentence structures to keep contextual points intact.
        Safe fallback that prevents chopping vital financial sentences in half.
        """
        if not text:
            return []
            
        # Split across typical punctuation markers
        sentences = re.split(r'(?<=[\.\?\!])\s+', text)
        chunks = []
        
        for i in range(0, len(sentences), max_sentences_per_chunk):
            chunk = " ".join(sentences[i:i + max_sentences_per_chunk]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks