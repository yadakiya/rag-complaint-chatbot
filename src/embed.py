import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import os

# Pure Python alternative to prevent library version conflicts
class LocalTextSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def split_text(self, text):
        if not isinstance(text, str):
            return []
        chunks = []
        start = 0
        text_len = len(text)
        
        if text_len <= self.chunk_size:
            return [text]
            
        while start < text_len:
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += (self.chunk_size - self.chunk_overlap)
            if start >= text_len or (self.chunk_size - self.chunk_overlap) <= 0:
                break
        return chunks

def generate_vector_store(processed_csv_path, vector_store_dir, sample_size=12000):
    print("📊 Extracting sample from filtered records...")
    
    if not os.path.exists(processed_csv_path):
        print(f"❌ Error: Filtered dataset not found at {processed_csv_path}. Run preprocess.py first.")
        return

    # Stream a subset of rows to stay safely within memory limits
    chunks = []
    for chunk in pd.read_csv(processed_csv_path, chunksize=5000):
        # Dynamically normalize any uppercase column variations to lowercase
        chunk.columns = [c.lower() for c in chunk.columns]
        if 'product_clean' in chunk.columns and 'product' not in chunk.columns:
            chunk = chunk.rename(columns={'product_clean': 'product'})
        chunks.append(chunk)
        if sum(len(c) for c in chunks) >= sample_size * 2:
            break
            
    full_pool = pd.concat(chunks, ignore_index=True)
    
    # Stratify using lowercase normalized column
    sample_df = full_pool.groupby('product', group_keys=False).apply(
        lambda x: x.sample(min(len(x), sample_size // 4), random_state=42)
    )
    
    print(f"🎯 Created stratified subset of {len(sample_df):,} entries for embedding.")
    
    splitter = LocalTextSplitter(chunk_size=500, chunk_overlap=50)
    
    print("📥 Initializing local embedding processor...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    chroma_client = chromadb.PersistentClient(path=vector_store_dir)
    collection = chroma_client.get_or_create_collection(name="complaints_idx")
    
    documents, metadatas, ids = [], [], []
    counter = 0
    
    for _, row in sample_df.iterrows():
        # Safeguard text fields extraction
        narrative = row.get('cleaned_narrative', row.get('consumer_complaint_narrative', ''))
        chunks = splitter.split_text(str(narrative))
        
        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            documents.append(chunk)
            metadatas.append({
                "complaint_id": str(row.get('complaint_id', counter)),
                "product_category": str(row.get('product', 'General')),
                "issue": str(row.get('issue', 'General'))
            })
            ids.append(f"id_{counter}_{idx}")
        counter += 1

    print(f"🧠 Vectorizing {len(documents):,} text fragments... (This takes a moment)")
    embeddings = encoder.encode(documents, batch_size=64, show_progress_bar=True).tolist()
    
    print("📥 Saving structural indexes into Chroma DB disk storage...")
    batch_size = 2000
    for i in range(0, len(documents), batch_size):
        end = i + batch_size
        collection.add(
            embeddings=embeddings[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )
    print("✅ Local Vector DB build complete!")

if __name__ == "__main__":
    generate_vector_store("data/processed/filtered_complaints.csv", "vector_store")