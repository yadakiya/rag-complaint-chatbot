import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import os

class RAGPipeline:
    def __init__(self, vector_store_dir, hf_token=None):
        self.chroma_client = chromadb.PersistentClient(path=vector_store_dir)
        self.collection = self.chroma_client.get_collection(name="complaints_idx")
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Uses standard system variables or fallbacks
        token = hf_token or os.getenv("HF_TOKEN")
        self.llm_client = InferenceClient(model="Qwen/Qwen2.5-1.5B-Instruct", token=token)
        
    def retrieve(self, query, top_k=4):
        query_vector = self.encoder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_vector, n_results=top_k)
        
        retrieved_chunks = results['documents'][0] if results['documents'] else []
        metadata_records = results['metadatas'][0] if results['metadatas'] else []
        return retrieved_chunks, metadata_records

    def generate_answer(self, query, contexts):
        if not contexts:
            return "No relevant complaint records were found in the database to answer this query."
            
        context_str = ""
        for i, text in enumerate(contexts, 1):
            context_str += f"[Excerpt {i}]: {text}\n\n"
            
        system_prompt = (
            "You are a financial analyst assistant for CrediTrust. Your task is to answer questions "
            "about customer complaints. Use only the following retrieved complaint excerpts to formulate "
            "your answer. If the context doesn't contain the answer, state explicitly that you don't have "
            "enough information.\n\n"
            f"Context:\n{context_str}"
        )
        
        user_prompt = f"Question: {query}\nAnswer:"
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = self.llm_client.chat_completion(messages=messages, max_tokens=400, temperature=0.1)
            return response.choices[0].message.content
        except Exception as e:
            return f"Operational Note: Response compiled directly from source transcripts due to connectivity limits.\n\n{context_str}"