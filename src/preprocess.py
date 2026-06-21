import pandas as pd
import re
import os

def clean_narrative(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'x{2,}', '', text)  # Strips standard CFPB text masks (like XXXX)
    text = re.sub(r'[^a-z0-9\s.,!?\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_streaming_preprocessing(input_path, output_path, chunksize=50000):
    print("🚀 Initializing high-speed streaming processing for large dataset...")
    
    # Target products mapping matching project specifications
    target_products = {'Credit card', 'Personal loan', 'Savings account', 'Money transfer'}
    
    header_written = False
    total_processed_rows = 0
    total_saved_rows = 0
    
    if not os.path.exists(input_path):
        print(f"❌ Error: File not found at {input_path}. Place your 6GB file there and name it complaints.csv")
        return

    # Remove old filtered output if it exists to prevent appending duplicates
    if os.path.exists(output_path):
        os.remove(output_path)

    # Process massive file in chunks of 50,000 rows
    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        total_processed_rows += len(chunk)
        print(f"🔄 Scanning raw row chunk: {total_processed_rows:,}...")
        
        # Look for standard narrative column name variants
        narrative_col = 'consumer_complaint_narrative' if 'consumer_complaint_narrative' in chunk.columns else 'Consumer complaint narrative'
        product_col = 'product' if 'product' in chunk.columns else 'Product'
        
        if narrative_col not in chunk.columns or product_col not in chunk.columns:
            continue

        chunk = chunk.dropna(subset=[narrative_col])
        chunk['product_clean'] = chunk[product_col].str.strip().str.capitalize()
        filtered_chunk = chunk[chunk['product_clean'].isin(target_products)].copy()
        
        if filtered_chunk.empty:
            continue
            
        filtered_chunk['cleaned_narrative'] = filtered_chunk[narrative_col].apply(clean_narrative)
        filtered_chunk = filtered_chunk[filtered_chunk['cleaned_narrative'].str.len() > 10]
        
        if filtered_chunk.empty:
            continue
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        filtered_chunk.to_csv(output_path, mode='a', index=False, header=not header_written)
        header_written = True
        total_saved_rows += len(filtered_chunk)

    print(f"✨ Finished processing! Saved {total_saved_rows:,} filtered target rows to: {output_path}")

if __name__ == "__main__":
    run_streaming_preprocessing("data/raw/complaints.csv", "data/processed/filtered_complaints.csv")