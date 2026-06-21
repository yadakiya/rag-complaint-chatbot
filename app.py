import streamlit as st
from src.rag import RAGPipeline

@st.cache_resource
def init_pipeline():
    return RAGPipeline("vector_store")

st.set_page_config(page_title="CrediTrust Intelligent Complaint System", layout="wide")

st.title("🛡️ CrediTrust Financial Intelligence Agent")
st.subheader("RAG-Powered Customer Feedback & Compliance Auditor Dashboard")
st.markdown("---")

try:
    rag = init_pipeline()
except Exception as e:
    st.error("⚠️ Local Vector Database Index Not Found! Run python src/preprocess.py then python src/embed.py first to create it.")
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🔍 Query Engine")
    user_query = st.text_input(
        "Enter natural-language operational questions:", 
        placeholder="e.g., Why are people unhappy with credit card fees?"
    )
    
    if st.button("Analyze Records", type="primary"):
        if user_query.strip():
            with st.spinner("Scanning semantic indices..."):
                contexts, metadatas = rag.retrieve(user_query, top_k=4)
                answer = rag.generate_answer(user_query, contexts)
                
                st.markdown("#### 🤖 AI-Generated Operational Assessment")
                st.info(answer)
                
                st.session_state['current_contexts'] = contexts
                st.session_state['current_metadata'] = metadatas
        else:
            st.warning("Please enter a query sentence first.")

with col2:
    st.markdown("### 📄 Auditable Evidence Sources")
    if 'current_contexts' in st.session_state:
        for idx, (ctx, meta) in enumerate(zip(st.session_state['current_contexts'], st.session_state['current_metadata'])):
            with st.expander(f"Source Record #{idx+1} (ID: {meta.get('complaint_id', 'N/A')})"):
                st.caption(f"**Vertical Segment:** {meta.get('product_category', 'General')}")
                st.markdown(f"*{ctx}*")
    else:
        st.write("No active query run logged. Context snippets will appear here dynamically.")