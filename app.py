import time
import json
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from brute_force_index import BruteForceIndex
from ivf_flat_index import IVFFlatIndex

# Page Config
st.set_page_config(
    page_title="Custom Vector Database",
    # page_icon="⚡",
    layout="wide"
)

# Cache model and index loading so it doesn't re-cluster on every click
@st.cache_resource
def load_database():
    with open("texts.json", "r") as f:
        texts = json.load(f)
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Initialize indices
    bf_index = BruteForceIndex("vectors.npy")
    ivf_index = IVFFlatIndex("vectors.npy", num_clusters=100, n_probe=5)
    
    return texts, model, bf_index, ivf_index

# Load Resources
texts, model, bf_index, ivf_index = load_database()

# Header
st.title("Custom Vector Database Engine")
st.caption("Brute-Force vs. IVF-Flat Approx Search")

# Sidebar Controls
st.sidebar.header("⚙️ Database Controls")
st.sidebar.metric("Total DB Vectors", f"{len(bf_index.vectors):,}")
st.sidebar.metric("Embedding Dimensions", "384")

st.sidebar.subheader("IVF Parameters")
n_probe = st.sidebar.slider("n_probe (Clusters to Search)", min_value=1, max_value=20, value=5)
ivf_index.n_probe = n_probe  # Update probe depth dynamically

st.sidebar.info("💡 **Tip:** Higher `n_probe` increases Recall accuracy but takes slightly more latency time.")

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Search & Compare", "➕ Insert Vector", "❌ Delete Vector"])

# ==========================================
# TAB 1: SEARCH & COMPARE
# ==========================================
with tab1:
    st.subheader("Semantic Search Engine")
    
    col_input, col_k = st.columns([4, 1])
    with col_input:
        query_text = st.text_input(
            "Enter a search query:",
            value="Breaking news about artificial intelligence and space exploration"
        )
    with col_k:
        top_k = st.number_input("Top-K", min_value=1, max_value=20, value=5)
        
    if st.button("🔎 Execute Search", type="primary"):
        # 1. Embed query
        t0 = time.perf_counter()
        query_vec = model.encode(query_text)
        embed_time = (time.perf_counter() - t0) * 1000
        
        # 2. Search Brute-Force
        t0 = time.perf_counter()
        bf_ids, bf_scores = bf_index.search(query_vec, k=top_k)
        bf_time = (time.perf_counter() - t0) * 1000
        
        # 3. Search IVF-Flat
        t0 = time.perf_counter()
        ivf_ids, ivf_scores = ivf_index.search(query_vec, k=top_k)
        ivf_time = (time.perf_counter() - t0) * 1000
        
        # Calculate Overlap Recall
        overlap = len(set(bf_ids).intersection(ivf_ids))
        recall = (overlap / len(bf_ids)) * 100 if bf_ids else 100.0
        speedup = bf_time / ivf_time if ivf_time > 0 else 1.0

        # Performance Banner Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Brute-Force Latency", f"{bf_time:.2f} ms")
        m2.metric("IVF-Flat Latency", f"{ivf_time:.2f} ms")
        m3.metric("IVF Speedup", f"{speedup:.1f}x Faster")
        m4.metric("Recall Accuracy", f"{recall:.0f}% Match")

        st.divider()

        # Side-by-side comparison columns
        col_bf, col_ivf = st.columns(2)
        
        with col_bf:
            st.markdown("### Brute-Force Results (Ground Truth)")
            for rank, (idx, score) in enumerate(zip(bf_ids, bf_scores), 1):
                st.success(f"**#{rank} [Score: {score:.4f}] — ID: {idx}**\n\n{texts[idx]}")
                
        with col_ivf:
            st.markdown("### IVF-Flat Results (Approximate)")
            for rank, (idx, score) in enumerate(zip(ivf_ids, ivf_scores), 1):
                # Highlight in blue if exact match with ground truth
                is_match = idx in bf_ids
                badge = "Ground Truth Match" if is_match else "⚠️ Approximate Difference"
                st.info(f"**#{rank} [Score: {score:.4f}] — ID: {idx}** ({badge})\n\n{texts[idx]}")

# ==========================================
# TAB 2: INSERT VECTOR
# ==========================================
with tab2:
    st.subheader("Add New Document to Database")
    new_text = st.text_area("Enter text to embed and insert:", value="Quantum computing breakthrough enables faster cryptography.")
    
    if st.button("➕ Insert into Both Indices"):
        if new_text.strip():
            # 1. Embed on the fly
            new_vec = model.encode(new_text)
            
            # 2. Insert into indices
            new_bf_id = bf_index.insert(new_vec)
            new_ivf_id = ivf_index.insert(new_vec)
            
            # 3. Append to local text registry
            texts.append(new_text)
            
            st.success(f"Successfully inserted! Assigned Vector ID: **{new_bf_id}**")
            st.json({
                "assigned_id": new_bf_id,
                "text_snippet": new_text,
                "vector_dimensions": len(new_vec)
            })

# ==========================================
# TAB 3: DELETE VECTOR
# ==========================================
with tab3:
    st.subheader("Delete Document (Tombstone Strategy)")
    delete_id = st.number_input("Enter Vector ID to delete:", min_value=0, max_value=len(texts) - 1, value=0, step=1)
    
    if st.button("❌ Mark as Deleted"):
        bf_success = bf_index.delete(delete_id)
        ivf_success = ivf_index.delete(delete_id)
        
        if bf_success and ivf_success:
            st.warning(f"Vector ID **{delete_id}** marked as deleted using tombstone masking!")
            st.code(f"Deleted Text at ID {delete_id}:\n'{texts[delete_id]}'")
            st.info("Subsequent searches will ignore this vector score automatically.")