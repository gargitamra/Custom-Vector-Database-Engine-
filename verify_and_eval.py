import time
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from brute_force_index import BruteForceIndex
from ivf_flat_index import IVFFlatIndex

print("==================================================")
print("       VECTOR DATABASE VERIFICATION & EVAL        ")
print("==================================================\n")

# 1. Load Data & Models
print("[1/4] Loading texts, vectors, and embedding model...")
with open("texts.json", "r") as f:
    texts = json.load(f)

vectors = np.load("vectors.npy")
query_vectors = np.load("test_vectors.npy")
if len(vectors) < 50000:
    raise ValueError("vectors.npy must contain at least 50,000 database vectors")
if len(query_vectors) < 500:
    raise ValueError("test_vectors.npy must contain at least 500 query vectors")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize both indices
bf_index = BruteForceIndex("vectors.npy")
ivf_index = IVFFlatIndex("vectors.npy", num_clusters=100, n_probe=5)

print("\n--------------------------------------------------")
# 2. Test API Methods (Insert & Delete)
print("[2/4] Testing API Operations (Insert & Delete)...")

test_text = "SpaceX successfully launches starship into orbit."
test_vec = model.encode(test_text)

# Test Insert
new_id = bf_index.insert(test_vec)
ivf_new_id = ivf_index.insert(test_vec)
texts.append(test_text)  # Sync local texts array

print(f"  ✓ Inserted new vector at Index ID: {new_id}")

# Search for the newly inserted text
search_indices, _ = bf_index.search(test_vec, k=1)
assert search_indices[0] == new_id, "Insert test failed on BruteForce!"
print("  ✓ Search successfully retrieved inserted vector.")

# Test Delete
bf_index.delete(new_id)
ivf_index.delete(ivf_new_id)
search_indices_after_del, _ = bf_index.search(test_vec, k=1)
assert search_indices_after_del[0] != new_id, "Delete test failed! Tombstone vector still returned."
print("  ✓ Delete successfully hid vector using tombstone flag.")

print("\n--------------------------------------------------")
# 3. Test Semantic Query Matching
print("[3/4] Testing Semantic Quality...")
sample_query = "Recent updates on computer technology and software."
sample_vec = model.encode(sample_query)

bf_res, bf_scores = bf_index.search(sample_vec, k=3)
ivf_res, ivf_scores = ivf_index.search(sample_vec, k=3)

print(f"Query: '{sample_query}'\n")
print("Brute-Force Top Result:")
print(f"  [ID {bf_res[0]} | Score {bf_scores[0]:.4f}]: {texts[bf_res[0]][:90]}...")
print("IVF-Flat Top Result:")
print(f"  [ID {ivf_res[0]} | Score {ivf_scores[0]:.4f}]: {texts[ivf_res[0]][:90]}...\n")

print("\n--------------------------------------------------")
# 4. Benchmark Evaluation over the prepared 500-query set
NUM_QUERIES = 500
print(f"[4/4] Running {NUM_QUERIES}-Query Benchmark (Recall@10 & Latency)...")
query_vectors = query_vectors[:NUM_QUERIES]

bf_times = []
ivf_times = []
recalls = []

for q_vec in query_vectors:
    # Measure Brute Force
    t0 = time.perf_counter()
    bf_ids, _ = bf_index.search(q_vec, k=10)
    t1 = time.perf_counter()
    bf_times.append((t1 - t0) * 1000)  # Convert to ms

    # Measure IVF-Flat
    t0 = time.perf_counter()
    ivf_ids, _ = ivf_index.search(q_vec, k=10)
    t1 = time.perf_counter()
    ivf_times.append((t1 - t0) * 1000) # Convert to ms

    # Calculate Recall@10: Fraction of exact ground truth found by IVF
    intersection = set(bf_ids).intersection(ivf_ids)
    recall_at_10 = len(intersection) / len(bf_ids) if bf_ids else 1.0
    recalls.append(recall_at_10)

avg_bf_latency = np.mean(bf_times)
avg_ivf_latency = np.mean(ivf_times)
avg_recall = np.mean(recalls) * 100
speedup = avg_bf_latency / avg_ivf_latency if avg_ivf_latency > 0 else 0

print("\n==================================================")
print("               BENCHMARK RESULTS                  ")
print("==================================================")
print(f" Total Queries Tested : {NUM_QUERIES}")
print(f" Brute-Force Latency  : {avg_bf_latency:.3f} ms / query")
print(f" IVF-Flat Latency     : {avg_ivf_latency:.3f} ms / query")
print(f" Search Speedup       : {speedup:.2f}x faster")
print(f" Average Recall@10    : {avg_recall:.2f}%")
print("==================================================\n")