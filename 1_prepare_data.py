import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATABASE_SIZE = 50000
QUERY_SIZE = 500

print("--- Step 1: Loading local CSV datasets ---")
df_train = pd.read_csv("ag_news/train.csv", header=None)
if len(df_train) < DATABASE_SIZE:
    raise ValueError(f"train.csv must contain at least {DATABASE_SIZE} rows")
train_texts = [(str(row[1]) + ". " + str(row[2]))[:150] for row in df_train.iloc[:DATABASE_SIZE].values]

df_test = pd.read_csv("ag_news/test.csv", header=None)
if len(df_test) < QUERY_SIZE:
    raise ValueError(f"test.csv must contain at least {QUERY_SIZE} rows")
test_texts = [(str(row[1]) + ". " + str(row[2]))[:150] for row in df_test.iloc[:QUERY_SIZE].values]

# Save text listings
with open("texts.json", "w") as f:
    json.dump(train_texts, f)
with open("test_texts.json", "w") as f:
    json.dump(test_texts, f)

print(f"Saved {len(train_texts)} database texts to texts.json")
print(f"Saved {len(test_texts)} test query texts to test_texts.json")


print("\n--- Step 2: Loading embedding model ---")
model = SentenceTransformer('all-MiniLM-L6-v2')
model.max_seq_length = 64  # Optimizes sequence length for CPU encoding speed


print(f"\n--- Step 3: Encoding {len(train_texts)} Database Vectors ---")
raw_train = model.encode(train_texts, batch_size=256, show_progress_bar=True)
norm_train = raw_train / np.linalg.norm(raw_train, axis=1, keepdims=True)
np.save("vectors.npy", norm_train)
print("Saved database vectors to vectors.npy")


print(f"\n--- Step 4: Encoding {len(test_texts)} Test Query Vectors ---")
raw_test = model.encode(test_texts, batch_size=256, show_progress_bar=True)
norm_test = raw_test / np.linalg.norm(raw_test, axis=1, keepdims=True)
np.save("test_vectors.npy", norm_test)
print("Saved test query vectors to test_vectors.npy")

print("\nData preparation complete!")