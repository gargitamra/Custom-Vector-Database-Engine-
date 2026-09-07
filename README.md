# Custom Vector Database Engine

A lightweight, high-performance vector search engine implemented entirely from scratch in Python using pure NumPy arithmetic. This project demonstrates how modern vector databases execute exact ground-truth search (Brute-Force) and approximate search (IVF-Flat) over a dataset of 50,000 text embeddings.

---

## ⚙️ Execution & Mocking Scope

* **How to Run:** Follow the step-by-step commands in the [🚀 Getting Started](#-getting-started) section below.
* **What is Mocked:** **Nothing is mocked.** The engine runs 100% live locally on real AG News dataset articles, real 384-dimensional embeddings generated via `all-MiniLM-L6-v2`, and custom pure-NumPy vector math. Third-party vector database libraries (FAISS, Pinecone, Chroma, scikit-learn) are strictly excluded.

---

## 🚀 Getting Started

### 1. Installation & Environment Setup
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/gargitamra/Custom-Vector-Database-Engine-.git
cd Custom-Vector-Database-Engine-
pip install numpy pandas sentence-transformers streamlit tqdm