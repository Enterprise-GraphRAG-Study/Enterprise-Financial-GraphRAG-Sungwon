# 🚀 Enterprise-Financial-GraphRAG (Core-Sungwon)

## 📌 Project Overview

This repository is the core development hub for an **Enterprise-grade GraphRAG system** specialized in **Financial Intelligence**. By leveraging **Ontology-based Knowledge Graphs (Neo4j)** and **Multi-Agent Orchestration (LangGraph)**, this project aims to solve the limitations of traditional RAG in complex financial reasoning, such as analyzing SEC 10-K filings and cross-company risk dependencies.

## 💻 High-Performance Development Environment

To simulate and handle enterprise-scale datasets, this project is developed on a high-spec infrastructure optimized for AI research.

### **Hardware Specification**

* **Processor:** Apple M4 Max (GPU Accelerated via Metal Performance Shaders)
* **Memory:** **128GB Unified Memory**
* **Storage:** Ultra-fast NVMe SSD

### **The 128GB Advantage**

The **128GB Unified Memory** architecture provides a significant edge in building GraphRAG:

* **Massive In-Memory Processing:** Ability to load and process entire financial ontologies and large-scale vector indices without OOM errors.
* **Local LLM Excellence:** Capability to run high-parameter models (e.g., Llama-3-70B) locally, ensuring data privacy for sensitive financial documents.
* **High-Throughput Embeddings:** Drastic reduction in latency for large batch processing by eliminating CPU-GPU data transfer bottlenecks.

---

## 📈 Performance Benchmark (Week 1)

| Task | Target Device | Metric | Result |
| :--- | :--- | :--- | :--- |
| **Tensor Embedding** | Apple M4 Max (MPS) | Latency | **~1007.34 ms** (Cold Start) |
| **Data Throughput** | Unified Memory | Rate | **1.46 MB/s** |
| **Dataset Scale** | Sujet Financial RAG | Samples | 98,590 rows |

### **Execution Proof**
![M4 Max Week 1 Benchmark Result](<img width="306" height="65" alt="Image" src="https://github.com/user-attachments/assets/c3cacb74-b305-4dc1-9b07-457e31b2b36d" />)

> **Technical Note:** The latency recorded includes initial MPS kernel warm-up and model loading. In-memory processing of subsequent batches shows near-instantaneous inference.

---

## 🛠️ Tech Stack

* **Frameworks:** PyTorch (MPS Accelerated), Hugging Face Transformers
* **Graph DB:** Neo4j (Planned)
* **Orchestration:** LangGraph (Planned)
* **Monitoring:** Ragas, LangSmith (Planned)

---

## 🚀 Getting Started

### **1. Environment Setup**

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch transformers datasets pandas

```

### **2. Running Week 1 Benchmark**

```bash
python week01_financial_tensor_bench.py

```

---

## 🗺️ Roadmap (12 Weeks Journey)

* [x] **Week 1: PyTorch Foundation & MPS Optimization**
* [ ] **Week 2: Financial Data Preprocessing & Tokenization**
* [ ] **Week 3: Vector Embeddings & Similarity Search**
* [ ] **Week 4: Ontology Design for Financial Entities**
* [ ] **Week 5: Knowledge Graph Construction (Neo4j)**
* [ ] ... (Progressing towards Agentic GraphRAG)

---

## 🧑‍💻 Author

**Seongwon Im**

* Study Leader at **Enterprise-GraphRAG-Study**
* Focus: AI Engineering, Financial Intelligence, Knowledge Graphs

---
