# Policy Document Question Answering using RAG

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system that answers questions based on company policy documents such as Refund Policy, Cancellation Policy, and Shipping Policy.

The system retrieves relevant policy sections and generates **grounded answers**, while explicitly avoiding hallucinations when information is not available. The primary focus of this project is **prompt engineering, retrieval quality, and evaluation**, rather than UI or model complexity.

---

## Architecture

Policy Documents (TXT)
↓
Text Loading & Cleaning
↓
Chunking (500 characters with overlap)
↓
Sentence Embeddings
↓
Chroma Vector Database
↓
Semantic Retrieval (Top-K)
↓
Prompt Engineering
↓
Local Open-Source LLM (FLAN-T5)


---

## Data Preparation
- Sample policy documents were created to simulate a real-world company knowledge base:
  - `refund_policy.txt`
  - `cancellation_policy.txt`
  - `shipping_policy.txt`
- Documents are stored externally in a `data/` directory, mimicking real production document ingestion.

### Chunking Strategy
- **Chunk size:** 500 characters  
- **Overlap:** 50 characters  

**Reasoning:**  
This balances semantic completeness with retrieval precision. Overlap ensures that important context is not lost between adjacent chunks.

---

## RAG Pipeline
1. Policy documents are loaded and chunked.
2. Each chunk is converted into an embedding using a sentence-transformer model.
3. Embeddings are stored in a **Chroma vector database** along with metadata (source policy).
4. For a given question, semantic search retrieves the most relevant chunks.
5. Retrieved context is passed to the language model for answer generation.

---

## Prompt Engineering
Two prompt iterations were designed:

### Prompt Version 1
- Simple instruction to answer only from retrieved context.

### Prompt Version 2 (Improved)
- Explicit rules to:
  - Use only retrieved context
  - Avoid outside knowledge
  - Gracefully handle missing information
  - Cite source policy names

This iteration significantly reduced hallucinations and improved answer clarity.

---

## Hallucination Control (Gold Standard)
Before answer generation, the system checks whether **meaningful keywords (excluding stopwords)** from the question appear in the retrieved context.

- If relevant information is found → generate answer
- If not found → explicitly respond that the information is unavailable

This ensures **safe, honest, and production-ready behavior**.

---

## Evaluation

### Evaluation Method
A fixed set of representative questions was hardcoded in the script to enable **manual and reproducible evaluation**, as allowed by the assignment.

### Evaluation Questions & Results

| Question | Expected Behavior | Result |
|--------|------------------|--------|
| Refund timeline | 5–7 business days | ✅ |
| Digital product refund | Not refundable | ✅ |
| Shipping duration | 3–5 business days | ✅ |
| Cancel after shipping | Partial info | ⚠️ |
| Free shipping | Above ₹999 only | ⚠️ |
| International shipping | Not mentioned | ❌ |
| Student discount | Not mentioned | ❌ |

**Legend:**  
- ✅ Correct and grounded  
- ⚠️ Partially answerable due to limited context  
- ❌ Correct refusal (no hallucination)

---

## Edge Case Handling
- If no relevant documents are retrieved, the system explicitly states that the information is not available.
- If a question is outside the policy knowledge base, the system avoids fabrication and provides a clear refusal.

---

## Technology Stack
- **Language:** Python  
- **Embeddings:** Sentence Transformers  
- **Vector Database:** Chroma  
- **LLM:** FLAN-T5 (open-source, local, cost-free)  

---

## Key Trade-offs
- Used a lightweight open-source LLM to keep the project **free, local, and reproducible**.
- Focused on **retrieval correctness and hallucination control** rather than model size or UI complexity.

---

## Future Improvements
- Add a reranking step for retrieved chunks
- Introduce automated evaluation metrics (faithfulness, relevance)
- Support PDF ingestion and larger document sets

---

## Final Note
This project prioritizes **clarity, reasoning, and hallucination avoidance**, aligning with real-world RAG system design principles and production best practices.
