import os
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ---------------- CONFIG ----------------
DATA_DIR = "data"
CHROMA_DIR = os.path.join(os.getcwd(), "chroma_db")


# Load FREE local LLM once
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)

# ---------------- STEP 2: LOAD & CHUNK ----------------
def load_documents(data_dir):
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                documents.append({
                    "source": filename,
                    "content": file.read().strip()
                })
    return documents


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# ---------------- STEP 3: VECTOR STORE ----------------
def create_vector_store(chunks):
    client = chromadb.Client(
        chromadb.config.Settings(persist_directory=CHROMA_DIR)
    )

    collection = client.get_or_create_collection(name="policy_docs")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [c["chunk"] for c in chunks]
    metadatas = [{"source": c["source"]} for c in chunks]
    ids = [str(i) for i in range(len(chunks))]
    embeddings = model.encode(texts).tolist()

    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )

    print("✅ Vector store created and embeddings added.")


# ---------------- STEP 4: RETRIEVAL ----------------
def retrieve_relevant_chunks(query, top_k=3):
    client = chromadb.Client(
        chromadb.config.Settings(persist_directory=CHROMA_DIR)
    )

    collection = client.get_collection(name="policy_docs")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved = []
    for i in range(len(results["documents"][0])):
        retrieved.append({
            "content": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"]
        })

    return retrieved


# ---------------- GOLD STANDARD SAFETY CHECK ----------------
def is_answer_present(retrieved_chunks, question):
    stopwords = {
        "is", "are", "was", "were", "the", "a", "an", "there",
        "what", "does", "do", "available", "offer", "provide",
        "can", "we", "you", "i", "to", "of", "for", "in", "on"
    }

    keywords = [
        word for word in question.lower().split()
        if word not in stopwords
    ]

    for chunk in retrieved_chunks:
        content = chunk["content"].lower()
        for word in keywords:
            if word in content:
                return True

    return False


# ---------------- STEP 5: PROMPT ENGINEERING ----------------
def combine_context(retrieved_chunks):
    context = ""
    for chunk in retrieved_chunks:
        context += f"\n[Source: {chunk['source']}]\n{chunk['content']}\n"
    return context


def build_prompt(context, question):
    return f"""
You are a policy question-answering assistant.

Rules:
- Use ONLY the information from the context below
- Do NOT use outside knowledge
- If the answer is missing or unclear, say so explicitly
- Mention the source policy name

Context:
{context}

Question:
{question}

Answer format:
- Answer:
- Source Policy:
"""


def generate_answer(prompt):
    response = llm(prompt)
    return response[0]["generated_text"]


if __name__ == "__main__":

    # -------- INDEXING PHASE --------
    docs = load_documents(DATA_DIR)

    all_chunks = []
    for doc in docs:
        for chunk in chunk_text(doc["content"]):
            all_chunks.append({
                "source": doc["source"],
                "chunk": chunk
            })

    print(f"Total chunks created: {len(all_chunks)}")
    create_vector_store(all_chunks)

    # -------- EVALUATION QUESTIONS --------
    test_questions = [
        "What is the refund timeline?",                 # Answerable
        "Can an order be cancelled after shipping?",    # Partially answerable
        "Is there a student discount available?"        # Unanswerable
    ]

    # -------- QUERY PHASE --------
    for question in test_questions:
        print("\n" + "=" * 60)
        print("Question:", question)

        retrieved = retrieve_relevant_chunks(question)

        # GOLD STANDARD REFUSAL
        if not is_answer_present(retrieved, question):
            print("\nAnswer:")
            print("This information is not available in the provided policy documents.")
            print("\nSource Policy:\nN/A")
        else:
            context = combine_context(retrieved)
            prompt = build_prompt(context, question)
            answer = generate_answer(prompt)

            print("\nAnswer:\n", answer)

