import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class OpsMindRAGRetriever:
    """
    Lightweight RAG retriever for OpsMind AI.

    This retriever loads markdown policy documents from knowledge_base/,
    splits them into chunks, and retrieves the most relevant chunks using TF-IDF similarity.
    """

    def __init__(self, knowledge_base_path="knowledge_base"):
        self.knowledge_base_path = knowledge_base_path
        self.documents = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.document_vectors = None

        self._load_documents()
        self._build_index()

    def _load_documents(self):
        if not os.path.exists(self.knowledge_base_path):
            self.documents = []
            return

        for filename in os.listdir(self.knowledge_base_path):
            if filename.endswith(".md"):
                file_path = os.path.join(self.knowledge_base_path, filename)

                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()

                chunks = self._chunk_text(content)

                for index, chunk in enumerate(chunks):
                    self.documents.append({
                        "source": filename,
                        "chunk_id": index + 1,
                        "content": chunk
                    })

    def _chunk_text(self, text, max_words=120):
        words = text.split()
        chunks = []

        for start in range(0, len(words), max_words):
            chunk_words = words[start:start + max_words]
            chunk_text = " ".join(chunk_words)

            if chunk_text.strip():
                chunks.append(chunk_text)

        return chunks

    def _build_index(self):
        if not self.documents:
            self.document_vectors = None
            return

        texts = [doc["content"] for doc in self.documents]
        self.document_vectors = self.vectorizer.fit_transform(texts)

    def retrieve(self, query, top_k=3):
        if not self.documents or self.document_vectors is None:
            return []

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()

        top_indices = similarities.argsort()[::-1][:top_k]

        results = []

        for index in top_indices:
            score = similarities[index]

            if score > 0:
                doc = self.documents[index]
                results.append({
                    "source": doc["source"],
                    "chunk_id": doc["chunk_id"],
                    "score": round(float(score), 4),
                    "content": doc["content"]
                })

        return results

    def format_context(self, retrieved_chunks):
        if not retrieved_chunks:
            return "No relevant policy context was retrieved."

        formatted_context = []

        for chunk in retrieved_chunks:
            formatted_context.append(
                f"Source: {chunk['source']} | Chunk: {chunk['chunk_id']} | Score: {chunk['score']}\n"
                f"{chunk['content']}"
            )

        return "\n\n---\n\n".join(formatted_context)


if __name__ == "__main__":
    retriever = OpsMindRAGRetriever()
    results = retriever.retrieve("What features make an invoice high risk?", top_k=3)

    for item in results:
        print("SOURCE:", item["source"])
        print("SCORE:", item["score"])
        print(item["content"])
        print("-" * 60)