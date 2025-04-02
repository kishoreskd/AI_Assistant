import chromadb


class ChromaClient:
    def __init__(self, path="./chromo_db"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name="code_functions")

    def store_embeddings(self, uniqueId, embedding, metaData):
        print(f"Storing embedding for ID: {uniqueId}")
        print(f"Embedding: {embedding}")
        self.collection.add(ids=uniqueId, embeddings=embedding, metadatas=metaData)
        print(f"Stored embedding...")

    def retrive_data(self):
        print("Retrieving data from ChromaDB...")
        return self.collection.get()

    def query_embedding(self, embedding, top_k=5):
        return self.collection.query(query_embeddings=embedding, n_results=top_k)
