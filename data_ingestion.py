from datasets import load_dataset
from model import EmbeddingModel
from chroma_client import ChromaClient
import numpy as np

dataset = load_dataset("MohammadOthman/mo-customer-support-tweets-945k", split="train")


def storeEmbeddings():

    chroma_client = ChromaClient()
    embedding_model = EmbeddingModel()

    batchSize = 100
    all_ids, all_embeddings, all_metadatas = [], [], []

    for i, data in enumerate(dataset):
        try:
            input_text = data["input"]
            output_text = data["output"]
            uniqueId = f"ID-{i}"

            embeddings = embedding_model.get_embeddings(input_text)

            metaData = {"input": input_text, "output": output_text}

            all_ids.append(uniqueId)
            all_embeddings.append(embeddings)
            all_metadatas.append(metaData)

            # print(all_ids)
            # print(all_embeddings)
            # print(all_metadatas)

            if i == 3:
                break

            if (i + 1) % batchSize == 0:
                chroma_client.store_embeddings(all_ids, all_embeddings, all_metadatas)
                all_ids, all_embeddings, all_metadatas = [], [], []
        except Exception as e:
            print(f"Error processing index {i}: {e}")
            continue

    if all_ids:
        try:
            chroma_client.store_embeddings(all_ids, all_embeddings, all_metadatas)
        except Exception as e:
            print(f"Error storing remaining embeddings: {e}")

    print("✅ Dataset stored in ChromaDB!")
