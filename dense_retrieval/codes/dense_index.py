import argparse
import json
import numpy as np
from tqdm import tqdm
from pyserini.encode import FaissRepresentationWriter

# Set up argparse
parser = argparse.ArgumentParser(description='Batch-wise indexing of precomputed embeddings from a JSONL file.')
parser.add_argument('-j', '--jsonl_file_path', required=True, help='Path to the JSONL file')
parser.add_argument('-o', '--output_index_path', required=True, help='Path to the output FAISS index directory')
parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size for processing embeddings')
args = parser.parse_args()

# Assuming the embedding dimension is known
embedding_dimension = 768

# Initialize FaissRepresentationWriter for indexing
embedding_writer = FaissRepresentationWriter(args.output_index_path, dimension=embedding_dimension)

def process_batch(batch):
    if batch:  # Check if the batch is not empty
        # Convert list of embeddings to a NumPy array for FAISS
        vectors = np.vstack(batch['vector'])
        # Index the batch of embeddings
        embedding_writer.write({'id': batch['id'], 'vector': vectors})

with open(args.jsonl_file_path, 'r') as jsonl_file, embedding_writer:
    batch = {'id': [], 'vector': []}
    for line in tqdm(jsonl_file, total=1118658, desc="Indexing documents"):
        doc = json.loads(line)
        embedding = np.array(doc['embeddings'], dtype=np.float32)
        batch['id'].append(doc['id'])
        batch['vector'].append(embedding)
        if len(batch['id']) == args.batch_size:
            process_batch(batch)
            batch = {'id': [], 'vector': []}
    process_batch(batch)

print("Indexing complete.")
