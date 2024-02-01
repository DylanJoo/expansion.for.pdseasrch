import pandas as pd
from encoder import ContrieverEncoder
from tqdm import tqdm
import json
import argparse
    

def encode_query(query_file: str, output_file: str, batch_size: int = 32):
    
    def process_batch(batch: list):
        qids = [row['qid'] for row in batch]
        queries = [row['query'] for row in batch]

        # Encode queries using the encoder
        embeddings = encoder.encode(queries)

        # Return a list of tuples containing qid and embedding
        return zip(qids, embeddings)
    
    query_df = pd.read_csv(query_file, delimiter='\t', header=None, names=['qid', 'query'])
    encoder = ContrieverEncoder()

    with open(output_file, 'w') as writer:  # Open the JSONL file for writing
        batch = []
        for _, row in tqdm(query_df.iterrows(), total=query_df.shape[0], desc="Encoding titles"):
            batch.append(row)

            if len(batch) == batch_size:
                for doc_id, embedding in process_batch(batch):
                    # Convert embedding to list if it's numpy array or similar
                    embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else embedding
                    writer.write(json.dumps({"qid": doc_id, "embeddings": embedding_list}) + '\n')
                batch = []

        if batch:
            for doc_id, embedding in process_batch(batch):
                embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else embedding
                writer.write(json.dumps({"qid": doc_id, "embeddings": embedding_list}) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Encode queries from a TSV query file.')

    # Define the arguments
    parser.add_argument('-i', '--input_path', type=str, required=True, help='Input query file path')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Output file path for embeddings')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for encoding (default: 32)')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with arguments from the command line
    encode_query(args.input_path, args.output_path, args.batch_size)

if __name__ == "__main__":
    main()
