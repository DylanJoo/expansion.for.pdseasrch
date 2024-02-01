from encoder import ContrieverEncoder
import jsonlines
from tqdm import tqdm
import argparse

# Function to count total number of lines in the file
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def encode_titles(corpus_file: str, output_file: str, batch_size: int = 32):
    # Initialize the encoder
    encoder = ContrieverEncoder()

    # Function to process and encode a batch of titles
    def process_batch(batch):
        # Extract titles from the batch
        titles = [obj['title'] for obj in batch]

        # Encode using the Contriever model
        embeddings = encoder.encode(titles)

        # Convert embeddings to list for JSON serialization
        embeddings = embeddings.tolist()

        return [(batch[i]['doc_id'], embeddings[i]) for i in range(len(batch))]

    with jsonlines.open(corpus_file) as reader, jsonlines.open(output_file, mode='w') as writer:
        batch = []
        for obj in tqdm(reader, total=file_len(corpus_file), desc="Encoding titles"):
            batch.append(obj)

            # When the batch is full, process it
            if len(batch) == batch_size:
                for doc_id, embedding in process_batch(batch):
                    writer.write({
                        "id": doc_id,
                        "embeddings": embedding
                    })
                batch = []  # Reset the batch

        # Process the remaining batch (if any)
        if batch:
            for doc_id, embedding in process_batch(batch):
                writer.write({
                    "id": doc_id,
                    "embeddings": embedding
                })

def main():
    parser = argparse.ArgumentParser(description='Encode titles from a JSONL corpus file.')

    # Define the arguments
    parser.add_argument('-i', '--input_path', type=str, required=True, help='Input corpus file path')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Output file path for embeddings')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for encoding (default: 32)')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with arguments from the command line
    encode_titles(args.input_path, args.output_path, args.batch_size)

if __name__ == "__main__":
    main()
