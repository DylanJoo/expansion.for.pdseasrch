import os
from tqdm import tqdm
import jsonlines
from BLIP_infer import BLIP_captioning

# path for the images
IMAGE2TEXT = "/tmp2/chiuws/expansion.for.pdseasrch/data/image2text.jsonl"
IMG_PATH_FILE = "/home/jhju/datasets/pdsearch/images"
APPEND_QUERY_COUNT = 10

def generate_captions_and_save(jsonl_file: str, batch_size: int, output_file: str):
    """
    Generate captions for images in the directory using BLIP_captioning and save the results in a jsonl file.

    Args:
    - img_directory (str): Path to the directory containing images.
    - batch_size (int): Number of images to process at once.
    - output_file (str): Path to save the jsonl file.
    """

    # Collect all image paths in the directory
    img_paths = []
    with jsonlines.open(jsonl_file, 'r') as file:
        for line in file:
            # Extract and print the image filename
            image_filename = line.get('image')
            img_paths.append(os.path.join(IMG_PATH_FILE, image_filename))
    
    # Open the output file in write mode
    with jsonlines.open(output_file, mode='w') as writer:
        # Process images in batches
        for i in tqdm(range(0, len(img_paths), batch_size)):
            batch = img_paths[i: i + batch_size]

            # Generate captions BLIP
            captions = BLIP_captioning(batch, return_sequences=APPEND_QUERY_COUNT)
            
            assert len(captions) == APPEND_QUERY_COUNT * len(batch)

            # Organize results and write to jsonl file
            for i, path in enumerate(batch):
                result = {
                    "doc_id": path.split('/')[-1].split('.')[0],
                    "caption": ' '.join(captions[i*APPEND_QUERY_COUNT : (i+1)*APPEND_QUERY_COUNT])
                }
                writer.write(result)


if __name__ == "__main__":
    generate_captions_and_save(IMAGE2TEXT, batch_size=128, output_file="/tmp2/Kai/caption_data/captions_BLIP_base_16k.jsonl")