#############################################################
########   please use official docker conainer     ##########
#############################################################

import torch
from tqdm import tqdm
import jsonlines
from pathlib import Path
from mPLUG_zero_shot_infer import mPLUG_captioning

# path for the images
IMG_FILE_PATH = "/root/images"

def generate_captions_and_save(img_directory: str, batch_size: int, output_file: str):
    """
    Generate captions for images in the directory using BLIP2_captioning and save the results in a jsonl file.

    Args:
    - img_directory (str): Path to the directory containing images.
    - batch_size (int): Number of images to process at once.
    - output_file (str): Path to save the jsonl file.
    """

    # Collect all image paths in the directory
    img_paths = [str(p) for p in Path(img_directory).rglob("*.jpg")]
    
    # Open the output file in write mode
    with jsonlines.open(output_file, mode='w') as writer:
        # Process images in batches
        for i in tqdm(range(0, len(img_paths), batch_size)):
            batch = img_paths[i: i + batch_size]

            # Generate captions using mPLUG
            captions = mPLUG_captioning(batch)

            # Organize results and write to jsonl file
            for path, caption in zip(batch, captions):
                result = {
                    "file_name": Path(path).name,
                    "caption": caption
                }
                writer.write(result)

if __name__ == "__main__":
    generate_captions_and_save(IMG_FILE_PATH, batch_size=64, output_file="captions_mPLUG_base_coco.jsonl")