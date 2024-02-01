import argparse
from encoder import ClipEncoder
import os
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import jsonlines

CLIP = ClipEncoder()
IMAGE_FOLDER = "/home/jhju/datasets/pdsearch/images"

def load_image(img_dir: str) -> Image:
    img_path = os.path.join(IMAGE_FOLDER, img_dir)
    
    try:
        # Open the image and convert it to RGB
        img = Image.open(img_path).convert('RGB')
        # Resize the image to the target size
        img = img.resize((224, 224))
        return img
    except UnidentifiedImageError:
        # Creating a black image as a placeholder
        return Image.new('RGB', (224, 224))

def load_images_concurrently(image_list: list) -> list:
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(load_image, image_list))
    return images

def CLIP_encoding(image_list: list) -> torch.Tensor:
    image_embeddings = CLIP.vision_encode(image_list)
    
    return image_embeddings

def encode_and_store(folder_name: str, batch_size: int, output_file: str) -> None:
    files = [f for f in tqdm(os.listdir(folder_name), desc="Reading files")]
    
    # Iterate over files in batches and encode them
    with jsonlines.open(output_file, mode='w') as writer:
        for i in tqdm(range(0, len(files), batch_size), desc="Encoding images"):
            # Get the current batch of files
            batch = files[i:i + batch_size]
            numbers = [f.split('.')[0] for f in batch]
            
            images = load_images_concurrently(batch)
            
            image_embeddings = CLIP_encoding(images)
            
            for num, embed in zip(numbers, image_embeddings):
                data = {
                    "id": num,
                    "embeddings": embed.cpu().detach().numpy().tolist()
                }
                writer.write(data)

def main():
    parser = argparse.ArgumentParser(description='Encode images using CLIP')
    parser.add_argument('-f', '--folder_name', type=str, help='Path to the images folder')
    parser.add_argument('-o', '--output_file', type=str, help='Path to the output file')
    parser.add_argument('-b', '--batch_size', type=int, help='Number of images to process in each batch')
    
    args = parser.parse_args()
    
    encode_and_store(args.folder_name, args.batch_size, args.output_file)

if __name__ == "__main__":
    main()
