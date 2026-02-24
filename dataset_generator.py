import os
import shutil
import pandas as pd
import argparse

def process_covid_images(data_root):
    print("Starting Task 1: Processing COVID images...")
    
    metadata_path = os.path.join(data_root, 'covid-chestxray-dataset', 'metadata.csv')
    dest_img_dir = os.path.join(data_root, 'COVID_images')
    txt_output_path = os.path.join(data_root, 'COVID.txt')
    
    os.makedirs(dest_img_dir, exist_ok=True)
    
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    df = pd.read_csv(metadata_path)
    
    # Filter rows where 'finding' contains 'COVID' (case-insensitive)
    covid_df = df[df['finding'].str.contains('COVID', case=False, na=False)]
    
    with open(txt_output_path, 'w') as f_out:
        for filename in covid_df['filename']:
            # Write the relative path to COVID.txt
            f_out.write(f"covid-chestxray-dataset/images/{filename}\n")
                
    print(f"Task 1 Complete. Check '{txt_output_path}' and '{dest_img_dir}'.\n")

def process_healthy_images(data_root):
    print("Starting Task 2: Processing healthy images...")
    
    txt_files = ['train_1.txt', 'val_1.txt', 'test_1.txt']
    source_base_dir = os.path.join(data_root, 'ChestX-ray14')
    dest_base_dir = os.path.join(data_root, 'healthy_images')
    txt_output_path = os.path.join(data_root, 'healthy.txt')
    
    with open(txt_output_path, 'w') as f_out:
        for txt_file in txt_files:
            txt_file_path = os.path.join(source_base_dir, txt_file)
            if not os.path.exists(txt_file_path):
                print(f"Warning: Text file {txt_file} not found. Skipping...")
                continue
                
            with open(txt_file_path, 'r') as f_in:
                for line in f_in:
                    parts = line.strip().split()
                    
                    if not parts:
                        continue
                        
                    filepath = parts[0]  # e.g., 'images_010/00023313_001.png'
                    labels = parts[1:]   # e.g., ['0', '0', '0', ...]
                    
                    # Check if ALL labels are strictly '0'
                    if all(label == '0' for label in labels):
                        
                        # Write to healthy.txt
                        f_out.write(f"ChestX-ray14/{filepath}\n")

    print(f"Task 2 Complete. Check '{txt_output_path}' and '{dest_base_dir}'.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset splits and aggregate images for DinoX-Ray.")
    parser.add_argument('--data_root', type=str, required=True, help='Absolute path to the parent directory containing the raw datasets.')
    args = parser.parse_args()
    
    process_covid_images(args.data_root)
    process_healthy_images(args.data_root)
    print("All processing finished successfully.")