
import os
import shutil
import random

def split_dataset(base_path, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    # Paths
    image_dir = os.path.join(base_path)
    train_dir = os.path.join(base_path, 'train')
    valid_dir = os.path.join(base_path, 'valid')
    test_dir = os.path.join(base_path, 'test')
    
    # Create output directories
    for dir_path in [train_dir, valid_dir, test_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Gather all images and xmls
    files = os.listdir(image_dir)
    image_list = sorted([f for f in files if f.endswith('.jpg')])
    xml_list = sorted([f for f in files if f.endswith('.xml')])
    print(f"Found {len(image_list)} images and {len(xml_list)} XML files.")

    # Pair images and XMLs
    paired_files = [(img, img.replace('.jpg', '.xml')) for img in image_list if img.replace('.jpg', '.xml') in xml_list]
    print(f"Found {len(paired_files)} paired files.")

    # Shuffle and split dataset
    random.shuffle(paired_files)
    total_count = len(paired_files)
    train_count = int(total_count * train_ratio)
    valid_count = int(total_count * valid_ratio)

    train_files = paired_files[:train_count]
    valid_files = paired_files[train_count:train_count + valid_count]
    test_files = paired_files[train_count + valid_count:]

    # Helper function to copy files
    def copy_files(file_pairs, dest_dir):
        for img_file, xml_file in file_pairs:
            shutil.copy(os.path.join(image_dir, img_file), os.path.join(dest_dir, img_file))
            shutil.copy(os.path.join(image_dir, xml_file), os.path.join(dest_dir, xml_file))

    # Copy files to respective folders
    copy_files(train_files, train_dir)
    copy_files(valid_files, valid_dir)
    copy_files(test_files, test_dir)

    print(f"Dataset split completed. Train: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")

# Run the function
base_path = '/Users/danpengair/TreeCrownDetection/static/NeonTree900'
split_dataset(base_path)