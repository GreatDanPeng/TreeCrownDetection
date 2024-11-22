import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

class TreeDataset(Dataset):
    def __init__(self, image_dir, xml_dir, transform=None, mask_transform=None, image_size=(512, 512), image_list = None):
        """
        Args:
            image_dir (str): Directory with all the .tif images.
            xml_dir (str): Directory with all the .xml annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            image_size (tuple): Size to resize images and masks to.
        """
        self.image_dir = image_dir
        self.xml_dir = xml_dir
        self.transform = transform
        self.image_size = image_size
        self.mask_transform = mask_transform

        if image_list is not None:
            self.images = {os.path.splitext(f)[0]: f for f in image_list if f.endswith('.jpg')}
        else:
            self.images = {os.path.splitext(f)[0]: f for f in os.listdir(image_dir) if f.endswith('.jpg')}

        if xml_dir is not None:
            self.annotations = {os.path.splitext(f)[0]: f for f in os.listdir(xml_dir) if f.endswith('.xml')}
        else:
            self.annotations = {}

        self.files = list(self.images.keys())

    def get_bounding_boxes(self, xml_file):
        """
        Parse the XML file to extract bounding boxes.
        Args:
            xml_file (str): Path to the XML file.
        Returns:
            List[Tuple[int, int, int, int]]: List of bounding boxes (xmin, ymin, xmax, ymax).
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()

        boxes = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append((xmin, ymin, xmax, ymax))

        return boxes

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_id = self.files[idx]

        # Construct paths for the image and annotation (if available)
        image_path = os.path.join(self.image_dir, self.images[file_id])
        xml_path = os.path.join(self.xml_dir, self.annotations[file_id]) if file_id in self.annotations else None

        # Load the image and apply resizing and transforms
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # Save the original size (width, height)
        image = image.resize(self.image_size, Image.BILINEAR)
        if self.transform:
            image = self.transform(image)

        # Apply mask transform
        if xml_path is not None:
            boxes = self.get_bounding_boxes(xml_path)

        if self.mask_transform:
            mask = self.mask_transform(boxes, original_size)
        else:
            # Default to an empty mask if no transform is given
            mask = torch.zeros((1, *self.image_size), dtype=torch.float32)

        return image, mask

    def show_with_annotations(self, idx):
        """
        Display the image with bounding box annotations (as a binary mask).
        Args:
            idx (int): Index of the image to display.
        """
        # Load image and mask
        data = self[idx]
        if isinstance(data, tuple):
            image, mask = data  # For labeled data, image and mask are returned
        else:
            image, mask = data, None  # For unlabeled data, only the image is returned

        # Convert image tensor to numpy array for visualization
        to_pil_image = transforms.ToPILImage()
        image = to_pil_image(image)  # Convert to PIL image
        image = np.array(image)  # Convert to numpy array for plotting

        # Set up plot
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image)
        ax[0].set_title("Image")
        ax[0].axis('off')

        if mask is not None:
            # Convert mask tensor to numpy array for visualization
            mask = mask.squeeze(0).numpy()  # Remove the channel dimension and convert to numpy

            # Display the mask overlay on the image
            ax[1].imshow(image, alpha=0.8)
            ax[1].imshow(mask, cmap='gray', alpha=0.2)  # Overlay the mask in gray
            ax[1].set_title("Image with Mask")
        else:
            ax[1].imshow(image)
            ax[1].set_title("Image (Unlabeled)")

        ax[1].axis('off')
        plt.show()

print("TreeDataset class defined!")