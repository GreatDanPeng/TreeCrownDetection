<p align="center">
    <img src="https://github.com/user-attachments/assets/8a0e4dcc-9ae6-4d02-8fd3-46dc3dd5ccbe" alt="Tree segmentation model performance" width="800">
</p>
<p align="center"><b>Fig1:</b> Models performance on segmenting tree crowns in dense forest</p>

<p align="center">
<img width="800" alt="sparse" src="https://github.com/user-attachments/assets/e6d20a5c-3693-4814-b27e-4b2c445f5fdc" />
</p>
<p align="center"><b>Fig2:</b> Models performance on segmenting tree crowns in sparse forest</p>


# TreeCrownDetection

Paper Link: https://www.overleaf.com/read/kvpkfqdcbnxh#8a9469

Existing tree-crown detection models that use deep neutral networks suffer from poor accuracy and extremely limited trainable benchmark dataset. There are growing applications of deep neutral networks in the delineation of tree crowns, but a systematic comparison of the performance of state-of-the-art models has not been made so far. Thus, we implemented YOLOv11 for tree crown detection, U-Net and SAM2 (Segment Anything 2) with CNN baseline for tree crown segmentation. With no available training set with bounding box and mask labels, we first manually annotated over 500 images with over 5,000 labels by ourselves. Secondly, we attempted pseudo-masking by bounding box labels from human annotations and YOLOv11-m predictions for tree crown segmentation. Further more, we leveraged zero-shot learning from pretrained SAM2, which can predict tree crown segmentation by inferenece. Our results demonstrated that YOLOv11-m outperforms other models in tree crown detection tasks and U-Net trained on human-annotated mask labels performs best in all benchmark metrics. SAM2 demonstrated the potential of better tree segmentation but is limited by user prompts (bounding box labels or points). 

