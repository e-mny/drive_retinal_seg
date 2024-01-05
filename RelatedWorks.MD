# Related Works

Inspired by the following papers:
1. [Full-Resolution Network and Dual-Threshold Iteration for Retinal Vessel and Coronary Angiograph Segmentation](https://doi.org/10.1109/JBHI.2022.3188710) - Liu et al. (IEEE 2022)
2. [RV-GAN: Segmenting Retinal Vascular Structure in Fundus Photographs using a Novel Multi-scale Generative Adversarial Network](https://doi.org/10.48550/arXiv.2101.00535) - Kamran et al. (MICCAI 2021)


---
### Works of each Paper
1. Liu et al. (IEEE 2022)
    - Contributions
        - Full-resolution UNet (FR-UNet) for vessel segmentation
        - Embed feature aggregation module before each convolution block to aggregate feature maps from up-sampling and down-sampling
        - Dual-threshold iterative algorithm to improve the connectivity of vessels that gradually extracts weak vessel pixels from the probability map
        - Introduces a quantitative evaluation method for vessel connectivity by calculating the number of connected components.
    - Parameters
        - Adam Optimizer; Weight Decay of 1e<sup>-5</sup>; 40 Epochs
        - Initial learning rate of 1e<sup>-4</sup> which is gradually reduced by [Cosine Annealing algorithm](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html).
    - 48 × 48 sliding window with a stride of 6 is used to extract patches from vessel images to increase the quantity of **training** data. 
        - Whole images are used for testing data
    - Data Augmentation: Random Horizontal flipping, Vertical flipping, and [90,180,270] degree rotation

2. Kamran et al. (MICCAI 2021)
    - Contributions
        - 2 generators and 2 multi-scale autoencoding discriminators for better microvessel localization and segmentation
        - Introduced a new weighted feature matching loss to avoid fidelity suffered by traditional GAN-based segmentation systems
    - Parameters
        - Adam Optimizer; Batch-size of 24; 100 Epochs
        - Learning rate of 2e<sup>-4</sup>
        - First moment estimate (mean): β1 = 0.5
        - Second moment estimate (uncentered variance): β2 = 0.999
    - 128 x 128 sliding window with stride of 32 was used to extract patches for training and validation data
        - 4,200 training/validation images in total
        - Overlapping image patches with stride of 3 were extracted and averaged from the 20 test images.


--- 
### Experiment Details

#### Performance Metrics
- AUC (Area under Curve)
- Accuracy
- Sensitivity
- Specificity
- F1 Score
- IoU (Intersection of Union)

---
### Results on DRIVE (Digital Retinal Images for Vessel Extraction) dataset
| Papers | Parameters (M) | Accuracy | Sensitivity | Specificity | AUC | F1 Score | IoU |
|--------|---|---|---|------|------|---|------|
| U-Net | 7.76 | 0.9678 | 0.8057 | 0.9833 | 0.9825 | 0.8141 | 0.6864 |
| Paper 1 | 5.72 | 0.9705 | 0.8356 | 0.9837 | 0.9889 | 0.8316 | 0.7120 |
| Paper 2 | 14.81 | 0.9790 | 0.7927 | 0.9969 | 0.9887 | 0.8690 | 0.9762 |
<!-- | Paper 3 | |||||||
| Paper 4 | ||||||| -->
