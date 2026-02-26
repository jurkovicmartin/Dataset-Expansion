# Dataset-Expansion
Repository to extent dataset size for my thesis, which is used for anomaly detection within welds using neural networks.

### Dataset Characteristics
- CT scans of welds - sliced
- Few slices are masked
- Only anomalous samples

## Methods

### Inpainting
Create synthetic normal samples.

Slices with small anomalies are inpainted using opencv.

### Using Optical Flow
Optical flow takes advantage of sequential order of slices extracted from 3D CT views.

The flow (~transformation/transition) is calculated from a GT (Ground Truth) masked image and unmasked (unlabeled) image. It is then applied to the GT mask resulting in a propagated mask. This mask corresponds to the unlabeled image.

This process is effective only with slices that are near to each other = there are no significant changes. This means that these images as well as these masks are very similar, but not identical => new data for the dataset.

## Results