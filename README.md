# EN3160 Assignment 3: Neural Networks

This repository contains the notebook and supporting data for Assignment 3 of EN3160, Image Processing and Machine Vision. The assignment compares a manually trained dense network, CNNs for MNIST, and two ResNet-18 transfer-learning approaches for the Hymenoptera ants-and-bees dataset.

**Student:** R.A.C.D. Ranawaka  
**Index number:** 230526A

## Project contents

- `A03.ipynb` - assignment code, plots, confusion matrices, training logs, and discussions. Run this notebook from top to bottom.
- `data/` - datasets used by the notebook. CIFAR-10 and MNIST are loaded by `torchvision`; the notebook downloads the Hymenoptera archive if the extracted dataset is not available locally.
- `230526A_a03.html` and `230526A_a03.pdf` - notebook exports. The notebook is the authoritative version for the latest code and results.

## Assignment work

### Question 1: Dense CIFAR-10 classifier

The original manually calculated forward and backward paths are extended with a 100-unit sigmoid hidden layer and cross-entropy loss. The network is trained for 10 epochs, and its loss history and final training and test accuracy are reported.

### Question 2: MNIST CNNs

LeNet-5 and a three-convolution CNN with batch normalization and dropout are trained for 10 epochs. The notebook records per-epoch training and test loss and accuracy, plots the histories, and shows a test-set confusion matrix for each model.

### Question 3: ResNet-18 transfer learning

An ImageNet-1K pretrained ResNet-18 is applied to the Hymenoptera dataset in two ways:

1. **Fine tuning:** train the full network, including its replacement classifier.
2. **Feature extraction:** freeze the pretrained backbone and train only the replacement classifier.

The notebook reports per-epoch training and validation metrics and compares the best validation accuracies. The tutorial dataset provides a validation split, so these are validation results rather than test-set results.

## Results

These are the results saved in the notebook. Accuracy may vary slightly if the models are trained again.

| Question | Model | Epochs | Training accuracy | Evaluation accuracy |
| --- | --- | ---: | ---: | ---: |
| Q1 | Dense network, CIFAR-10 | 10 | 50.52% | 47.38% test |
| Q2(a) | LeNet-5, MNIST | 10 | 99.44% | 98.82% test |
| Q2(b) | Three-layer CNN, MNIST | 10 | 97.97% | 98.99% test |
| Q3(a) | ResNet-18 fine tuning | 10 | 92.21%* | 94.77% best validation |
| Q3(b) | ResNet-18 feature extraction | 10 | 94.26%* | 96.08% best validation |

For Q2, the final test losses were 0.0395 for LeNet-5 and 0.0321 for the three-layer CNN. For Q3, the validation split contains 153 images: fine tuning classified 145 correctly at its best epoch, and feature extraction classified 147 correctly. This small difference should be interpreted cautiously.

\*Q3 training accuracy is the value logged in epoch 10 while the model was in training mode with random image augmentation. It is not directly comparable to validation accuracy.

## Requirements

- Python 3
- Jupyter Notebook
- PyTorch and TorchVision
- NumPy
- Matplotlib

Install the packages in an active Python environment:

```bash
python -m pip install torch torchvision numpy matplotlib notebook
```

For GPU training, install a PyTorch build compatible with the computer's CUDA setup by following the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). The notebook automatically uses a CUDA device when available and otherwise runs on CPU.

## Run the notebook

From the project directory, start Jupyter:

```bash
jupyter notebook A03.ipynb
```

Then run the cells in order. CIFAR-10 and MNIST are loaded under `./data`. Question 3 uses this folder layout:

```text
data/
└── hymenoptera_data/
    ├── train/
    │   ├── ants/
    │   └── bees/
    └── val/
        ├── ants/
        └── bees/
```

If the Hymenoptera dataset is missing, the notebook attempts to download and extract it. Internet access is also needed to download the pretrained ResNet-18 weights if they are not already cached locally. If downloading the dataset fails, download `hymenoptera_data.zip` manually, place it in `data/`, and rerun the Q3 setup cell.

## Notes

- The Q2 training accuracy is measured during training. For the CNN, dropout is active during those training batches, while it is disabled for test evaluation; the two figures therefore are not directly comparable.
- Q3 uses separate train and validation splits. Its validation accuracy is not a test-set result.
- Training times depend on the available hardware. The saved run used CPU for the MNIST training, as reported in the notebook.
