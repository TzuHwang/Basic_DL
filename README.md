# Basic_DL
Wellcome your visit!
This repo was made for simple Deep-learning application for new comers.

## Requirements
### Hardware
* An nvidia GPU with at least 8Gb memory.
### Software 
1. Python==3.8.13
2. Install the pytorch with cudaversion corresponding to your GPU from https://pytorch.org/ or https://pytorch.org/get-started/previous-versions/.
    `pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113`
3. Install the rest of packages in requirement.txt
    `pip install -r requirements.txt`

## Test Dataset Download
1. fish dataset: https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset

## sh Scripts
All shell scripts shall be stored in the 'configs' folder. Copy and modify the script following the description in the arguments for your own use.

1. Run classification task demo:
    `bash configs\fishcls.sh`
2. Run segmentation task demo:
    `bash configs\fishseg.sh`

## Architecture
### arguments
Parse the arguments you put in the sh script. Ensure that each argument in the script has a corresponding entry in the __init__.py file.

### dataset
The folder stores the data loaders. For adding new data loaders, design a data loaders to fit a similar format with the already existing ones and add new arguments in arguments/__init__.py file. 

### model
* UNet: Convolutional neural network for semantic segmentation with a U-shaped architecture and skip connections
* VGG16: A deep convolutional neural network architecture consisting of 16 weight layers, known for its simplicity and effectiveness in image classification tasks.

### loss
#### Segmentaion Loss:
* SoftDiceLoss: Measures segmentation mask similarity with given groundtruth
* CrossEntropyND: Measures CrossEntropyLoss for images. The input of this loss must be logit.

#### Classification Loss:
* CrossEntropyLoss: A loss for multi class classification. The input of this loss must be logit.
* BCEloss: A loss for binary classification. The input of this loss must be probability.
* CEWithLogitsLoss: A loss for binary classification. The input of this loss must be logit.

### tb_visualizaion

### main

## Future Plane