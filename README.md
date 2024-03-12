# AI_Framework
Wellcome your visit!
This repo was made for simple Deep-learning application.

## Requirements
### Hardware
* An nvidia GPU with at least 8Gb memory.

### Software 
1. Python==3.8.13
2. Install the pytorch with cudaversion corresponding to your GPU from https://pytorch.org/ or https://pytorch.org/get-started/previous-versions/.
    
```sh
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113`
```
3. Install the rest of packages in requirement.txt

```sh    
pip install -r requirements.txt`
```
## Test Dataset Download
1. fish dataset: https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset
2. dogvscat dataset: https://www.kaggle.com/competitions/dogs-vs-cats/data
3. mutag dataset: https://huggingface.co/datasets/graphs-datasets/MUTAG

## sh Scripts
All shell scripts shall be stored in the 'configs' folder. Copy and modify the script following the description in the arguments for your own use.

### Run training section from scratch:
    
```sh    
bash configs\dogcatcls_v8m_bce.sh

bash configs\dogcatcls_vgg.sh
```  
    
### Eval the performance of trained model:
    
```sh    
bash configs\dogcatcls_v8m_bce_eval.sh
    
bash configs\dogcatcls_vgg_eval.sh
```

### Details
data_root: The folder contains the saved data.
output_root: The folder contains all test info.
config_name: The folder locates under the output_root which contains the log and ckpt.

p.s. Do not toggle on --rm-exist-log if you want to retain the training log.

## Architecture
### arguments
Parse the arguments you put in the sh script. Ensure that each argument in the script has a corresponding entry in the __init__.py file.

### dataset
The folder stores the data loaders. For adding new data loaders, design a data loaders to fit a similar format with the already existing ones and add new arguments in arguments/__init__.py file. 

### model
* UNet: Convolutional neural network for semantic segmentation with a U-shaped architecture and skip connections
* VGG16: A deep convolutional neural network architecture consisting of 16 weight layers, known for its simplicity and effectiveness in image classification tasks.
* YOLOs: An efficient object detection algorithm that processes images in one pass, detecting objects with bounding boxes and classifying them simultaneously.

### loss
#### Segmentaion Loss:
* SoftDiceLoss: Measures segmentation mask similarity with given groundtruth
* CrossEntropyND: Measures CrossEntropyLoss for images. The input of this loss must be logit.

#### Classification Loss:
* CrossEntropyLoss: A loss for multi class classification. The input of this loss must be logit.
* BCEloss: A loss for binary classification. The input of this loss must be probability.
* CEWithLogitsLoss: A loss for binary classification. The input of this loss must be logit.

### tb_visualization: real time model performance evaluation
Once there are log recored. The script below could visualize the model's performance in real time.

```sh 
tensorboard --logdir=./outputs/${config_name}/log
```
## Future Plane
There are few models I was planed to add to this repo like Mask-RCNN.