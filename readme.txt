
## Requirement
 Python 3.5
 PyTorch 0.4.1
 torchvision
 numpy
 Cython
 pydensecrf

## Path Setting
 You can modify the path in config.py

## Training
 1. First downloading the pretrained ReseNext model. (https://drive.google.com/file/d/1dnH-IHwmu9xFPlyndqI6MfF4LvH6JKNQ/view) and put it into ./resnext/
 2. Download the Dataset. Here we only provide our distraction datasets, since you can download the other datasets from the corresponding official website.
 2. python train.py

## Testing
 1. Download the pretrained model (https://drive.google.com/file/d/1zSY2QWqauncB29ALov7q6yjWSZPc08GU/view) and put them in ckpt/models
 2. python test.py



 



