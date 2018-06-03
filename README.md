


<!-- ![Header](./assets/crop_header.png) -->

<div style="text-align:center"><img src=./assets/Udacity_Header.png width="1000" height="400" ></div>


# Lyft Perception Challenge

The [lyft Perception challenge](https://www.udacity.com/lyft-challenge) in association with Udacity had an image segmentation task where the candidates had to submit their algorithm which could segment road and cars pixels precisely in real time. The challenge started on *May 1st,2018* and went through *June 3rd, 2018*.

## Approach
Although it was a segmentation problem and did not require instance segmentation, I went ahead with [MASK-RCNN](https://arxiv.org/pdf/1703.06870.pdf) as it was the state of the art algorithm in image segmentation and I was always intrigued to learn about it. Also I started on *28th*, just after finishing my first term, so transfer learning was my only shot. :sweat:

#### Mask-RCNN (A brief overview)

Mask-RCNN, also known as [Detectron](https://github.com/facebookresearch/Detectron) is a research platform for object detection developed by facebookresearch. It is mainly a modification of Faster RCNN with a segmentation branch parallel to class predictor and bounding box regressor. The vanilla ResNet is used in an FPN setting as a backbone to Faster RCNN so that features can be extracted at multiple levels of the feature pyramid
The network heads consists of the Mask branch which predicts the mask and a classification with bounding box regression branch. The architecture with FPN was used for the purpose of this competition

| Backbone 					| Heads 					 |
|:-------------------------:|:--------------------------:|
| ![FPN](./assets/fpn.png)  | ![FPN](./assets/heads.png) |
| Feature Pyramid network with Resnet | different head architecture with and without FPN |

The loss function consists of 3 losses *L = L<sub>class</sub> + L<sub>box</sub> + L<sub>mask</sub>* where
 - *L<sub>class</sub>*  uses log loss for true classes
 - *L<sub>box</sub>* uses smooth<sub>L1</sub> loss defined in [Fast RCNN]
 - *L<sub>mask</sub>* uses average binary cross entropy loss

The masks are predicted by a [Fully Connected Network](https://arxiv.org/pdf/1605.06211.pdf) for each RoI. This maintains the mxm dimension for each mask and thus for each instance of the object we get distinct masks. 

The model description after compiliation can be found at [model](./assets/model.png)

## Training

#### Pre-processing Data
The CARLA datset provided contains train and test images in png format where the masks are in the RED channel of the ground truth mask. 
```python
def process_labels(self,labels):
        
        # label 6 - lane lines pixels
        # label 7 - lane pixels
        # label 10 - car

        labels_new = np.zeros(labels.shape)
        labels_new_car = np.zeros(labels.shape)
        
        lane_line_idx = (labels == 6).nonzero()
        lane_idx = (labels == 7).nonzero()
        car_pixels = (labels == 10).nonzero()

        # remove car hood pixels
        car_hood_idx = (car_pixels[0] >= 495).nonzero()[0]
        car_hood_pixels = (car_pixels[0][car_hood_idx], \
                       car_pixels[1][car_hood_idx])

        labels_new[lane_line_idx] = 1
        labels_new[lane_idx] = 1

        labels_new_car[car_pixels] = 1
        labels_new_car[car_hood_pixels] = 0

        
        return np.dstack([labels_new,labels_new_car])
```

#### MaskRCNN Configuration
For this application Resnet-50 was used by setting `BACKBONE = "resnet50"` in config.

```python
class LyftChallengeConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    # BACKBONE = "resnet101"
    BACKBONE = "resnet50"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 2 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
```


#### Data Augmentation
As the samples provided were very less (1K), data augmentation was necessary to avoid overfitting. [imgaug](https://imgaug.readthedocs.io/en/latest/) is a python module which came handy in adding augmentation to the dataset. The train function of the MaskRCNN takes augmentation object and augments the images in its generator according to the rules defined in the augmentation object 

```python
augmentation = iaa.SomeOf((0, None), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=45),
                   iaa.Affine(rotate=90),
                   iaa.Affine(rotate=135)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0)),
        iaa.Affine(scale=(0.5, 1.5)),
        iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)}),
    ])
```

Examples of augmentation are given in [https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)

#### Training Loss

Instead of a single training loop, it was trained multiple times in smaller epochs to observe change in the loss with changes in parameters and to avoid overfitting. As the data was less, the network used to saturate quickly and required more augmentations to proceed. Also i did not wanted to go overboard on the augmentation so was observing which one works best. Below are the logs of final training setting with the above given augmentation.   

| heads Epoch | all Epoch | loss 							| val_loss 								|
|:-----------:|:---------:|:-------------------------------:|:-------------------------------------:|
| 10 		  | 40		  | ![loss](./assets/loss_40.png) 	| ![val_loss](./assets/val_loss_40.png)	|
| 40 		  | 100		  | ![loss](./assets/loss2.png) 	| ![val_loss](./assets/val_loss2.png) 	|
| 10		  | 40 		  | ![loss](./assets/loss3.png) 	| ![val_loss](./assets/val_loss3.png) 	|
| 20		  | 60 		  | ![loss](./assets/loss4.png) 	| ![val_loss](./assets/val_loss4.png) 	|


## Results

```
Your program runs at 1.703 FPS

Car F score: 0.519 | Car Precision: 0.509 | Car Recall: 0.521 | Road F score: 0.961 | Road Precision: 0.970 | Road Recall: 0.926 | Averaged F score: 0.740
```


## Inference and Submission

### Submission
Submission requires files to be encoded in a json. `test_inference.py` contains the inference and submission code. In attempt to increase the FPS, The encode function was replaced with the follows which was shared on the forum
```python
def encode(array):
    retval, buffer = cv2.imencode('.png', array)
    return base64.b64encode(buffer).decode("utf-8")
```

## Reference
https://github.com/matterport/Mask_RCNN
```
@misc{Charles2013,
  author = {waleedka et.al},
  title = {Mask R-CNN for Object Detection and Segmentation},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/matterport/Mask_RCNN}},
  commit = {6c9c82f5feaf5d729a72c67f33e139c4bc71399b}
}
```

 - [Mask RCNN](https://arxiv.org/pdf/1703.06870.pdf)
 - [Fast RCNN](https://arxiv.org/pdf/1504.08083.pdf)
 - [Faster RCNN](https://arxiv.org/pdf/1506.01497.pdf)
 - [Feature Pyramid Networks for Object Detection](https://arxiv.org/pdf/1612.03144.pdf)
 - [Fully Connected Network](https://arxiv.org/pdf/1605.06211.pdf)


## Author

Ameya Wagh [aywagh@wpi.edu](aywagh@wpi.edu)