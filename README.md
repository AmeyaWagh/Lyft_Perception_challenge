
# LYFT Perception Challenge


## Training

<!-- Loss

heads = 10 all = 40 

![loss](./assets/loss_40.png)

heads = 40 all = 100

![loss](./assets/loss2.png)

Validation loss

![val_loss](./assets/val_loss_40.png)

![val_loss](./assets/val_loss2.png) -->


| heads Epoch | all Epoch | loss | val_loss |
|:-----------:|:---------:|:----:|:--------:|
| 10 		  | 40		  | ![loss](./assets/loss_40.png) | ![val_loss](./assets/val_loss_40.png) |
| 40 		  | 100		  | ![loss](./assets/loss2.png) | ![val_loss](./assets/val_loss2.png) |
| 10		  | 40 		  | ![loss](./assets/loss3.png) | ![val_loss](./assets/val_loss3.png) |
| 20		  | 60 		  | ![loss](./assets/loss4.png) | ![val_loss](./assets/val_loss4.png) |


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