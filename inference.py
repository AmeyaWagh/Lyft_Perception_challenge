

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import skimage.io
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT_DIR = os.path.abspath("./")
MODEL_DIR = os.path.join('./', "logs")

sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append(os.path.join(os.getcwd(),"./Mask_RCNN/"))

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from moviepy.editor import VideoFileClip



class LyftChallengeConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "lyft_perception_challenge"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    # IMAGE_MAX_DIM = 128
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
    

    
config = LyftChallengeConfig()
config.display()

class InferenceConfig(LyftChallengeConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

model_path = os.path.join('./', "mask_rcnn_lyft.h5")

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)



RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
colors = [RED,GREEN,BLUE]


def segment_images(original_image):
    results = model.detect([original_image], verbose=0)
    r = results[0]
    f_mask = r['masks']
    f_class = r["class_ids"]
    

    no_ch = f_mask.shape[2]
    final_img = np.copy(original_image)
    for ch in range(no_ch):

        _id = f_class[ch]
        if _id==1: 
            color_id=0
        else:
            color_id=1
        mask_1 = f_mask[:,:,ch]
        mask1 = np.dstack([mask_1*colors[color_id][0],
                            mask_1*colors[color_id][1],
                            mask_1*colors[color_id][2]])
        final_img = cv2.addWeighted(final_img, 1, mask1.astype(np.uint8), 1, 0)
    return final_img


import sys, skvideo.io, json, base64


video = skvideo.io.vread('./test_video.mp4')
print(video.shape)

for rgb_frame in video:
    try:
        final_img = segment_images(rgb_frame)
        cv2.imshow('output', final_img[:,:,::-1])
        cv2.waitKey(1)
    except KeyboardInterrupt as e:
        break

def process_video(INPUT_FILE,OUTPUT_FILE):
    video = VideoFileClip(INPUT_FILE)
    processed_video = video.fl_image(segment_images)
    processed_video.write_videofile(OUTPUT_FILE, audio=False)

process_video('./test_video.mp4','./output_video.mp4')

exit()