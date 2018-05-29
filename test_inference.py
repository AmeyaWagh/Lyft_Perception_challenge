import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO


import os

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


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

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
    

config = ShapesConfig()
config.display()


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

file = sys.argv[-1]

if file == 'demo.py':
  print ("Error loading video")
  quit

model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

model_path = os.path.join('./', "mask_rcnn_lyft.h5")
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


def segment_image(image_frame):
    results = model.detect([image_frame], verbose=0)
    r = results[0]
    road_mask = r['masks'][:,:,0]
    car_mask = r['masks'][:,:,1]

    return car_mask,road_mask

# Define encoder function
def encode(array):
    pil_img = Image.fromarray(array)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

video = skvideo.io.vread(file)

answer_key = {}

# Frame numbering starts at 1
frame = 1

for rgb_frame in video:
    
    # Grab red channel  
    # red = rgb_frame[:,:,0]    
    # Look for red cars :)
    # binary_car_result = np.where(red>250,1,0).astype('uint8')
    
    # Look for road :)
    # binary_road_result = binary_car_result = np.where(red<20,1,0).astype('uint8')
    binary_car_result,binary_road_result = segment_image(rgb_frame)

    answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
    
    # Increment frame
    frame+=1

# Print output in proper json format
print (json.dumps(answer_key))