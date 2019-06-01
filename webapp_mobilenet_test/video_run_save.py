import os
#from io import BytesIO
import tarfile
import tempfile
#from six.moves import urllib
#from matplotlib import gridspec
#from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import sys
#import scipy
import cv2
import tensorflow as tf
import time
# Needed to show segmentation colormap labels
sys.path.append('utils')
import get_dataset_colormap

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True



class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 321 #513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(config = config, graph=self.graph)

  def run(self, image):
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

    global start_t
    start_t = time.time()
    #print(resized_image.size)

    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_pascal_label_colormap():
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):

  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

#model
MODEL = DeepLabModel("models/mobilev2_restride32_100000.tar.gz")
#MODEL = DeepLabModel("models/mobilev2_stride16.tar.gz") 
print('model loaded successfully!')

## Webcam demo
cap = cv2.VideoCapture("models/1.mp4")
#cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output3.mp4',fourcc, 20.0, (642,280))
# Next line may need adjusting depending on webcam resolution
#final = np.zeros((1, 288, 1026, 3))
elapsed_times = []
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
      
      # From cv2 to PIL
      cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      pil_im = Image.fromarray(cv2_im)

      # Run model
      resized_im, seg_map = MODEL.run(pil_im)

      global start_t
      duration = time.time() - start_t
      elapsed_times.append(duration)
      
      # Adjust color of mask
      seg_image = get_dataset_colormap.label_to_color_image(
          seg_map).astype(np.uint8)
      frame = np.array(pil_im)
      r = seg_image.shape[1] / frame.shape[1] #*1.0
      dim = (int(frame.shape[0] * r), seg_image.shape[1])[::-1]
      print(dim) #something wrong with dim
    #print(frame.shape)
      resized = cv2.resize(frame, dim, interpolation = cv2.INTER_CUBIC)
        #print("resize done",resized.shape)
      resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        
        # Stack horizontally color frame and mask
      color_and_mask = np.hstack((resized, seg_image))

        #out.write(color_and_mask)
      outputdata = color_and_mask
    
      out.write(outputdata)
  
      cv2.imshow('frame', outputdata)
      if cv2.waitKey(25) & 0xFF == ord('q'):
          break
    else:
        break
cap.release()
out.release()
print("video_end")
print('Average time: {:.4f}, about {:.6f} fps'.format(np.mean(elapsed_times), 1/np.mean(elapsed_times)))
