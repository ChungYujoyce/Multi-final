import os
import tarfile
import tempfile
import numpy as np
from PIL import Image
import sys
import cv2
import tensorflow as tf
import time
# Needed to show segmentation colormap labels
sys.path.append('utils')
from utils import get_dataset_colormap

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True



class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513 #513
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

    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map

def image_inference(model_name, picure, label_color, output_name, output_name_color):
  #run single image on model
  MODEL = DeepLabModel(model_name) 
  print('model loaded successfully!')
  
  image = Image.open(picure)
  resized_im, seg_map = MODEL.run(image)
  seg_image = get_dataset_colormap.label_to_color_image(seg_map,label_color).astype(np.uint8)
  
  resized_im = np.array(resized_im)
  color_and_mask = cv2.addWeighted(resized_im, 0.4, seg_image, 0.6, 0.0)

  resized_im = Image.fromarray(resized_im, 'RGB')
  seg_image = Image.fromarray(seg_image, 'RGB')
  color_and_mask = Image.fromarray(color_and_mask, 'RGB')

  #resized_im = cv2.cvtColor(resized_im, cv2.COLOR_RGB2BGR)
  #seg_image.show()
  global start_t
  duration = time.time()-start_t
  print(duration)
  duration = format(duration,'.4f')

  seg_image.save('static/'+output_name, format='PNG')
  color_and_mask.save('static/'+output_name_color, format='PNG')

  return duration
