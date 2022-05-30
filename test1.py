####### Yield Detection Using Tensorflow Object Detection ########
# Setup Paths
import os
CUSTOM_MODEL_NAME = 'my_EfficientDet'
PRETRAINED_MODEL_NAME = 'efficientdet_d0_coco17_tpu-32'
PRETRAINED_MODEL_URL ='http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'
paths = {'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),'APIMODEL_PATH': os.path.join('Tensorflow','models'),
'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME),
'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'),
}
files = {'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'pipeline.config'),
'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}
for path in paths.values():
if not os.path.exists(path):
if os.name == 'posix':

!mkdir -p {path}
if os.name == 'nt':
!mkdir {path}
# Download TF Pretrained Models from Tensorflow Model Zoo and
Install TFOD
if os.name=='nt':
!pip install wget
import wget
if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research',
'object_detection')):
!git clone https://github.com/tensorflow/models
{paths['APIMODEL_PATH']}
# Install Tensorflow Object Detection
if os.name=='posix':
!apt-get install protobuf-compiler
!cd Tensorflow/models/research && protoc object_detection/protos/*.proto --
python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install .
if os.name=='nt':
url="https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
wget.download(url)
!move protoc-3.15.6-win64.zip {paths['PROTOC_PATH']}
!cd {paths['PROTOC_PATH']} && tar -xf protoc-3.15.6-win64.zip
os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths
['PROTOC_PATH'], 'bin'))
!cd Tensorflow/models/research && protoc object_detection/protos/*.pro
to --
python_out=. && copy object_detection\\packages\\tf2\\setup.py setup.py
&& python setup.py build && python setup.py install
!cd Tensorflow/models/research/slim && pip install -e .
VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research',
'object_detection', 'builders', 'model_builder_tf2_test.py')
# Verify Installation
!python {VERIFICATION_SCRIPT}
!pip install tensorflow --upgrade
!pip uninstall protobuf matplotlib -y
!pip install protobuf matplotlib==3.2
! pip install pyyaml
import object_detection
!pip list
if os.name =='posix':
!wget {PRETRAINED_MODEL_URL}
!mv {PRETRAINED_MODEL_NAME+'.tar.gz'} {paths['PRETRAINED_MODEL_PATH']
}
!cd {paths['PRETRAINED_MODEL_PATH']} && tar -
zxvf {PRETRAINED_MODEL_NAME+'.tar.gz'}
if os.name == 'nt':
wget.download(PRETRAINED_MODEL_URL)
!move {PRETRAINED_MODEL_NAME+'.tar.gz'} {paths['PRETRAINED_MODEL_
PATH']}
!cd {paths['PRETRAINED_MODEL_PATH']} && tar -
zxvf {PRETRAINED_MODEL_NAME+'.tar.gz'}
# Create Label Map
labels = [{'name':'Ripe', 'id':1}, {'name':'Unripe', 'id':2}]
with open(files['LABELMAP'], 'w') as f:
for label in labels:
f.write('item { \n')
f.write('\tname:\'{}\'\n'.format(label['name']))
f.write('\tid:{}\n'.format(label['id']))
f.write('}\n')
# Create TF records
ARCHIVE_FILES = os.path.join(paths['IMAGE_PATH'], 'archive.tar.gz')
if os.path.exists(ARCHIVE_FILES):
!tar -zxvf {ARCHIVE_FILES}
!pip install pytz
!python {files['TF_RECORD_SCRIPT']} -
x {os.path.join(paths['IMAGE_PATH'], 'train')} -l {files['LABELMAP']} -
o {os.path.join(paths['ANNOTATION_PATH'], 'train.record')}
!python {files['TF_RECORD_SCRIPT']} -
x {os.path.join(paths['IMAGE_PATH'], 'test')} -l {files['LABELMAP']} -
o {os.path.join(paths['ANNOTATION_PATH'], 'test.record')}
# Copy Model Config to Training Folder
if os.name =='posix':
!cp {os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_
NAME, 'pipeline.config')} {os.path.join(paths['CHECKPOINT_PATH'])}
if os.name == 'nt':
!copy {os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_
NAME, 'pipeline.config')} {os.path.join(paths['CHECKPOINT_PATH'])}
# Update Config For Transfer Learning
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos 
from google.protobuf import text_format
config = config_util.get_configs_from_pipeline_file(files['PIPELINE_
CONFIG'])
config
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:
proto_str = f.read()
text_format.Merge(proto_str, pipeline_config)
pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths
['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt0')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
pipeline_config.train_input_reader.tf_record_input_reader.input_path
[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = files['LABEL
MAP']
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_pa
th[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]
config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:
f.write(config_text)
# Train the model
TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research','object_detection', 'model_main_tf2.py')
command = "python {} --model_dir={} --pipeline_config_path={}".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'])
print(command)
# Evaluate the Model
command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'])
print(command)
!{command}
# Load Train Model From Checkpoint
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELI
NE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'],
is_training=False)
# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt15')).expect_partial()
@tf.function
def detect_fn(image):
image, shapes = detection_model.preprocess(image)
prediction_dict = detection_model.predict(image, shapes)
detections = detection_model.postprocess(prediction_dict, shapes)
return detections

# Detect from an Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
category_index = label_map_util.create_category_index_from_labelmap(
files['LABELMAP'])
IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', 'imagename.jpg')
img = cv2.imread(IMAGE_PATH)
image_np = np.array(img)
input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype
=tf.float32)
detections = detect_fn(input_tensor)
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
for key, value in detections.items()}
detections['num_detections'] = num_detections
# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].as
type(np.int64)
label_id_offset = 1
image_np_with_detections = image_np.copy()
viz_utils.visualize_boxes_and_labels_on_image_array(
image_np_with_detections,
detections['detection_boxes'],
detections['detection_classes']+label_id_offset,
detections['detection_scores'],
category_index,
use_normalized_coordinates=True,
min_score_thresh=.8,
agnostic_mode=False)

plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
plt.show()
# Freezing the Graph
FREEZE_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research','object_detection', 'exporter_main_v2.py ')
command = "python {} --input_type=image_tensor --pipeline_config_path={} --trained_checkpoint_dir={} --output_directory={}".format(FREEZE_SCRIPT ,files['PIPELINE_CONFIG'],
paths['CHECKPOINT_PATH'], paths['OUTPUT_PATH'])
print(command)