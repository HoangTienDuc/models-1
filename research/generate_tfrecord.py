"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record

  csv file be like:
  	filename	xmin	ymin	xmax	ymax	class
0	000000537548.jpg	267	104	496	424	person
1	000000117891.jpg	206	1	639	409	person
2	000000120021.jpg	276	0	337	118	person
3	000000403255.jpg	355	118	385	160	person

"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import random
import math
from tensorflow.python.framework.versions import VERSION
from PIL import Image
from collections import namedtuple, OrderedDict
import contextlib2
from six.moves import range
import tensorflow.compat.v1 as tf
if VERSION >= "2.0.0a0":
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf




flags = tf.app.flags
flags.DEFINE_string('csv_input', default="person_labels.csv", help='Path to the CSV input')
flags.DEFINE_string('image_dir', help='Path to the image directory', default="/home/ducht/Documents/ducht/people/dataset/training/images")
flags.DEFINE_string('output_tfrecord_dir', default="./tfrecord", help='Path to output TFRecord')
flags.DEFINE_string('num_shards', default='10', help='num_shards')
FLAGS = flags.FLAGS


def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
  """Opens all TFRecord shards for writing and adds them to an exit stack.
  Args:
    exit_stack: A context2.ExitStack used to automatically closed the TFRecords
      opened in this function.
    base_path: The base path for all shards
    num_shards: The number of shards
  Returns:
    The list of opened TFRecords. Position k in the list corresponds to shard k.
  """
  tf_record_output_filenames = [
      '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
      for idx in range(num_shards)
  ]

  tfrecords = [
      exit_stack.enter_context(tf.python_io.TFRecordWriter(file_name))
      for file_name in tf_record_output_filenames
  ]

  return tfrecords

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def class_text_to_int(row_label):
    if row_label == 'person':
        return 1
    else:
        None

def _is_png(filename):
    """Determine if a file contains a PNG format image.
    Args:
      filename: string, path of the image file.
    Returns:
      boolean indicating if the image is a PNG.
    """
    _, file_extension = os.path.splitext(filename)
    return file_extension.lower() == '.png'

def create_tf_example(group, path):
  if os.path.exists(os.path.join(path, '{}'.format(group.filename))):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/source_id': bytes_feature(filename),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))
    return tf_example
  else:
    return None

def saver(grouped, image_dir, output_path, num_shards):
    writer = tf.python_io.TFRecordWriter(output_path)
    if num_shards == 1:
      for group in grouped:
          tf_example = create_tf_example(group, image_dir)
          if tf_example is not None:
            writer.write(tf_example.SerializeToString())
      writer.close()
    else:
      with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = open_sharded_output_tfrecords(
            tf_record_close_stack, output_path, num_shards)
        for index, group in enumerate(grouped):
          tf_example = create_tf_example(group, image_dir)
          output_shard_index = index % num_shards
          if tf_example is not None:
            output_tfrecords[output_shard_index].write(tf_example.SerializeToString())

def split_dataset(image_dir, grouped):
    valid_grouped = []
    for group in grouped:
      image_path = os.path.join(image_dir, group.filename)
      if os.path.isfile(image_path):
        valid_grouped.append(group)
    
    random.shuffle(valid_grouped)
    num_line = len(valid_grouped)
    train_ratio = 0.9
    train_grouped =  valid_grouped[ : int(math.floor(num_line * train_ratio))]
    validation_grouped = valid_grouped[int(math.ceil(num_line * train_ratio)) : ]
    return train_grouped, validation_grouped


def main(_):
    image_dir = os.path.join(os.getcwd(), FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    output_tfrecord_dir = FLAGS.output_tfrecord_dir
    if not os.path.exists(output_tfrecord_dir):
      os.makedirs(output_tfrecord_dir)
    grouped = split(examples, 'filename')
    train_grouped, validation_grouped = split_dataset(image_dir, grouped)
    num_shards=int(FLAGS.num_shards)
    saver(train_grouped, image_dir, os.path.join(output_tfrecord_dir, "./train.record"), num_shards)
    saver(validation_grouped, image_dir, os.path.join(output_tfrecord_dir, "./valid.record"), 1)
    print('Successfully created the TFRecords')

if __name__ == '__main__':
    tf.app.run()