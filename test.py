import argparse
import datetime
import sys
import os
import time
import cv2

import numpy as np
import tensorflow as tf
from input_generator import input_generator
import models

FLAGS = None
MODEL_DIR = './data/train/'


def test():
    reader = input_generator(FLAGS.data_path)

    model = models.style_transfer_model()

    if not tf.gfile.Exists(FLAGS.output_path):
        tf.gfile.MakeDirs(FLAGS.output_path)

    file_names = os.listdir(FLAGS.data_path)

    with tf.Graph().as_default():
        input_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(1, 512, 512, 3))
        output_op = model.test(input_placeholder)

        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True  # pylint: disable=no-member
        with tf.Session(config=config) as sess:

            print('Init all variables...')
            sess.run(tf.global_variables_initializer())
            print('Complete.')

            print('Restore.')
            model.restore(sess, os.path.join(
                MODEL_DIR, model.get_name()), None)
            print('Complete.')

            for file_name in file_names:
                input_image = sess.run(reader.read_one_img(
                    os.path.join(FLAGS.data_path, file_name), 1))
                output_image = sess.run(output_op, feed_dict={
                                        input_placeholder: input_image})
                cv2.imwrite(os.path.join(FLAGS.output_path,
                                         file_name), np.squeeze(output_image))


def main(argv=None):  # pylint: disable=unused-argument
    test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data/test_data',
        help='Directory containing train data.')
    parser.add_argument(
        '--output_path',
        type=str,
        default='./data/test_result',
        help='Style image.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
