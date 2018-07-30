import argparse
import datetime
import sys
import os
import time

import numpy as np
import tensorflow as tf
from input_generator import input_generator
import models

FLAGS = None
VGG_16_CKPT = './data/vgg_16.ckpt'


def train():
    reader = input_generator(FLAGS.data_path)

    model = models.style_transfer_model()

    FLAGS.train_dir += model.get_name()

    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    else:
        tf.gfile.MakeDirs(FLAGS.train_dir)

    with tf.Graph().as_default():
        style_img = reader.read_one_img(FLAGS.style_img, FLAGS.batch_size)

        global_step = tf.train.get_or_create_global_step()

        train_op, loss_op = model.train(reader.input(FLAGS.batch_size), style_img)

        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            # saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            print('Init all variables...')
            sess.run(tf.global_variables_initializer())
            print('Complete.')

            model.restore(sess, None, VGG_16_CKPT)

            for i in range(FLAGS.max_steps):
                _, loss, step = sess.run([train_op, loss_op, global_step])
                if step % FLAGS.save_checkpoint_interval == 0:
                    model.save(sess, os.path.join(
                        FLAGS.train_dir, 'model.ckpt'))
                if step % FLAGS.save_summary_interval == 0:
                    summary = sess.run(summary_op)
                    summary_writer.add_summary(summary, global_step=step)

                    print('%s: step %d, loss %.2f' %
                          (datetime.datetime.now(), step, loss))

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data/train_data',
        help='Directory containing train data.')
    parser.add_argument(
        '--style_img',
        type=str,
        default='2-style2.jpg',
        help='Style image.')
    parser.add_argument(
        '--train_dir',
        type=str,
        default='./data/train/',
        help='Directory where to write event logs and checkpoint.')
    parser.add_argument(
        '--max_steps',
        type=int,
        default=50000,
        help='Number of batches to run.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size.')
    parser.add_argument(
        '--save_checkpoint_interval',
        type=int,
        default=1000,
        help='Save checkpoint every n steps.')
    parser.add_argument(
        '--save_summary_interval',
        type=int,
        default=100,
        help='Save summary every n steps.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
