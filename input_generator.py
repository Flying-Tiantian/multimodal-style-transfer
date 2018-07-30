import os
import sys
import tensorflow as tf

DEBUG = False

class input_generator:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.num_preprocess_threads = 4
        self.min_queue_examples = 100

    def _debug(self, tensor_list):
        if not DEBUG:
            return

        with tf.Session() as sess:
            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            values = sess.run(tensor_list)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

        return values

    def _name_queue(self, dir_path):
        file_names = sorted(os.listdir(dir_path))
        file_paths = [os.path.join(dir_path, file_name) for file_name in file_names]
        filename_queue = tf.train.string_input_producer(file_paths)

        return filename_queue

    def _file_reader(self, filename_queue):
        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)
        
        image = tf.image.decode_jpeg(value, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        

        return image

    def _img_preprocess(self, image, size=512):
        shape = tf.shape(image)
        dims = tf.unstack(shape)
        cent_size = tf.minimum(dims[0], dims[1])
        square_img = tf.image.resize_image_with_crop_or_pad(image, cent_size, cent_size)
        resized = tf.image.resize_images(square_img, (size, size))
        resized.set_shape([size, size, 3])

        return resized

    def _gen_batch(self, image, batch_size):
        image_batch = tf.train.shuffle_batch(
                [image],
                batch_size=batch_size,
                num_threads=self.num_preprocess_threads,
                capacity=self.min_queue_examples + 3 * batch_size,
                min_after_dequeue=self.min_queue_examples)
        self._debug([image_batch])

        return image_batch

    def input(self, batch_size):
        with tf.name_scope('input_generator') as scope:
            filename_queue = self._name_queue(self.dir_path)
            image = self._img_preprocess(self._file_reader(filename_queue))
            image_batch = self._gen_batch(image, batch_size)

        return image_batch

    def read_one_img(self, img_path, batch_size, size=512):
        file_name = tf.read_file(img_path)
        image = tf.image.decode_jpeg(file_name)
        image = tf.image.convert_image_dtype(image, tf.float32)
        resized = tf.image.resize_images(image, (size, size))
        resized.set_shape([size, size, 3])
        batch = tf.stack([resized] * batch_size)

        return batch

if __name__ == '__main__':
    DEBUG = True
    reader = input_generator('./data/train_data')
    data = reader.input(1)
