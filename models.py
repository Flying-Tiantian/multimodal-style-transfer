import os
import tensorflow as tf
import abc

from model_util import *


class ABCModel(object):
    """Abstract meta class for model.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyparams):
        self.name = ''
        self.input_size = 512
        self.hyparams = hyparams
        self.saver = None

    def get_name(self):
        return self.name

    @abc.abstractmethod
    def forward_pass(self, input_layer, trainable=True):
        pass

    def get_input_size(self):
        return self.input_size

    def _loss(self, logits, labels):
        # Calculate the average cross entropy loss across the batch.
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(
            cross_entropy, name='cross_entropy')
        tf.add_to_collection(tf.GraphKeys.LOSSES, cross_entropy_mean)

        acc_loss = tf.add_n(tf.get_collection(
            tf.GraphKeys.LOSSES), name='acc_loss')
        tf.summary.scalar('acc_loss', acc_loss)
        l2_loss = tf.add_n(tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_loss')
        tf.summary.scalar('l2_loss', l2_loss)
        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        total_loss = tf.add(acc_loss, l2_loss, name='total_loss')
        tf.summary.scalar('total_loss', total_loss)

        return total_loss

    def train(self, train_data):
        with tf.name_scope('train') as scope:
            logits = self.forward_pass(train_data['images'])
        self.total_loss = self._loss(logits, train_data['labels'])

        global_step = tf.train.get_global_step()

        opt = tf.train.AdamOptimizer()
        train_op = opt.minimize(self.total_loss, global_step)

        return train_op, self.total_loss

    def test(self, test_data):
        with tf.name_scope('test') as scope:
            logits = self.forward_pass(test_data['images'])
            # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, test_data['labels'], 1)

        return top_k_op

    def _get_saver(self):
        if self.saver is None:
            variables = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, self.name)
            self.saver = tf.train.Saver(variables)
        return self.saver

    def save(self, sess, save_path):
        saver = self._get_saver()

        return saver.save(sess, save_path, tf.train.get_global_step())

    def restore(self, sess, checkpoint_dir):
        # pylint: disable=no-member
        if os.path.exists(checkpoint_dir):
            if os.path.isdir(checkpoint_dir):
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    checkpoint_path = ckpt.model_checkpoint_path
                else:
                    print('No checkpoint file found')
                    return -1
            else:
                checkpoint_path = checkpoint_dir
            saver = self._get_saver()
            # Restores from checkpoint
            print('Restore from %s ...' % checkpoint_path)
            saver.restore(sess, checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = checkpoint_path.split(
                '/')[-1].split('-')[-1]

            try:
                global_step = int(global_step)
            except ValueError:
                global_step = 0
            
            print('Complete.')
            return global_step

        else:
            print('No checkpoint file found')
            return -1
        # pylint: enable=no-member


class style_transfer_model(ABCModel):
    def __init__(self):
        super().__init__(None)
        self.name = 'style_transfer_model'
        self.weight_decay = 0.001
        self.input_size = 1024
        self.loss_model = vgg_16_model(None)
        self._styled = None

    def _style_subnet(self, input_layer, trainable=True):
        def rgb_l_block(input_layer, trainable):
            with tf.variable_scope('conv1') as scope:
                conv1 = style_conv2d_layer(
                    input_layer, 9, 32, 1, self.weight_decay, trainable)
            with tf.variable_scope('conv2') as scope:
                conv2 = style_conv2d_layer(
                    conv1, 3, 64, 2, self.weight_decay, trainable)
            with tf.variable_scope('conv3') as scope:
                conv3 = style_conv2d_layer(
                    conv2, 3, 128, 2, self.weight_decay, trainable)
            with tf.variable_scope('block1') as scope:
                block1 = basic_block(conv3, 0, 1, self.weight_decay, trainable)
            with tf.variable_scope('block2') as scope:
                block2 = basic_block(block1, 0, 1, self.weight_decay, trainable)
            with tf.variable_scope('block3') as scope:
                block3 = basic_block(block2, 0, 1, self.weight_decay, trainable)

            return block3
        with tf.variable_scope('rgb_2_gray') as scope:
            gray = tf.image.rgb_to_grayscale(input_layer)

        with tf.variable_scope('rgb_block') as scope:
            rgb_block = rgb_l_block(input_layer, trainable)
        with tf.variable_scope('l_block') as scope:
            l_block = rgb_l_block(gray, trainable)

        with tf.variable_scope('combine') as scope:
            combine = tf.concat([rgb_block, l_block], 3)

        with tf.variable_scope('conv_block') as scope:
            with tf.variable_scope('block1') as scope:
                block1 = basic_block(combine, 0, 1, self.weight_decay, trainable)
            with tf.variable_scope('block2') as scope:
                block2 = basic_block(block1, 0, 1, self.weight_decay, trainable)
            with tf.variable_scope('block3') as scope:
                block3 = basic_block(block2, 0, 1, self.weight_decay, trainable)
            with tf.variable_scope('upsample1') as scope:
                upsample1 = upsample_conv2d_layer(
                    block3, 3, 64, self.weight_decay, trainable)
            with tf.variable_scope('upsample2') as scope:
                upsample2 = upsample_conv2d_layer(
                    upsample1, 3, 32, self.weight_decay, trainable)
            with tf.variable_scope('output') as scope:
                output = style_conv2d_layer(
                    upsample2, 3, 3, 1, self.weight_decay, trainable)
        return output

    def _enhance_subnet(self, input_layer, trainable=True):
        with tf.variable_scope('conv1') as scope:
            conv1 = style_conv2d_layer(
                input_layer, 9, 32, 1, self.weight_decay, trainable)
        with tf.variable_scope('conv2') as scope:
            conv2 = style_conv2d_layer(
                conv1, 3, 64, 2, self.weight_decay, trainable)
        with tf.variable_scope('conv3') as scope:
            conv3 = style_conv2d_layer(
                conv2, 3, 128, 2, self.weight_decay, trainable)
        with tf.variable_scope('conv4') as scope:
            conv4 = style_conv2d_layer(
                conv3, 3, 256, 2, self.weight_decay, trainable)
        with tf.variable_scope('block1') as scope:
            block1 = basic_block(conv4, 0, 1, self.weight_decay, trainable)
        with tf.variable_scope('block2') as scope:
            block2 = basic_block(block1, 0, 1, self.weight_decay, trainable)
        with tf.variable_scope('block3') as scope:
            block3 = basic_block(block2, 0, 1, self.weight_decay, trainable)
        with tf.variable_scope('block4') as scope:
            block4 = basic_block(block3, 0, 1, self.weight_decay, trainable)
        with tf.variable_scope('block5') as scope:
            block5 = basic_block(block4, 0, 1, self.weight_decay, trainable)
        with tf.variable_scope('block6') as scope:
            block6 = basic_block(block5, 0, 1, self.weight_decay, trainable)
        with tf.variable_scope('upsample1') as scope:
            upsample1 = upsample_conv2d_layer(
                block6, 3, 0, self.weight_decay, trainable)
        with tf.variable_scope('upsample2') as scope:
            upsample2 = upsample_conv2d_layer(
                upsample1, 3, 0, self.weight_decay, trainable)
        with tf.variable_scope('upsample3') as scope:
            upsample3 = upsample_conv2d_layer(
                upsample2, 3, 0, self.weight_decay, trainable)
        with tf.variable_scope('output') as scope:
            output = style_conv2d_layer(
                    upsample3, 3, 3, 1, self.weight_decay, trainable)

        return output

    def _refine_subnet(self, input_layer, trainable=True):
        with tf.variable_scope('conv1') as scope:
            conv1 = style_conv2d_layer(
                input_layer, 9, 32, 1, self.weight_decay, trainable)
        with tf.variable_scope('conv2') as scope:
            conv2 = style_conv2d_layer(
                conv1, 3, 64, 2, self.weight_decay, trainable)
        with tf.variable_scope('conv3') as scope:
            conv3 = style_conv2d_layer(
                conv2, 3, 128, 2, self.weight_decay, trainable)
        with tf.variable_scope('block1') as scope:
            block1 = basic_block(conv3, 3, 1, self.weight_decay, trainable)
        with tf.variable_scope('block2') as scope:
            block2 = basic_block(block1, 3, 1, self.weight_decay, trainable)
        with tf.variable_scope('block3') as scope:
            block3 = basic_block(block2, 3, 1, self.weight_decay, trainable)
        with tf.variable_scope('upsample1') as scope:
            upsample1 = upsample_conv2d_layer(
                block3, 3, 0, self.weight_decay, trainable)
        with tf.variable_scope('upsample2') as scope:
            upsample2 = upsample_conv2d_layer(
                upsample1, 3, 0, self.weight_decay, trainable)
        with tf.variable_scope('output') as scope:
            output = style_conv2d_layer(
                upsample2, 3, 3, 1, self.weight_decay, trainable)

        return output

    '''
    RGB --> YUV(YCbCr)
    Y = 0.2989*R + 0.5866*G + 0.1145*B
    U(Cb) = -0.1684*R -0.3311*G + 0.4997*B
    V(Cr) = 0.4998*R -0.4187*G -0.0813*B
    '''

    def forward_pass(self, input_layer, trainable=True):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:
            with tf.variable_scope('resize') as scope:
                image_256 = tf.image.resize_bilinear(input_layer, (256, 256))
            
            with tf.variable_scope('style_subnet') as scope:
                styled = self._style_subnet(image_256, trainable)
            with tf.variable_scope('upsample1') as scope:
                image_512 = upsample_layer(styled)
            with tf.variable_scope('enhance_subnet') as scope:
                enhanced = self._enhance_subnet(image_512, trainable)
            with tf.variable_scope('upsample2') as scope:
                if not trainable:
                    to_refine = upsample_layer(enhanced)
                else:
                    to_refine = enhanced
            with tf.variable_scope('refine_subnet') as scope:
                output = tf.add(self._refine_subnet(
                    to_refine, trainable), to_refine)

            if self._styled is None:
                self._styled = styled
                self._enhanced = enhanced
                self._output = output

            tf.summary.image('content', input_layer, max_outputs=10)
            tf.summary.image('styled', styled, max_outputs=10)
            tf.summary.image('enhanced', enhanced, max_outputs=10)
            tf.summary.image('output', output, max_outputs=10)

        return output

    def _loss(self, input_images, style_image):
        image_512 = input_images
        with tf.name_scope('downsample') as scope:
            image_256 = downsample_layer(image_512)

        with tf.name_scope('style_loss_styled') as scope:
            style_loss1 = self.loss_model.style_loss(image_256, self._styled)
        with tf.name_scope('style_loss_enhanced') as scope:
            style_loss2 = self.loss_model.style_loss(image_512, self._enhanced)
        with tf.name_scope('style_loss_output') as scope:
            style_loss3 = self.loss_model.style_loss(image_512, self._output)

        with tf.name_scope('content_loss') as scope:
            content_loss = self.loss_model.content_loss(image_512, self._output)

        l2_loss = tf.add_n(tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_loss')

        total_style_loss = weighted_add([style_loss1, style_loss2, style_loss3], [
                                        1, 0.5, 0.25], name='total_style_loss')

        total_loss = tf.add_n(
            [total_style_loss, content_loss, l2_loss], name='total_loss')

        tf.summary.scalar('style_loss1', style_loss1)
        tf.summary.scalar('style_loss2', style_loss2)
        tf.summary.scalar('style_loss3', style_loss3)
        tf.summary.scalar('content_loss', content_loss)
        tf.summary.scalar('l2_loss', l2_loss)

        return total_loss

    def train(self, input_images, style_image):
        with tf.name_scope('train') as scope:
            output = self.forward_pass(input_images)
            self.total_loss = self._loss(input_images, style_image)

        global_step = tf.train.get_global_step()

        opt = tf.train.AdamOptimizer()
        train_op = opt.minimize(self.total_loss, global_step)

        return train_op, self.total_loss

    def test(self, input_images):
        with tf.name_scope('test') as scope:
            output = self.forward_pass(input_images, trainable=False)

        return output

    def restore(self, sess, train_dir, vgg_16_ckpt):
        self.loss_model.restore(sess, vgg_16_ckpt)
        if train_dir is not None:
            super().restore(sess, train_dir)


class vgg_16_model(ABCModel):
    def __init__(self, hyparams):
        super().__init__(hyparams)
        self.name = 'vgg_16'
        self.weight_decay = 0.001

    def _gram_matrices(self, feature):
        shape = feature.get_shape()
        nlc = tf.reshape(feature, (shape[0], -1, shape[3]))
        ncl = tf.transpose(nlc, [0, 2, 1])
        gm = tf.div(tf.matmul(ncl, nlc), (shape[1] * shape[2] * shape[3]).value)

        return gm

    def forward_pass(self, input_layer, trainable=True):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:
            with tf.variable_scope('conv1') as scope:
                with tf.variable_scope('conv1_1') as scope:
                    conv1_1 = vgg_conv2d_layer(
                        input_layer, 3, 64, 1, self.weight_decay, trainable)
                with tf.variable_scope('conv1_2') as scope:
                    conv1_2 = vgg_conv2d_layer(
                        conv1_1, 3, 64, 1, self.weight_decay, trainable)
            with tf.variable_scope('pool1') as scope:
                pool1 = tf.nn.avg_pool(conv1_2, [1, 2, 2, 1], [
                                       1, 2, 2, 1], 'VALID')
            with tf.variable_scope('conv2') as scope:
                with tf.variable_scope('conv2_1') as scope:
                    conv2_1 = vgg_conv2d_layer(
                        pool1, 3, 128, 1, self.weight_decay, trainable)
                with tf.variable_scope('conv2_2') as scope:
                    conv2_2 = vgg_conv2d_layer(
                        conv2_1, 3, 128, 1, self.weight_decay, trainable)
            with tf.variable_scope('pool2') as scope:
                pool2 = tf.nn.avg_pool(conv2_2, [1, 2, 2, 1], [
                                       1, 2, 2, 1], 'VALID')
            with tf.variable_scope('conv3') as scope:
                with tf.variable_scope('conv3_1') as scope:
                    conv3_1 = vgg_conv2d_layer(
                        pool2, 3, 256, 1, self.weight_decay, trainable)
                with tf.variable_scope('conv3_2') as scope:
                    conv3_2 = vgg_conv2d_layer(
                        conv3_1, 3, 256, 1, self.weight_decay, trainable)
                with tf.variable_scope('conv3_3') as scope:
                    conv3_3 = vgg_conv2d_layer(
                        conv3_2, 3, 256, 1, self.weight_decay, trainable)
            with tf.variable_scope('pool3') as scope:
                pool3 = tf.nn.avg_pool(conv3_3, [1, 2, 2, 1], [
                                       1, 2, 2, 1], 'VALID')
            with tf.variable_scope('conv4') as scope:
                with tf.variable_scope('conv4_1') as scope:
                    conv4_1 = vgg_conv2d_layer(
                        pool3, 3, 512, 1, self.weight_decay, trainable)
                with tf.variable_scope('conv4_2') as scope:
                    conv4_2 = vgg_conv2d_layer(
                        conv4_1, 3, 512, 1, self.weight_decay, trainable)
                with tf.variable_scope('conv4_3') as scope:
                    conv4_3 = vgg_conv2d_layer(
                        conv4_2, 3, 512, 1, self.weight_decay, trainable)
            with tf.variable_scope('pool4') as scope:
                pool4 = tf.nn.avg_pool(conv4_3, [1, 2, 2, 1], [
                                       1, 2, 2, 1], 'VALID')
            with tf.variable_scope('conv5') as scope:
                with tf.variable_scope('conv5_1') as scope:
                    conv5_1 = vgg_conv2d_layer(
                        pool4, 3, 512, 1, self.weight_decay, trainable)
                with tf.variable_scope('conv5_2') as scope:
                    conv5_2 = vgg_conv2d_layer(
                        conv5_1, 3, 512, 1, self.weight_decay, trainable)
                with tf.variable_scope('conv5_3') as scope:
                    conv5_3 = vgg_conv2d_layer(
                        conv5_2, 3, 512, 1, self.weight_decay, trainable)

            with tf.variable_scope('gram_matrices') as scope:
                gm1 = self._gram_matrices(conv1_1)
                gm2 = self._gram_matrices(conv2_1)
                gm3 = self._gram_matrices(conv3_1)
                gm4 = self._gram_matrices(conv4_1)
                # gm5 = self._gram_matrices(conv5_1)

        return [gm1, gm2, gm3, gm4], conv4_2

    def style_loss(self, input_layer1, input_layer2, weights=[0.2, 0.2, 0.2, 0.2, 0.2]):
        gms1, _ = self.forward_pass(input_layer1, trainable=False)
        gms2, _ = self.forward_pass(input_layer2, trainable=False)

        losses = []

        for gm1, gm2, weight in zip(gms1, gms2, weights):
            style_loss = tf.reduce_mean(tf.square(gm1 - gm2))
            style_loss = tf.multiply(
                style_loss, tf.constant(weight, dtype=tf.float32))
            losses.append(style_loss)

        return tf.add_n(losses, name='style_loss')

    def content_loss(self, input_layer1, input_layer2):
        _, content1 = self.forward_pass(input_layer1, trainable=False)
        _, content2 = self.forward_pass(input_layer2, trainable=False)

        content_loss = tf.reduce_mean(
            tf.square(content1 - content2), name='content_loss')

        return content_loss
