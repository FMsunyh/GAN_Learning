# -*- coding: utf-8 -*-
# @Time    : 5/28/2018 3:15 PM
# @Author  : sunyonghai
# @File    : dcgan_mnist.py
# @Software: ZJ_AI
import PIL.Image
import tensorflow as tf
import numpy as np
import pickle
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_logger():
    fmt = '%(levelname)s:%(message)s'
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(fmt))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    return logger

logger = get_logger()

logger.info('info')
logger.debug('debug')
logger.warn('warn')

from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets("../MNIST_data")
# input

def get_inputs(noise_size, image_height,image_width, image_depth):
    real_img = tf.placeholder(tf.float32, [None, image_height,image_width, image_depth], name='real_img')
    noise_img = tf.placeholder(tf.float32, [None, noise_size], name='noise_img')

    return real_img, noise_img

# 生成器
def get_generator(noise_img, out_dim, is_train=True, alpha=0.01):
    """

    :param noise_img:  产生的噪声输入
    :param out_dim:
    :param reuse:
    :param alpha:
    :return:
    """
    with tf.variable_scope("generator", reuse=(not is_train)):
        layer1 = tf.layers.dense(noise_img, 4*4*512)
        layer1 = tf.reshape(layer1, [-1, 4, 4, 512])
        layer1 = tf.layers.batch_normalization(layer1, training=is_train)

        # Leaky ReLU
        layer1 = tf.maximum(alpha * layer1, layer1)
        layer1 = tf.nn.dropout(layer1, keep_prob=0.8)

        # 4 x 4 x  512 to 7 x 7 x 256
        layer2 = tf.layers.conv2d_transpose(layer1, 256, 4, strides=1, padding='valid')
        layer2 = tf.layers.batch_normalization(layer2, training=is_train)
        layer2 = tf.maximum(alpha * layer2, layer2)
        layer2 = tf.nn.dropout(layer2, keep_prob=0.8)

        # 7 x 7 x 256 to 14 x 14 x 128
        layer3 = tf.layers.conv2d_transpose(layer2, 128, 3, strides=2, padding='same')
        layer3 = tf.layers.batch_normalization(layer3, training=is_train)
        layer3 = tf.maximum(alpha * layer3, layer3)
        layer3 = tf.nn.dropout(layer3, keep_prob=0.8)

        # 14 x 14 x 128 to 28 x 28 x 1
        logits = tf.layers.conv2d_transpose(layer3, out_dim, 3, strides=2, padding='same')

        # MNIST 原始数据集的像素范围在0-1,这里的生成图片范围为（-1, 1）
        # 因此在训练时，记住要把MNIST像素范围进行resize
        outputs = tf.tanh(logits)
        return outputs

def get_discriminator(img, reuse=False, alpha=0.01):
    """

    :param img: 输入
    :param reuse: 使用2次
    :param alpha:
    :return:
    """
    with tf.variable_scope("discriminator", reuse=reuse):
        # 28 x 28 x1 to 14 x 14 x128
        layer1 = tf.layers.conv2d(img, 128, 3, strides=2, padding='same')
        layer1 = tf.maximum(alpha * layer1, layer1)
        layer1 = tf.nn.dropout(layer1, keep_prob=0.8)

        # 14 x 14 x 128 to 7 x 7 x 256
        layer2 = tf.layers.conv2d(layer1, 256, 3, strides=2, padding='same')
        layer2 = tf.layers.batch_normalization(layer2, training=True)
        layer2 = tf.maximum(alpha * layer2, layer2)
        layer2 = tf.nn.dropout(layer2, keep_prob=0.8)

        # 7 x 7 x 256 to 4 x 4 x 512
        layer3 = tf.layers.conv2d(layer2, 512, 3, strides=2, padding='same')
        layer3 = tf.maximum(alpha * layer3, layer3)
        layer3 = tf.nn.dropout(layer3,keep_prob=0.8)

        # 4 x 4 x 512 to 4 x 4 x 512 x 1
        flatten = tf.reshape(layer3, (-1, 4*4*512))
        logits = tf.layers.dense(flatten, 1)
        outputs = tf.sigmoid(logits)

        return logits, outputs

def get_loss(inputs_real, inputs_noise, image_depth, smooth=0.1):
    g_outputs = get_generator(inputs_noise, image_depth, is_train=True)
    d_logits_real, d_outputs_real = get_discriminator(inputs_real)
    d_logits_fake,d_outputs_fake = get_discriminator(g_outputs, reuse=True)

    # loss
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_outputs_fake)) * (1-smooth))
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_outputs_real)*(1-smooth)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_outputs_fake)))

    d_loss = tf.add(d_loss_real, d_loss_fake)
    return g_loss, d_loss

def get_optimizer(g_loss, d_loss, betal=0.4, learning_rate=0.001):
    # 优化器
    train_vars = tf.trainable_variables()
    g_vars = [var for var in train_vars if var.name.startswith("generator")]
    d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

    return g_train_opt, d_train_opt, g_vars

# 构建网络

tf.reset_default_graph()

# batch_size
batch_size = 64
noise_size = 100
epochs = 2
n_samples = 25
learning_rate = 0.001
betal = 0.4
samples = []

def train(noise_size, data_shape, batch_size, n_samples):
    losses=[]
    steps = 0

    real_img, noise_img = get_inputs(noise_size, data_shape[1], data_shape[2], data_shape[3])
    g_loss, d_loss = get_loss(real_img, noise_img, data_shape[-1])
    g_train_opt, d_train_opt, g_vars = get_optimizer(g_loss, d_loss, betal, learning_rate)

    saver = tf.train.Saver(var_list=g_vars)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for batch_i in range(mnist.train.num_examples//batch_size):
                steps +=1
                batch = mnist.train.next_batch(batch_size)
                batch_images = batch[0].reshape((batch_size, data_shape[1], data_shape[2], data_shape[3]))
                batch_images = batch_images*2 - 1

                # generator的输入噪声
                batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))

                #Run optimizers
                _ = sess.run(d_train_opt, feed_dict={real_img: batch_images, noise_img: batch_noise})
                _ = sess.run(g_train_opt, feed_dict={real_img: batch_images, noise_img:batch_noise})

            # 每一轮结束计算loss
            train_loss_d =  d_loss.eval({real_img: batch_images, noise_img:batch_noise})

            # generator loss
            train_loss_g = g_loss.eval({noise_img: batch_noise})

            losses.append((train_loss_d, train_loss_g))

            msg = "Epoch {}/{}".format(e+1, epochs), "判断器损失：{:.4f}".format(train_loss_d) ,"生成器损失:{:.4f}".format(train_loss_g)
            logger.info(msg)

            # 保存样本
            sample_noise = np.random.uniform(-1, 1, size=(n_samples, noise_size))
            gen_samples = sess.run(get_generator(noise_img, data_shape[-1],  is_train=False), feed_dict={noise_img: sample_noise})
            samples.append(gen_samples)

            saver.save(sess, './checkpoints_bat/generator_{}.ckpt'.format(e+1))
            saver.save(sess, './checkpoints/generator.ckpt')

    with open('dcgan_train_samples.pkl', 'wb') as f:
        pickle.dump(samples, f)
        print("save dcgan_train_samples.pkl")


def view_samples(epoch, samples):
    os.mkdir("image")

    for idx, img in enumerate(samples[epoch][1]):
        img = img.reshape((28,28))
        img = PIL.Image.fromarray(img, mode='L')

        img.save('image/{}-{}.jpg'.format(epoch,idx))
        # cv2.imwrite('image/{}-{}.jpg'.format(epoch,idx), img)

def test():
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
        sample_noise = np.random.uniform(-1, 1, size=(n_samples, noise_size))
        _, noise_img =  get_inputs(noise_size,28, 28,1)
        gen_samples = sess.run(get_generator(noise_img, 1, reuse=True ), feed_dict={noise_img:sample_noise})

        with open('dcgan_test_samples.pkl', 'wb') as f:
            pickle.dump([gen_samples], f)

if __name__ == '__main__':

    ##
    train(noise_size,[-1, 28, 28, 1], batch_size, n_samples)

    ## view the epoch result
    # with open('train_samples.pkl', 'rb') as f:
    #     samples = pickle.load(f)
    #
    # view_samples(50, samples)

    ## view the gan result
    # test()