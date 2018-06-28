# -*- coding: utf-8 -*-
# @Time    : 5/24/2018 6:06 PM
# @Author  : sunyonghai
# @File    : gan_mnist.py
# @Software: ZJ_AI
import PIL.Image
import tensorflow as tf
import numpy as np
import pickle
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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

def get_inputs(real_size, noise_size):
    real_img = tf.placeholder(tf.float32, [None, real_size])
    noise_img = tf.placeholder(tf.float32, [None, noise_size])

    return real_img, noise_img

# 生成器
def get_generator(noise_img, n_units, out_dim, reuse=False, alpha=0.01):
    """

    :param noise_img:  产生的噪声输入
    :param n_units:
    :param out_dim:
    :param reuse:
    :param alpha:
    :return:
    """
    with tf.variable_scope("generator", reuse=reuse):
        hidden1 = tf.layers.dense(noise_img, n_units)
        hidden1 = tf.maximum(alpha * hidden1, hidden1)
        hidden1 = tf.layers.dropout(hidden1, rate=0.2)

        logits = tf.layers.dense(hidden1, out_dim)
        outputs = tf.tanh(logits)

        return logits, outputs

def get_discriminator(img, n_units, reuse=False, alpha=0.01):
    """

    :param img: 输入
    :param n_units: 隐藏层单元数量
    :param reuse: 使用2次
    :param alpha:
    :return:
    """
    with tf.variable_scope("discriminator", reuse=reuse):
        hidden1 = tf.layers.dense(img, n_units)
        hidden1 = tf.maximum(alpha * hidden1, hidden1)

        logits = tf.layers.dense(hidden1, 1)
        outputs = tf.sigmoid(logits)

        return logits, outputs


img_size = mnist.train .images[0].shape[0]
noise_size = 100
g_units = 128
d_units = 128
learning_rate = 0.001
alpha = 0.01

# 构建网络

tf.reset_default_graph()

real_img, noise_img = get_inputs(img_size, noise_size)
g_logits,g_outputs = get_generator(noise_img, g_units, img_size)

d_logits_real, d_outputs_real = get_discriminator(real_img, d_units)
d_logits_fake, d_outputs_fake = get_discriminator(g_outputs, d_units, reuse=True)

# 目标函数

# discriminator 的loss [识别真实图片+ 识别生成图片]
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_logits_real)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_logits_fake)))

# 总的loss
d_loss = tf.add(d_loss_real, d_loss_fake)

# generator的loss
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)))

# 优化器
train_vars = tf.trainable_variables()
g_vars = [var for var in train_vars if var.name.startswith("generator")]
d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

# batch_size
batch_size = 64
epochs = 300
n_sample = 25
samples = []
losses = []

def train():
    saver = tf.train.Saver(var_list=g_vars)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for batch_i in range(mnist.train.num_examples//batch_size):
                batch = mnist.train.next_batch(batch_size)
                batch_images = batch[0].reshape((batch_size, 784))
                batch_images = batch_images*2 - 1

                # generator的输入噪声
                batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))

                #Run optimizers
                _ = sess.run(d_train_opt, feed_dict={real_img: batch_images, noise_img: batch_noise})
                _ = sess.run(g_train_opt, feed_dict={noise_img:batch_noise})

            # 每一轮结束计算loss
            train_loss_d = sess.run(d_loss, feed_dict={real_img: batch_images, noise_img:batch_noise})

            # real img loss
            train_loss_d_real = sess.run(d_loss_real, feed_dict={real_img: batch_images, noise_img:batch_noise})

            # fake img loss
            train_loss_d_fake = sess.run(d_loss_fake, feed_dict={real_img: batch_images, noise_img:batch_noise})

            # generator loss
            train_loss_g = sess.run(g_loss, feed_dict={noise_img: batch_noise})

            msg = "Epoch {}/{}".format(e+1, epochs), "判断器损失：{:.4f} (判断真实的:{:.4f} + 判断生成的:{:.4f})".format(train_loss_d, train_loss_d_real, train_loss_d_fake) ,"生成器损失:{:.4f}".format(train_loss_g)
            logger.info(msg)

            losses.append((train_loss_d, train_loss_d_real,train_loss_d_fake, train_loss_g))

            # 保存样本
            sample_noise = np.random.uniform(-1, 1, size=(n_sample, noise_size))
            gen_samples = sess.run(get_generator(noise_img, g_units, img_size, reuse=True), feed_dict={noise_img: sample_noise})
            samples.append(gen_samples)

            saver.save(sess, './checkpoints_bat/generator_{}.ckpt'.format(e+1))
            saver.save(sess, './checkpoints/generator.ckpt')

    with open('gan_train_samples.pkl', 'wb') as f:
        pickle.dump(samples, f)



def view_samples(epoch, samples):
    os.mkdir("image")

    for idx, img in enumerate(samples[epoch][1]):
        img = img.reshape((28,28))
        img = PIL.Image.fromarray(img, mode='L')

        img.save('image/{}-{}.jpg'.format(epoch,idx))
        # cv2.imwrite('image/{}-{}.jpg'.format(epoch,idx), img)

def test():
    saver = tf.train.Saver(var_list=g_vars)
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
        sample_noise = np.random.uniform(-1, 1, size=(25, noise_size))
        gen_samples = sess.run(get_generator(noise_img,g_units, img_size, reuse=True ), feed_dict={noise_img:sample_noise})

        with open('gan_test_samples.pkl', 'wb') as f:
            pickle.dump([gen_samples], f)

if __name__ == '__main__':

    ##
    train()

    ## view the epoch result
    # with open('train_samples.pkl', 'rb') as f:
    #     samples = pickle.load(f)
    #
    # view_samples(50, samples)

    ## view the gan result
    # test()