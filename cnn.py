import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

"""임의의 3x3 이미지 생성"""
sess = tf.InteractiveSession()
image = np.array([[[[1],[2],[3]],
                   [[4],[5],[6]],
                   [[7],[8],[9]]]], dtype=np.float32)
print(image.shape)
plt.imshow(image.reshape(3,3), cmap='Greys')
plt.show()

"""'Valid' padding, 1 filter(2,2,1,1)"""
print("image.shape", image.shape)
weight = tf.constant([[[[1.]],[[1.]]],
                     [[[1.]],[[1.]]]])
print("weight.shape", weight.shape)
conv2d = tf.nn.conv2d(image, weight, strides=[1,1,1,1], padding='VALID')
conv2d_img = conv2d.eval()
print("conv2d_img.shape", conv2d_img.shape)
plt.imshow(conv2d_img.reshape(2,2), cmap='Greys')
plt.show()

"""'Same' padding, 3 filter(2,2,1,3)"""
print("image.shape", image.shape)
weight = tf.constant([[[[1.,10.,-1]],[[1.,10.,-1]]],
                     [[[1.,10.,-1]],[[1.,10.,-1]]]])
print("weight.shape", weight.shape)
conv2d = tf.nn.conv2d(image, weight, strides=[1,1,1,1], padding='SAME')
conv2d_img = conv2d.eval()
print("conv2d_img.shape", conv2d_img.shape)
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,3,i+1), plt.imshow(one_img.reshape(3,3), cmap='Greys')
plt.show()

"""MNIST 실습"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

"""이미지 출력"""
img = mnist.train.images[0].reshape(28,28)
plt.imshow(img, cmap="gray")
plt.show()

"""MNIST 데이터 CNN 연습"""
sess = tf.InteractiveSession()
img = img.reshape(-1, 28,28, 1)
W1 = tf.Variable(tf.random_normal([3,3,1,5], stddev=0.01))
conv2d = tf.nn.conv2d(img, W1, strides=[1,2,2,1], padding='SAME')
print(conv2d)
sess.run(tf.global_variables_initializer())
conv2d_img = conv2d.eval()
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    plt.subplot(1,5,i+1) , plt.imshow(one_img.reshape(14,14), cmap='gray')
plt.show()

"""Ksize"""
pool = tf.nn.max_pool(conv2d, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
print(pool)
sess.run(tf.global_variables_initializer())
pool_img = pool.eval()
pool_img = np.swapaxes(pool_img, 0 ,3)
for i, one_img in enumerate(pool_img):
    plt.subplot(1,5,i+1) , plt.imshow(one_img.reshape(7,7), cmap='gray')
plt.show()

