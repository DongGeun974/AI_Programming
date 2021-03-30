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
# print(conv2d)
sess.run(tf.global_variables_initializer())
conv2d_img = conv2d.eval()
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
# for i, one_img in enumerate(conv2d_img):
#     plt.subplot(1,5,i+1) , plt.imshow(one_img.reshape(14,14), cmap='gray')
# plt.show()

"""Ksize"""
pool = tf.nn.max_pool(conv2d, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
# print(pool)
sess.run(tf.global_variables_initializer())
# pool_img = pool.eval()
# pool_img = np.swapaxes(pool_img, 0 ,3)
# for i, one_img in enumerate(pool_img):
#     plt.subplot(1,5,i+1) , plt.imshow(one_img.reshape(7,7), cmap='gray')
# plt.show()

"""벡터화"""
L1_flat = tf.reshape(pool, [-1,7*7*5])
print(L1_flat)

"""dnn 적용"""

learning_rate = 0.001
training_epochs = 15
batch_size = 100

"""입력 dimension, 출력 dimension, None : 데이터의 양"""
# input place holders
X = tf.placeholder(tf.float32, [None, 7*7*5])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.get_variable("W1", shape=[7*7*5, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.get_variable("W2", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.get_variable("W3", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

W4 = tf.get_variable("W4", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)

W5 = tf.get_variable("W5", shape=[512, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))

hypothesis = tf.matmul(L4, W5) + b5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

"""시작, 초기화"""
# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

"""epoch만큼 학습"""
# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    """데이터/배치만큼 반복"""
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        """트레이닝 데이터 가져오기"""
        feed_dict = {X: batch_xs, Y: batch_ys}
        """실제로 학습, cost와 optimizer 추출"""
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

"""평가 테스트, hypothesis : 예측값, Y : 데이터, argmax : 제일 큰 값의 인덱스 리턴, accuracy : 정확도(1~0)"""
# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))