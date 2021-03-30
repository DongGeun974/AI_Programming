import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA
import time

start = time.time()

"""one-hot encoding : 원하는 곳을 1로 채움-"""
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

"""알파 or 보폭, iteration : 학습 횟수, 배치 : n개 만큼 쪼갬"""
# parameters
learning_rate = 0.00000001
training_epochs = 10
batch_size = 1000

"""입력 dimension, 출력 dimension, None : 데이터의 양"""
# input place holders
Y = tf.placeholder(tf.float32, [None, 10])
X_train = tf.placeholder(tf.float32, [None, 4])

"""pca"""
pca = PCA(n_components=4)
train_data = pca.fit_transform(mnist.train.images)
test_data = pca.fit_transform(mnist.test.images)

"""shape=[입력 노드 수, 출력 노드 수], initializer : 초기화"""
# weights & bias for nn layers
# http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
"""1 hidden layer, sigmoid function"""
W1 = tf.get_variable("W1", shape=[4, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.sigmoid(tf.matmul(X_train, W1) + b1)
"""output node"""
W2 = tf.get_variable("W2", shape=[512, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([10]))
"""10개 노드, 각 숫자일 확률 벡터, 예측값"""
hypothesis = tf.matmul(L1, W2) + b2
"""cost function = (h-y)^2 """
# define cost/loss & optimizer
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
        batch_xs = train_data[i*batch_size : (i+1)*batch_size]
        """트레이닝 데이터 가져오기"""
        feed_dict = {X_train: batch_xs, Y: batch_ys}
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
      X_train: test_data, Y: mnist.test.labels}))

# r = random.randint(0, mnist.test.num_examples - 1)
# print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
# print("Prediction: ", sess.run(
#     tf.argmax(hypothesis, 1), feed_dict={X_train: train_data[r:r + 1]}))

print(round(time.time() - start, 2))