import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np



tf.disable_v2_behavior()



class SoftmaxRegression:
    def __init__(self):
        pass

    @staticmethod
    def softmax_regression():
        x_data = [[1, 2, 1, 1],
                  [2, 1, 3, 2],
                  [3, 1, 3, 4],
                  [4, 1, 5, 5],
                  [1, 7, 5, 5],
                  [1, 2, 5, 6],
                  [1, 6, 6, 6],
                  [1, 7, 7, 7]]
        y_data = [[0, 0, 1],
                  [0, 0, 1],
                  [0, 0, 1],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [1, 0, 0],
                  [1, 0, 0]]

        X = tf.placeholder("float", [None, 4])
        Y = tf.placeholder("float", [None, 3])
        nb_classes = 3

        W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
        b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

        hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

        cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for step in range(2001):
                _, cost_val = sess.run([optimizer, cost], feed_dict={X: x_data, Y: y_data})

                if step % 200 == 0:
                    print(step, cost_val)

            print('--------------')
            a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
            print(a, sess.run(tf.argmax(a, 1)))

            print('--------------')
            b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4, 3]]})
            print(b, sess.run(tf.argmax(b, 1)))

            print('--------------')
            c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})
            print(c, sess.run(tf.argmax(c, 1)))

            print('--------------')
            all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
            print(all, sess.run(tf.argmax(all, 1)))

    @staticmethod
    def softmax_cross_entropy_with_logits():
        # Predicting animal type based on various features
        xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
        x_data = xy[:, 0:-1]
        y_data = xy[:, [-1]]

        print(x_data.shape, y_data.shape)

        nb_classes = 7  # 0 ~ 6

        X = tf.placeholder(tf.float32, [None, 16])
        Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6

        Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
        print("one_hot:", Y_one_hot)
        Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
        print("reshape one_hot:", Y_one_hot)


        W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
        b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

        logits = tf.matmul(X, W) + b
        hypothesis = tf.nn.softmax(logits)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                         labels=tf.stop_gradient([Y_one_hot])))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

        prediction = tf.argmax(hypothesis, 1)
        correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for step in range(2001):
                _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], feed_dict={X: x_data, Y: y_data})

                if step % 100 == 0:
                    print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))

            pred = sess.run(prediction, feed_dict={X: x_data})
            for p, y in zip(pred, y_data.flatten()):
                print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))


if __name__ == '__main__':
    SoftmaxRegression.soft()