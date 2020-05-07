import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np



tf.disable_v2_behavior()



class TraningAndTest:
    def __init__(self):
        pass

    @staticmethod
    def train_and_test():
        x_data = [[1, 2, 1],
                  [1, 3, 2],
                  [1, 3, 4],
                  [1, 5, 5],
                  [1, 7, 5],
                  [1, 2, 5],
                  [1, 6, 6],
                  [1, 7, 7]]
        y_data = [[0, 0, 1],
                  [0, 0, 1],
                  [0, 0, 1],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [1, 0, 0],
                  [1, 0, 0]]

        x_test = [[2, 1, 1],
                  [3, 1, 2],
                  [3, 3, 4]]
        y_test = [[0, 0, 1],
                  [0, 0, 1],
                  [0, 0, 1]]

        X = tf.placeholder("float", [None, 3])
        Y = tf.placeholder("float", [None, 3])

        W = tf.Variable(tf.random_normal([3, 3]))
        b = tf.Variable(tf.random_normal([3]))

        hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

        cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

        prediction = tf.argmax(hypothesis, 1)
        is_correct = tf.equal(prediction, tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for step in range(201):
                cost_val, W_val, _ = sess.run([cost, W, optimizer], feed_dict={X: x_data, Y: y_data})
                print(step, cost_val, W_val)

            print("Prediction:", sess.run(prediction, feed_dict={X: x_test}))
            print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))

    @staticmethod
    def normalization():

        xy = np.array(
            [
                [828.659973, 833.450012, 908100, 828.349976, 831.659973],
                [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
                [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
                [816, 820.958984, 1008100, 815.48999, 819.23999],
                [819.359985, 823, 1188100, 818.469971, 818.97998],
                [819, 823, 1198100, 816, 820.450012],
                [811.700012, 815.25, 1098100, 809.780029, 813.669983],
                [809.51001, 816.659973, 1398100, 804.539978, 809.559998],
            ]
        )

        xy = TraningAndTest.min_max_scaler(xy)
        print(xy)

        x_data = xy[:, 0:-1]
        y_data = xy[:, [-1]]

        # placeholders for a tensor that will be always fed.
        X = tf.placeholder(tf.float32, shape=[None, 4])
        Y = tf.placeholder(tf.float32, shape=[None, 1])

        W = tf.Variable(tf.random_normal([4, 1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')

        hypothesis = tf.matmul(X, W) + b

        cost = tf.reduce_mean(tf.square(hypothesis - Y))

        train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for step in range(101):
                _, cost_val, hy_val = sess.run(
                    [train, cost, hypothesis], feed_dict={X: x_data, Y: y_data}
                )
                print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

    @staticmethod
    def min_max_scaler(data):
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        # noise term prevents the zero division
        return numerator / (denominator + 1e-7)

if __name__ == '__main__':
    pass