import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np



tf.disable_v2_behavior()



class LinearRegression:
    def __init__(self):
        pass

    @staticmethod
    def helloWorld():
        node1 = tf.constant(3.0, tf.float32)
        node2 = tf.constant(4.0)
        node3 = tf.add(node1, node2)
        print("node1:", node1, "node2:", node2)
        print("node3:", node3)
        sess = tf.Session()
        print(sess.run([node1, node2]))
        print(sess.run(node3))

    @staticmethod
    def placehold():
        a = tf.compat.v1.placeholder(tf.float32)
        b = tf.compat.v1.placeholder(tf.float32)
        adder_node = a + b
        sess = tf.Session()
        print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
        print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))

    @staticmethod
    def linear_regression():

        x_train = [1, 2, 3]
        y_train = [1, 2, 3]

        W = tf.Variable(tf.random.normal([1]), name='weight')
        b = tf.Variable(tf.random.normal([1]), name='bias')

        hypothesis = x_train * W + b

        cost = tf.reduce_mean(tf.square(hypothesis - y_train))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train = optimizer.minimize(cost)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for step in range(2001):
            sess.run(train)
            if step % 20 == 0:
                print(step, sess.run(cost), sess.run(W), sess.run(b))

    @staticmethod
    def linear_regression_placeholders():
        X = tf.placeholder(tf.float32, shape=[None])
        Y = tf.placeholder(tf.float32, shape=[None])
        W = tf.Variable(tf.random.normal([1]), name='weight')
        b = tf.Variable(tf.random.normal([1]), name='bias')
        hypothesis = X * W + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train = optimizer.minimize(cost)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for step in range(2001):
            cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
            feed_dict={X: [1, 2, 3, 4, 5], Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
            if step % 20 == 0:
                print(step, cost_val, W_val, b_val)


    @staticmethod
    def linear_regression_test():
        X = tf.placeholder(tf.float32, shape=[None])
        Y = tf.placeholder(tf.float32, shape=[None])
        W = tf.Variable(tf.random.normal([1]), name='weight')
        b = tf.Variable(tf.random.normal([1]), name='bias')
        hypothesis = X * W + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train = optimizer.minimize(cost)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for step in range(2001):
            cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
            feed_dict={X: [1, 2, 3, 4, 5], Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
            #if step % 20 == 0:
            #   print(step, cost_val, W_val, b_val)
        print(sess.run(hypothesis, feed_dict={X: [5]}))
        print(sess.run(hypothesis, feed_dict={X: [1.5, 3]}))


    @staticmethod
    def plt_of_cost_finction():
        X = [1, 2, 3]
        Y = [1, 2, 3]

        W = tf.placeholder(tf.float32)

        hypothesis = X * W

        cost = tf.reduce_mean(tf.square(hypothesis - Y))
        sess = tf.Session()

        sess.run(tf.global_variables_initializer())

        W_val = []
        cost_val = []
        for i in range(-30, 50):
            feed_W = i*0.1
            curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
            W_val.append(curr_W)
            cost_val.append(curr_cost)

        plt.plot(W_val, cost_val)
        plt.show()

        """
        cost function을 코딩한다면
        learning_rate = 0.1
        gradient  = tf.reduce_mean((W * X - Y) * X)
        descent = W - learning_rate * gradient
        update = W.assign(descent)
        """


    @staticmethod
    def cost_function():
        x_data = [1, 2, 3]
        y_data = [1, 2, 3]

        W = tf.Variable(tf.random_normal([1]), name='weight')
        X = tf.placeholder(tf.float32)
        Y = tf.placeholder(tf.float32)

        hypothesis = X * W

        cost = tf.reduce_sum(tf.square(hypothesis - Y))

        learning_rate = 0.1
        gradient = tf.reduce_mean((W * X - Y) * X)
        descent = W - learning_rate * gradient
        update = W.assign(descent)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for step in range(21):
            sess.run(update, feed_dict={X: x_data, Y: y_data})
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))


    @staticmethod
    def tf_cost_function():
        X = [1, 2, 3]
        Y = [1, 2, 3]

        W = tf.Variable(5.0)

        hypothesis = X * W

        cost = tf.reduce_mean(tf.square(hypothesis - Y))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        train = optimizer.minimize(cost)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for step in range(100):
            print(step, sess.run(W))
            sess.run(train)

    @staticmethod
    def compute_gradient():
        X = [1, 2, 3]
        Y = [1, 2, 3]

        W = tf.Variable(5.)

        hypothesis = X * W

        gradient = tf.reduce_mean((W * X - Y) * X * 2)

        cost = tf.reduce_mean(tf.square(hypothesis - Y))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

        gvs = optimizer.compute_gradients(cost)
        apply_gradients = optimizer.apply_gradients(gvs)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for step in range(100):
            print(step, sess.run([gradient, W, gvs]))
            sess.run(apply_gradients)


    @staticmethod
    def multi_variable_linear_regression():
        tf.set_random_seed(777)  # for reproducibility

        x1_data = [73., 93., 89., 96., 73.]
        x2_data = [80., 88., 91., 98., 66.]
        x3_data = [75., 93., 90., 100., 70.]

        y_data = [152., 185., 180., 196., 142.]

        x1 = tf.placeholder(tf.float32)
        x2 = tf.placeholder(tf.float32)
        x3 = tf.placeholder(tf.float32)

        Y = tf.placeholder(tf.float32)

        w1 = tf.Variable(tf.random_normal([1]), name='weight1')
        w2 = tf.Variable(tf.random_normal([1]), name='weight2')
        w3 = tf.Variable(tf.random_normal([1]), name='weight3')
        b = tf.Variable(tf.random_normal([1]), name='bias')

        hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

        cost = tf.reduce_mean(tf.square(hypothesis - Y))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
        train = optimizer.minimize(cost)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for step in range(2001):
            cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                           feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
            if step % 10 == 0:
                print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

    @staticmethod
    def matrix_linear_regression():
        x_data = [[73., 80., 75.],
                  [93., 88., 93.],
                  [89., 91., 90.],
                  [96., 98., 100.],
                  [73., 66., 70.]]
        y_data = [[152.],
                  [185.],
                  [180.],
                  [196.],
                  [142.]]

        X = tf.placeholder(tf.float32, shape=[None, 3])
        Y = tf.placeholder(tf.float32, shape=[None, 1])

        W = tf.Variable(tf.random_normal([3, 1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')

        hypothesis = tf.matmul(X, W) + b

        cost = tf.reduce_mean(tf.square(hypothesis - Y))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
        train = optimizer.minimize(cost)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for step in range(2001):
            cost_val, hy_val, _ = sess.run(
                [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
            if step % 10 == 0:
                print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

    @staticmethod
    def csv_data_linear_regression():
        xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
        x_data = xy[:, 0:-1]
        y_data = xy[:, [-1]]

        print(x_data, "\nx_data shape:", x_data.shape)
        print(y_data, "\ny_data shape:", y_data.shape)

        X = tf.placeholder(tf.float32, shape=[None, 3])
        Y = tf.placeholder(tf.float32, shape=[None, 1])

        W = tf.Variable(tf.random_normal([3, 1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')

        hypothesis = tf.matmul(X, W) + b

        cost = tf.reduce_mean(tf.square(hypothesis - Y))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
        train = optimizer.minimize(cost)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for step in range(2001):
            cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                           feed_dict={X: x_data, Y: y_data})
            if step % 10 == 0:
                print(step, "Cost:", cost_val, "\nPrediction:\n", hy_val)


        print("Your score will be ", sess.run(hypothesis,
                                              feed_dict={X: [[100, 70, 101]]}))

        print("Other scores will be ", sess.run(hypothesis,
                                                feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))

    @staticmethod
    def que_and_batch_linear_regression():
        filename_queue = tf.train.string_input_producer(
            ['data-01-test-score.csv'], shuffle=False, name='filename_queue')

        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)

        record_defaults = [[0.], [0.], [0.], [0.]]
        xy = tf.decode_csv(value, record_defaults=record_defaults)

        train_x_batch, train_y_batch = \
            tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

        X = tf.placeholder(tf.float32, shape=[None, 3])
        Y = tf.placeholder(tf.float32, shape=[None, 1])

        W = tf.Variable(tf.random_normal([3, 1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')

        hypothesis = tf.matmul(X, W) + b

        cost = tf.reduce_mean(tf.square(hypothesis - Y))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
        train = optimizer.minimize(cost)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(2001):
            x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
            cost_val, hy_val, _ = sess.run(
                [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
            if step % 10 == 0:
                print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

        coord.request_stop()
        coord.join(threads)

        print("Your score will be ",
              sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))

        print("Other scores will be ",
              sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))


if __name__ == '__main__':
    pass