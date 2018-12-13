import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500
BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99

import os
save_dir="path/to/model2/"
filename="model.ckpt"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def inference(input_tensor, weights1, biases1,weights2, biases2):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
    return tf.matmul(layer1, weights2) + biases2



def train(mnist):
    x_ = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x_input")
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y_input")
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))

    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    y=inference(x_,weights1,biases1,weights2,biases2)

    # y=tf.transpose(y)
    # global_step 变量不需要计算滑动平均值 所以trainable=False global_step代表的是i 也就是训练的次数
    # global_step: Optional `Variable` to increment by one after the
    #    variables have been updated.
    global_step = tf.Variable(0, trainable=False)  # 当前训练轮数
    # global_step = tf.Variable(0)
    #variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)  # global=num_updates
    # print(type(variable_averages))
    # 这里可训练的变量包括 weights biases
    #variables_averages_op = variable_averages.apply(tf.trainable_variables())  # 参数为一个列表
    #average_y = inference(x_, None, weights1, biases1, weights2, biases2)
    #
    ##cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,axis=1))
    ##cross_entropy_mean=tf.reduce_mean(cross_entropy)
    #
    #y1 = tf.nn.softmax(average_y)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, axis=1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 计算正则化损失
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    regularization = regularizer(weights1) + regularizer(weights2)
    # regularization = regularizer(variable_averages.average(weights1)) + \
    #                 regularizer(variable_averages.average(weights2))
    #   #计算总损失，减少过拟合问题
    loss = cross_entropy_mean + regularization

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean2)

    # 为了一次完成多个操作，tf提供了两个机制
    # tf.control_dependencies([op1,op2]) tf.group(op1,op2)
    train_op = tf.group( train_step)
    # 返回执行其所有输入的操作
    # with tf.control_dependencies([train_step,variables_averages_op]):
    #     train_op=tf.no_op(name="train")#表示执行完op之后什么操作都不做
    #
    ##correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        validate_feed = {x_: mnist.validation.images, y_: mnist.validation.labels}
        saver=tf.train.Saver()
        test_feed = {x_: mnist.test.images, y_: mnist.test.labels}
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s),validation accuracy on average model is %f" % (i, validate_acc))
                print("learning_rate: %s" % sess.run(learning_rate))
                print("global_step: %s" % sess.run(global_step))
                # print("weights1:\n %s\nweights2: \n%s\nbiases1: %s\nbiases2: %s" % (
                # weights1.eval()[:2, :2], weights2.eval()[:2, :2],
                # biases1.eval()[:2], biases2.eval()[:2]))

                #print("average_y", sess.run(y1, feed_dict={x_: xs})[:5, :])
                #print(ys[:5, :])

            sess.run(train_op, feed_dict={x_: xs, y_: ys})

        print("train over \n TEST")
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training steps(s) test accuracy using average model is %f" % (TRAINING_STEPS, test_acc))
        # print(tf.trainable_variables())
        saver.save(sess,save_dir+filename)

def main(argv=None):
    mnist = input_data.read_data_sets("data/", one_hot=True)
    print(mnist.train.images.shape)
    print(mnist.train.images.dtype)
    #print(mnist.train.images[0][:10])

    train(mnist)


if __name__ == "__main__":
    main()
































