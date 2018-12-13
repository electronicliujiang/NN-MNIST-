import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import two_layers_mnist_model
import cv2
import numpy as  np
import os
#from 项目扩展.图片预处理.img2mat import *
INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500
BATCH_SIZE=100
save_dir="path/to/model2/"
#def evaluate(mnist):
def evaluate(mat,yy):
    with tf.Graph().as_default() as g:
        x_=tf.placeholder(tf.float32,
                          [None,two_layers_mnist_model.INPUT_NODE],
                          name="x-input")
        y_=tf.placeholder(tf.float32,
                          [None,two_layers_mnist_model.OUTPUT_NODE],
                          name="y-input")

        # validation_feed={x_:mnist.validation.images,
        #                  y_:mnist.validation.labels}
        #weights1=tf.Variable(tf.truncated_normal())

        validation_feed={x_:mat,y_:yy}
        weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
        biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
        weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))

        biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
        y=two_layers_mnist_model.inference(x_,weights1,biases1,weights2,biases2)

        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, axis=1))
        global_step = tf.Variable(0, trainable=False)
        saver=tf.train.Saver()

        with tf.Session() as sess:
            ckpt=tf.train.get_checkpoint_state(save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                #global_step=ckpt.model_checkpoint_path\
                #.split("/")[-1].split("-")[-1]
                #print(sess.run(y,{x_:mat}))
                print(sess.run(cross_entropy,validation_feed))
                #print(sess.run(loss))
                accuracy_score=sess.run(accuracy,
                                        feed_dict=validation_feed)
                print(sess.run(global_step))
                print("预测结果：",sess.run(tf.argmax(y,1),{x_:mat}))
                print("真实结果：",sess.run(tf.argmax(y_,1),{y_:yy}))
                print("After training steps,validation accuracy=%g "%
                      (accuracy_score))

            else:
                print("No checkpoint file found")
                return


def filenames(dir):
    L=[]
    num=[]
    for root,dirs,files in os.walk(dir):
        for file in files:
            L.append(os.path.join(root,file))
            num.append(int(file[3]))
    #print(L)
    #print(num)
    return L,num

def img2mat(dir):

    file_list,num=filenames(dir)
    mat = np.zeros((len(file_list), 784), dtype=np.float32)
    yy = np.zeros((len(file_list), 10), dtype=np.float32)
    for i,file in enumerate(file_list):
        img=cv2.imread(file)
        (b,g,r)=cv2.split(img)
        gray_img=0.299*r+0.587*g+0.114*b

        #print(g.shape)
        #g=g.reshape(28*28)
        gray_img =gray_img.reshape(28*28)
        mat[i]=gray_img

    for i,n in enumerate(num):
        yy[i,n]=1
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i,j]>125:
                mat[i, j]=1
            else:
                mat[i,j]=0

        #cv2.imshow("%s" % (i), mat[i].reshape(28, 28))

    return mat,yy

def kernel(n):
    return np.ones((n,n),np.uint8)
def img_dilate(mat):
    kernel1=kernel(2)
    new_mat=cv2.dilate(mat,kernel1)
    return new_mat

def get_data():
    mnist=input_data.read_data_sets("data/",one_hot=True)
    data1=mnist.train.images[:10]
    # for i in range(len(data1)):
    #     img=data1[i].reshape(28,28)
    #     cv2.imshow("%i"%i,img)
    y_label=mnist.train.labels[:10]
    print(data1[0].reshape(28,28))
    for i in range(10):
        #cv2.imwrite("project/preprocess/image2/pic%s.jpg"%i,data1[i].reshape(28,28))
        cv2.imshow("%i"%i,data1[i].reshape(28,28))
    print(y_label)
    return data1,y_label
def get_data_own():
    #save_dir = "project/preprocess/image/"
    save_dir = "project/preprocess/image3/"
    mat, yy = img2mat(save_dir)
    new_mats=np.zeros((mat.shape[0],784),np.float32)
    for i in range(mat.shape[0]):
        #cv2.imshow("%i" % i, mat[i].reshape(28, 28))
        new_mat=img_dilate(mat[i].reshape(28, 28))
        cv2.imshow("new %i" % i, new_mat)
        new_mats[i]=new_mat.reshape(28*28)
    print(yy)
    return new_mats,yy
def main():
    #mnist=input_data.read_data_sets("data/",one_hot=True)
    mat, yy=get_data_own()
    #print("my own data")
    #mat, yy=get_data()
    #print("standard data ")
    #save_dir = "project/preprocess/image/"
    #mat, yy = img2mat(save_dir)

    # x1=mnist.train.images[0].reshape(28,28)
    # print(x1)
    # cv2.imshow("hh",x1)


    #mat, yy=get_data_own()

    #mat //= 255
    #print(mat1.shape)
    #print(mat1[1].reshape(28,28))

    evaluate(mat,yy)
    cv2.waitKey(0)


if __name__=="__main__":
    main()


