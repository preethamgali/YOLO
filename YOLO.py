import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

number_of_grids = 13*13
number_of_anchors = 3
number_of_classes = 3

input_img_dim = (100,100,1)

# [Pc, Mx, My, Hr, Wr, Pc1, Pc2, Pc3]
# Pc                 ----> Probabiblity that the grid contain  a class only if the midpoint of the object falls within the grid 
# (Mx,My)            ----> Midpoint of object within that grid wrt grid (always between 0 and 1)
# (Hr,Wr)            ---->  ratio of h & w of a object wrt the grid h & w
# Pc1, Pc2, Pc2, ... ----> one hot encoding
labels_dim = number_of_grids * number_of_anchors * (1+4+number_of_classes)

def loss_function(logits,label):
    # [Pc, Mx, My, Hr, Wr, Pc1, Pc2, Pc3]
    for grid in range(number_of_grids):
      for achor in range(number_of_anchors):
        index = grid+anchor
       
        Pc = label[index+0]; Mx= label[index+1]; My = label[index+2]; Hr = label[index+3]; Wr = label[index+4];
        Pclasses = [label[i] for i in range(index+5,index+number_of_classes)]
        boxes_hat = [Mx_hat, My_hat, Hr_hat, Wr_hat]
       
        Pc_hat = logits[index+0]; Mx_hat = logits[index+1]; My_hat = logits[index+2]; Hr_hat = logits[index+3]; Wr_hat = logits[index+4];
        Pclasses_hat = [logits[i] for i in range(index+5,index+number_of_classes)]
        boxes_hat = [Mx_hat, My_hat, Hr_hat, Wr_hat]

        Pc_error = tf.nn.sigmoid_cross_entropy_with_logits(logits= Pc_hat, labels=Pc)


    return

def model(is_training=True):
  input_img = tf.placeholder(dtype=tf.float32, shape = (None, *(input_img_dim)))
  label = tf.placeholder(dtype=tf.float32, shape= (None, labels_dim))
  # 100x100x1

  conv1 = tf.layers.conv2d(input_img, filters=512, kernel_size= 5, strides= 2, padding= 'same')
  conv1 = tf.layers.batch_normalization(conv1, training=is_training)
  conv1 = tf.maximum(conv1, conv1*0.2)
  # 50x50x512

  conv2 = tf.layers.conv2d(conv1, filters=256, kernel_size= 5, strides= 2, padding= 'same')
  conv2 = tf.layers.batch_normalization(conv2, training=is_training)
  conv2 = tf.maximum(conv2, conv2*0.2)
  # 25x25x256

  conv3 = tf.layers.conv2d(conv2, filters=128, kernel_size= 5, strides= 2, padding= 'same')
  conv3 = tf.layers.batch_normalization(conv3, training=is_training)
  conv3 = tf.maximum(conv3, conv3*0.2)
  # 13x13x128

  conv4 = tf.layers.conv2d(conv3, filters=64, kernel_size= 5, strides= 2, padding= 'same')
  conv4 = tf.layers.batch_normalization(conv4, training=is_training)
  conv4 = tf.maximum(conv4, conv4*0.2)
  # 7x7x64

  conv4_reshape = tf.reshape(conv4, shape= (-1,7*7*64))

  dense1 = tf.layers.dense(conv4_reshape, 4096)
  dense1 = tf.layers.batch_normalization(dense1, training=is_training)
  dense1 = tf.maximum(dense1, dense1*0.2)

  logits = tf.layers.dense(dense1, labels_dim)

  loss = loss_function(logits,label)

  train = tf.train.AdamOptimizer().minimize(loss)
  
  return (loss,train) if is_training else logits

