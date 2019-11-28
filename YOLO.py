import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
# https://blog.emmanuelcaradec.com/humble-yolo-implementation-in-keras/
# https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088#:~:targetText=YOLO%20uses%20sum%2Dsquared%20error,The%20loss%20function%20composes%20of%3A&targetText=the%20localization%20loss%20(errors%20between,the%20objectness%20of%20the%20box).

n_grid = 13
number_of_grids = n_grid*n_grid
number_of_anchors = 3
number_of_classes = 3
distance_of_grid = [i/n_grid for i in range(1,n_grid+1)]
single_grid_size = distance_of_grid[0]
anchor_boxes = [1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778]
input_img_dim = (100,100,1)

# [Pc, Mx, My, Hr, Wr, Pc1, Pc2, Pc3]
# Pc                 ----> Probabiblity that the grid contain  a class only if the midpoint of the object falls within the grid 
# (Mx,My)            ----> Midpoint of object within that grid wrt grid (always between 0 and 1)
# (Hr,Wr)            ---->  ratio of h & w of a object wrt the grid h & w
# Pc1, Pc2, Pc2, ... ----> one hot encoding
labels_dim = (n_grid, n_grid, number_of_anchors * (1+4+number_of_classes))
number_of_output_units = n_grid * n_grid * number_of_anchors * (1+4+number_of_classes)

def IOU(true_box, anchor):
  min_w, min_h =np.min(anchor, true_box)
  intersection_area = min_w * min_h
  union_area = true_box[0]*true_box[1] + anchor[0]*anchor[1] - intersection_area
  return intersection_area/union_area

def image_meas_to_grid_meas(data):
  modified_data = []
  for d in data:
    count = previous = 0 
    for i in distance_of_grid:
      if d > dist:
        previous = dist
      else:
        modified_data.append(i+(d - previous)/single_grid_size)
        break
  return modified_data


def create_output(actual_output):
  # cell_no, Pc, center_from_cell_origin , width_cell_ratio, height_cell_ratio
    modified_output = []
    modified_anchors = image_meas_to_grid_meas(anchor_boxes)
    for output in actual_output:
      obj_id, center_x, center_y, width, height = output

      modified_data  = image_meas_to_grid_meas([center_x, center_y, width, height])
      modified_x, modified_y, modified_w, modified_h  = modified_data
      anchor_w, anchor_h = 0,0

      IOU_s = []
      anchors = [[moadifies_anchors[i], moadifies_anchors[i+1]] for i in range(0,len(moadifies_anchors),2)]
      for i in range(len(anchors)):
        anchor = anchors[i]
        true_box = [modified_w, modified_h]
        IOU_s.append(IOU(true_box, anchor))

      anchor_for_img = np.argmax(IOU_s)
      max_IOU = IOU_s[anchor_for_img]
      anchor_w, anchor_h = anchors[anchor_for_img]

      Pc = 1 * max_IOU #box coffidence score

      Mx = modified_x - int(modified_x)
      My = modified_y - int(modified_y)
      Hr = modified_h/anchor_h
      Wr = modified_w/anchor_w

      Pcs = [0]* number_of_classes
      Pcs[obj_id] = 1 * max_IOU

      temp = [Pc, Mx, My, Hr, Wr, *Pcs]


    # [Pc, Mx, My, Hr, Wr, Pc1, Pc2, Pc3][Pc, Mx, My, Hr, Wr, Pc1, Pc2, Pc3][Pc, Mx, My, Hr, Wr, Pc1, Pc2, Pc3]
    
   
def loss_function(logits,label):
    # [Pc, Mx, My, Hr, Wr, Pc1, Pc2, Pc3][Pc, Mx, My, Hr, Wr, Pc1, Pc2, Pc3][Pc, Mx, My, Hr, Wr, Pc1, Pc2, Pc3]

    # Cx, Cy =

    error = 0
    for grid in range(number_of_grids):
      for anchor in range(number_of_anchors):
        index = (number_of_anchors * (1+4+number_of_classes)) * grid + (1+4+number_of_classes) *anchor
        
        Pc = label[:,index+0]
        Mx = label[:,index+1]; My = label[:,index+2]
        Hr = label[:,index+3]; Wr = label[:,index+4];
        Pclasses = label[:,index+5:index+5+number_of_classes]
        centroid = [Mx, My]
        anchor_dim = [Hr, Wr]

        Pc_hat = logits[:,index+0]
        Mx_hat = logits[:,index+1]; My_hat = logits[:,index+2]
        Hr_hat = logits[:,index+3]; Wr_hat = logits[:,index+4]
        Pclasses_hat = logits[:,index+5:index+5+number_of_classes]
        centroid_hat = tf.sigmoid([Mx_hat, My_hat])
        anchor_dim_hat = tf.nn.relu([Hr_hat, Wr_hat])
 
        Pc_error = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits= Pc_hat, labels= Pc))
        centroid_error = tf.reduce_sum(tf.squared_difference(centroid,centroid_hat))
        anchor_error = tf.reduce_sum(tf.squared_difference(tf.math.sqrt(anchor_dim),tf.math.sqrt(anchor_dim_hat)))
        Pclasses_error = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=Pclasses, labels= Pclasses_hat))

        error += Pc_error + centroid_error + anchor_error + Pclasses_error
        
    return error
def model(input_img,label,is_training=True):

  with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
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
    dense1 = tf.maximum(dense1, dense1*0.2)

    logits = tf.layers.dense(dense1, number_of_output_units)
    logits = tf.reshape(logits,(-1,*(labels_dim)))

    loss = loss_function(logits,label)
    train = tf.train.AdamOptimizer().minimize(loss)

  return (loss,train) if is_training else logits

input_img = tf.placeholder(dtype=tf.float32, shape = (None, *(input_img_dim)))
label = tf.placeholder(dtype=tf.float32, shape= (None, *(labels_dim)))

training = model(input_img,label)
predict = model(input_img,label,is_training=False)}

