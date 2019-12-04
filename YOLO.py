import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
# https://blog.emmanuelcaradec.com/humble-yolo-implementation-in-keras/
# https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088#:~:targetText=YOLO%20uses%20sum%2Dsquared%20error,The%20loss%20function%20composes%20of%3A&targetText=the%20localization%20loss%20(errors%20between,the%20objectness%20of%20the%20box).

n_grid = 13
number_of_grids = n_grid*n_grid
number_of_anchors = 3
number_of_classes = 1
distance_of_grid = [i/n_grid for i in range(1,n_grid+1)]
single_grid_size = distance_of_grid[0]
anchor_boxes = [2.06253, 2.06253, 5.47434, 5.47434, 7.88282, 7.88282]
input_img_dim = (100,100,1)
n_epoch = 100

# [Pc, Mx, My, Hr, Wr, Pc1, Pc2, Pc3]
# Pc                 ----> Probabiblity that the grid contain  a class only if the midpoint of the object falls within the grid 
# (Mx,My)            ----> Midpoint of object within that grid wrt grid (always between 0 and 1)
# (Hr,Wr)            ---->  ratio of h & w of a object wrt the grid h & w
# Pc1, Pc2, Pc2, ... ----> one hot encoding

single_anchor_output_len = 1+4+number_of_classes
labels_dim = (n_grid, n_grid, number_of_anchors , single_anchor_output_len)
number_of_output_units = n_grid * n_grid * number_of_anchors * single_anchor_output_len

# [class_id, x_distance, y_distance, width, height]

# def data_prep():
#     dir = "S:\DL python\Yolo\data_set\\"
#     mdir = "S:\DL python\Yolo\modified_data_set\\"
#     images = os.listdir(dir)
#     count = 0
#     for img in images:
#         count+= 1
#         filename = str(count)+'.jpg' 
#         image = cv2.imread(dir+img, cv2.IMREAD_GRAYSCALE)
#         w,h= image.shape
#         resize_ratio = min([w/100,h/100])
#         w, h = int(w/resize_ratio), int(h/resize_ratio)
#         dim = (h, w)
#         image = cv2.resize(image,dim)[:100,-100:]
#         cv2.imwrite(mdir+filename,image)

def get_data():
    imgs_dir = '/content/sample_data/images//'
    label_dir = "/content/sample_data/labels//"
    label_files = os.listdir(label_dir)
    imgs_data = []
    labels = []
    for label_f in label_files:
      if label_f.endswith(".txt"):
        label = open(label_dir+label_f,'r').read()
        if len(label) >= 5:
            file_name = label_f.split('.')[0]+'.jpg'
            img = cv2.imread(imgs_dir+file_name, cv2.IMREAD_GRAYSCALE).reshape(input_img_dim)
            imgs_data.append(img)
            temp = [float(i) for i in label.split()]
            label = []
            for i in range(0,len(temp),5):
              label.append(temp[i:i+5])
            labels.append(label)

    return np.asarray(imgs_data), np.asarray(labels)

def image_meas_to_grid_meas(data):
  modified_data = []
  for d in data:
    count = previous = 0 
    for dist in distance_of_grid:
      if d > dist:
        previous = dist
      else:
        modified_data.append(count+(d - previous)/single_grid_size)
        break
      count += 1
  return modified_data

def get_modified_labels(labels):
    modified_label = []
    mask = []
    for label in labels:
      temp_label, temp_mask = create_output(label)
      modified_label.append(temp_label)
      mask.append(temp_mask)
    return np.asarray(modified_label).astype('float32'), np.asarray(mask).astype('float32')

def IOU(true_box, anchor):
  min_w, min_h = min(anchor[0], true_box[0]), min(anchor[1], true_box[1])
  intersection_area = min_w * min_h
  union_area = true_box[0]*true_box[1] + anchor[0]*anchor[1] - intersection_area
  return intersection_area/union_area

def create_output(actual_output):
  # cell_no, Pc, center_from_cell_origin , width_cell_ratio, height_cell_ratio
    output_value = np.zeros(labels_dim)
    mask_value = np.zeros(labels_dim)

    for output in actual_output:
        obj_id, center_x, center_y, width, height = output

        modified_x, modified_y  = image_meas_to_grid_meas([center_x, center_y])
        modified_w, modified_h = image_meas_to_grid_meas([width, height])

        IOU_s = []
        anchors = [[anchor_boxes[i], anchor_boxes[i+1]] for i in range(0,len(anchor_boxes),2)]

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
        Pcs[int(obj_id)] = 1 * max_IOU

        temp = [Pc, Mx, My, Hr, Wr, *Pcs]

        ith_anchor = anchor_for_img
        grid_x, grid_y = int(modified_x), int(modified_y)
        
        for i in range(number_of_anchors):
          mask_value[:,:,i,:] = 1

        for i in range(len(temp)):
          output_value[grid_x, grid_y, ith_anchor,i] = temp[i]
          mask_value[grid_x, grid_y, ith_anchor,i] = 1

    # [Pc, Mx, My, Hr, Wr, Pc1, Pc2, Pc3][Pc, Mx, My, Hr, Wr, Pc1, Pc2, Pc3][Pc, Mx, My, Hr, Wr, Pc1, Pc2, Pc3].....

    return (output_value,mask_value)

def loss_function(logits,label):
    # [Pc, Mx, My, Hr, Wr, Pc1, Pc2, Pc3][Pc, Mx, My, Hr, Wr, Pc1, Pc2, Pc3][Pc, Mx, My, Hr, Wr, Pc1, Pc2, Pc3]
    # mask to ignpre all the grids for which no object present ,ie there value is set to zeros expect for the Pc

      Pc = label[:,:,:,:,0]
      centroid = label[:,:,:,:,1:3]
      anchor_dim = label[:,:,:,:,3:5]
      Pclasses = label[:,:,:,:,5:]
      anchor_dim = tf.nn.relu(anchor_dim)

      Pc_hat = logits[:,:,:,:,0]
      centroid_hat = logits[:,:,:,:,1:3]
      anchor_dim_hat = logits[:,:,:,:,3:5]
      Pclasses_hat = logits[:,:,:,:,5:]
      anchor_dim_hat = tf.maximum(anchor_dim_hat, 0)


      Pc_error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= Pc_hat, labels= Pc))
      centroid_error = tf.reduce_mean(tf.squared_difference(centroid,centroid_hat))
      anchor_error = tf.reduce_mean(tf.squared_difference(tf.math.sqrt(anchor_dim),tf.math.sqrt(anchor_dim_hat))) 
      Pclasses_error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Pclasses, labels= Pclasses_hat))
      # Pclasses_error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Pclasses, labels= Pclasses_hat)) 

      error = Pc_error + 5.0* centroid_error + 5.0* anchor_error + Pclasses_error
        
      return tf.reduce_sum(error)

def output_to_bounding_box(output):
    bounding_boxes = []
    for r in range(n_grid):
      for c in range(n_grid):
        for b in range(number_of_anchors):
          if output[r,c,b,0] > 0.5:
            temp = {
                'grid_no': [r,c,b],
                 'output': output[r,c,b,:]
            }

    bounding_boxes.append(temp)
    return bounding_boxes
  
  def forward_pass(input_img, is_training=False):

  with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
    # 100x100x1
    conv1 = tf.layers.conv2d(input_img, filters=512, kernel_size= 5, strides= 2, padding= 'same')
    conv1 = tf.layers.batch_normalization(conv1, training=is_training)
    conv1 = tf.maximum(conv1, conv1*0.2)
    # 50x50x512
    print(conv1.shape)

    conv2 = tf.layers.conv2d(conv1, filters=256, kernel_size= 5, strides= 2, padding= 'same')
    conv2 = tf.layers.batch_normalization(conv2, training=is_training)
    conv2 = tf.maximum(conv2, conv2*0.2)
    # 25x25x256
    print(conv2.shape)

    conv3 = tf.layers.conv2d(conv2, filters=128, kernel_size= 5, strides= 2, padding= 'same')
    conv3 = tf.layers.batch_normalization(conv3, training=is_training)
    conv3 = tf.maximum(conv3, conv3*0.2)
    # 13x13x128
    print(conv3.shape)

    conv4 = tf.layers.conv2d(conv3, filters=64, kernel_size= 5, strides= 2, padding= 'same')
    conv4 = tf.layers.batch_normalization(conv4, training=is_training)
    conv4 = tf.maximum(conv4, conv4*0.2)
    # 7x7x64
    print(conv4.shape)

    conv4_reshape = tf.reshape(conv4, shape= (-1,7*7*64))
    print(conv4_reshape.shape)

    dense1 = tf.layers.dense(conv4_reshape, 4096)
    dense1 = tf.maximum(dense1, dense1*0.2)
    print(dense1.shape)

    logits = tf.layers.dense(dense1, number_of_output_units)
    logits = tf.reshape(logits,(-1,*(labels_dim)))

    return logits

def model(input_img, label, mask, is_training=True):

    logits = forward_pass(input_img, is_training=True)
    logits = logits * mask
    loss = loss_function(logits,label)
    train = tf.train.AdamOptimizer().minimize(loss)
    return (loss,train)

yolo_input_img = tf.placeholder(dtype=tf.float32, shape = (None, *(input_img_dim)))
yolo_label = tf.placeholder(dtype=tf.float32, shape= (None, *(labels_dim)))
yolo_mask = tf.placeholder(dtype=tf.float32, shape= (None, *(labels_dim)))

training = model(yolo_input_img, yolo_label, yolo_mask)
predict = forward_pass(yolo_input_img)

data_set, labels = get_data()
modified_labels, masks = get_modified_labels(labels)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for e in range(n_epoch):
    loss,_ = sess.run(training,feed_dict = {yolo_input_img:data_set, yolo_label:modified_labels, yolo_mask:masks})
    print(loss)
    
image = [data_set[1,:,:,:]]
output = sess.run(predict,feed_dict = {yolo_input_img:image})
output_to_bounding_box(output[0])
