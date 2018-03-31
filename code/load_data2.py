# Author:kemo
# This file is to load the data from the data set

import os
from PIL import Image
import numpy as np
import random
import captcha_params_and_cfg
import tensorflow as tf
np.random.seed(1337)



height = captcha_params_and_cfg.get_height()
width = captcha_params_and_cfg.get_width()


def int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_data(tol_num,train_num):
      
    # input,tol_num: the numbers of all samples(train and test)
    # input,train_num: the numbers of training samples
 
	# definite the data matrix
    data = np.empty((tol_num, 1,height, width),dtype="float32")
    label = np.empty((tol_num,2),dtype="float32")
	
    el = np.empty((tol_num),dtype="float32")
    az = np.empty((tol_num),dtype="float32")
    f = open(captcha_params_and_cfg.data_path+'/main.txt')
    
	# load the data
    for i in range(tol_num):
        
        line = f.readline()
        lineVec = line.split(" ")
        uv = [float(lineVec[2]),float(lineVec[3])]
        el[i] = float(lineVec[1])
        az[i] = float(lineVec[4])
        label[i] = uv
        img = get_image_from_file(captcha_params_and_cfg.data_path+'/'+lineVec[0])
        
        arr = np.asarray(img,dtype="float32")
        data[i,:,:,:] = arr
        if i%1000==0:
            print(i)
    f.close()
	
    # the data, shuffled and split between train and test sets
    rr = [i for i in range(tol_num)]
    random.shuffle(rr)
    X_train = data
    y_train = label
    az_train = az
    el_train = el
    #X_test = data[rr][train_num:]
    #y_test = label[rr][train_num:]	
	
    print(X_train[0])
    print(y_train[0])
    
	# save the data into tf_record
    recordfilenum = 0
    num = 0
    for i in range(train_num):       
        if num % 2000==0:
            tfrecordfilename = ("traindata.tfrecords-%.3d" % recordfilenum)
            writer = tf.python_io.TFRecordWriter('./tf_records/'+tfrecordfilename)
            recordfilenum +=1
        num = num + 1
        image_raw = X_train[i].tostring()
        label = y_train[i].tostring()
        el = el_train[i].tostring()
        az = az_train[i].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={'label':bytes_feature(label),'image_raw': bytes_feature(image_raw),'el':bytes_feature(el),'az':bytes_feature(az)}))
        writer.write(example.SerializeToString())
    writer.close()
    '''
    writer = tf.python_io.TFRecordWriter('./tf_records/test.records')
    for i in range(tol_num-train_num):
        
        image_raw = X_test[i].tostring()
        label = y_test[i].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={'label':bytes_feature(label),'image_raw': bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()
    '''
def get_image_from_file(path_img):
    img = Image.open(path_img)
    return pre_process_image(img)

def load_image(img):
    tol_num = 1
    data = np.empty((tol_num, 1, height, width),dtype="float32")

    img = pre_process_image(img)

    arr = np.asarray(img,dtype="float32")
    data[0,:,:,:] = arr
    return data

def pre_process_image(img):
    img = img.convert('L')
    # Resize it.
    img = img.resize((width, height), Image.BILINEAR)

    return img

load_data(25920,25920)



