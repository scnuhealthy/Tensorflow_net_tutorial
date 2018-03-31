# Author: Kemo HO
# This file is to train the model with the networks builded in captcha_cnn_model.py

import tensorflow as tf
from load_data import *
import captcha_params_and_cfg
import captcha_cnn_model
import os
from inception_v1 import *
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FLAGS = None


IMG_ROWS, IMG_COLS = captcha_params_and_cfg.get_height(), captcha_params_and_cfg.get_width()

batch_size = 32
LEN_Y = 2

x = tf.placeholder(tf.float32, [None,IMG_ROWS,IMG_COLS,1])
y_ = tf.placeholder(tf.float32, [None, LEN_Y])

# The ouput of the network
y_conv  = inception_v1(inputs = x,num_classes = 2,spatial_squeeze = True)
print('y_conv: ',y_conv.shape)

# For every sample, we reshape it to the size of [MAX_CAPTCHA,CHAR_SET_LEN], which means one row is for one predicted element
label = tf.reshape(y_,[-1,LEN_Y])
y_conv = tf.reshape(y_conv,[-1,LEN_Y])

# Define loss and optimizer
with tf.name_scope('loss'):
  loss = tf.square(y_conv-label)
loss = tf.reduce_mean(loss)

with tf.name_scope('adam_optimizer'):
  optimizer = tf.train.AdadeltaOptimizer(0.0001).minimize(loss)
'''
# Define accuracy
with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(y_conv, 2), tf.argmax(label, 2))
  correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)
'''

saver = tf.train.Saver(max_to_keep = 1)
save_model = captcha_params_and_cfg.save_model

data_files = []
for i in range(13):
    data_files.append("./tf_records/traindata.tfrecords-%.3d" % i)

print(data_files)

filename_queue = tf.train.string_input_producer(data_files)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
img_features = tf.parse_single_example(
                                        serialized_example,
                                        features={
                                               'label': tf.FixedLenFeature([], tf.string),
                                               'image_raw': tf.FixedLenFeature([], tf.string),
                                               'az': tf.FixedLenFeature([], tf.string),
                                               'el': tf.FixedLenFeature([], tf.string)
                                               })
image = tf.decode_raw(img_features['image_raw'], tf.float32)
image = tf.reshape(image, [IMG_ROWS, IMG_COLS,1])
label = tf.decode_raw(img_features['label'], tf.float32)
label = tf.reshape(label,[LEN_Y])
print(image.shape)
print(label.shape)
image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,capacity=1000)
print(image_batch.shape)
print(label_batch.shape)

'''
filename_queue = tf.train.string_input_producer(['./tf_records/test.records'])
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
img_features = tf.parse_single_example(
                                        serialized_example,
                                        features={
                                               'label': tf.FixedLenFeature([], tf.string),
                                               'image_raw': tf.FixedLenFeature([], tf.string),
                                               })
image_test = tf.decode_raw(img_features['image_raw'], tf.float32)
image_test = tf.reshape(image_test, [IMG_ROWS, IMG_COLS,1])
label_test = tf.decode_raw(img_features['label'], tf.uint8)
label_test = tf.reshape(label_test,[2])
print(image.shape)
print(label.shape)
image_batch_test, label_batch_test = tf.train.batch([image_test, label_test],
                                                batch_size= batch_size,num_threads = 64,capacity=1000)

'''
with tf.Session() as sess:
	
	# Initialize the parameters in the network
	sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	step = 0
	
	# load the model if it exists
	try:
		model_file=tf.train.latest_checkpoint(captcha_params_and_cfg.model_path)
		saver.restore(sess, model_file)
		print('loading model from %s' % model_file)
		step = int(model_file.split('-')[-1])+1
	except:
		pass
	
	# Training
	j = 0
	for i in range(200000):
            batch_xs,batch_ys = sess.run([image_batch,label_batch])	
            sess.run([optimizer,loss],feed_dict={x: batch_xs, y_: batch_ys})
            if j % 20 ==0:
                train_loss = loss.eval(feed_dict={
                    x: batch_xs, y_:batch_ys})
                print('step %d, training loss %g' % (step, train_loss))
				
                predict_result = sess.run(y_conv, feed_dict={x: batch_xs})
                print(batch_ys[:10])
                print(predict_result[:10])
				
                #batch_xs_test,batch_ys_test = sess.run([image_batch_test,label_batch_test])
                #test_loss = loss.eval(feed_dict={
                #    x: batch_xs_test, y_:batch_ys_test, keep_prob: 1.0})
                #print('step %d, test loss %g' % (step, test_loss))
            j +=1
            if j % 120 ==0:
                print('Saving model into %s-%d'% (save_model,step))
                saver.save(sess, save_model,global_step = step)
            step +=1
    
	coord.request_stop()
	coord.join(threads)



