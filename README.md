# 如何用tensorflow写出高级网络?一个套路就够了

------
![1](https://github.com/scnuhealthy/Tensorflow_net_tutorial/blob/master/imgs/tensorflow.jpg)
tensorflow，一个美好的深度学习框架，但是它的高门槛对beginer不太友好。虽然官网通过的“开始使用”MNIST的例子介绍了tensorflow的基础语法，但是这对于我们做学术研究的“武器”还远远不足。本文主要介绍使用tensorflow在**图像处理**领域的编程思路，由于保密关系，我不方便公开代码处理的数据，读者可以参考我github的另一个[project](https://github.com/scnuhealthy/Tensorflow_Captcha)，生成验证码实现分类任务。本文不会详细解释每行代码的作用，将集中于介绍tensorflow的**套路**，让读者掌握一个模版。在前人代码的基础上，你会发现，什么googlenet,resnet这种很深的网络也很容易实现。

对于一个**完整**的项目,tensorflow编程的流程是
> * 数据处理，生成tf_record
> * 明确输入输出，建立模型
> * 读取tf_record，编写损失函数和优化函数，训练模型
> * 使用模型预测


## 数据处理，生成tf_record

tf_record是tensorflow的一种能高效且能多线程处理数据的数据格式。我们实际训练网络时，能每次从中读取batch，并不需要把全部数据都写入内存。这样一种内存友好，性能友好，且能并行的数据格式，为何不充分利用呢？

数据的读取大致可以分为两部分，一是将图片和标签写入numpy，二是转化成tf——record形式。整个流程代码其实都很模版，np.asarray()将图片转为[height,width,channel]的数据格式。而tf_record的转化就更模式化了，和文件输出类似，定义输出文件，将数据to_string，用tf.train.example()注明每个样本的格式，最后写入文件。需要注意的，tf_record是需要分part的，每个record存储2000个样本，千万不要把用一个record存储全部样本！！！这样训练网络时相当于全部数据写入内存，而且失去了多线程的意义。
```python

def load_data(tol_num,train_num):
      
    # input,tol_num: the numbers of all samples(train and test)
    # input,train_num: the numbers of training samples
 
    ##################################
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
    
    ##################################
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
```
![2](https://github.com/scnuhealthy/Tensorflow_net_tutorial/blob/master/imgs/tf_records.png)
## 明确输入和输出，定义网络

一般而言，图像处理的图像和标签的格式是这样的：
channel是指图像的通道，channel=1是黑白图像，channel>=3是彩色图像，y_的维度就是输出的维度，如果是分类问题，LEN_Y=nb_classes。

```python
x = tf.placeholder(tf.float32, [None,IMG_ROWS,IMG_COLS,channel])
y_ = tf.placeholder(tf.float32, [None, LEN_Y])
```
只要模型的输入输出与上面你定义的相同，并且输入数据的格式也确实这样的话，网络就能run起来。先不管网络是多么花里胡哨，run起来才是首要的。通过输入x获取网络的输出y_conv:
```python
y_conv  = inception_v1(inputs = x,num_classes = 2,spatial_squeeze = True)
```
这样网络输出y_conv与标签y_都有了，我们后续就可以定义损失函数和优化函数。

实际上全部流程中最简单的就是网络的编写，因为别人帮已经帮我们写了很多很多。我这里用的是inception_v1的网络，可能很多人没听过inception，但是我说这就是googlenet，相信很多人都不会陌生。如果让我从头实现，估计几个月都写不出来，但是coder要学会参考。我直接从github的[tensorflow project](https://github.com/tensorflow/models/tree/master/research/slim/nets)上拷贝网络模型代码，核对输入输出，直接可用。这个链接上有很多使用tf.slim实现的高级网络，我认为使用它们比起我们从零构建有效得多。在这里我简单介绍一下这份inception的代码，这是一个全卷积网络，没有全连接层，最后网络会将图像卷积到[1,1,num_classes]的大小，通过tf.squeeze()去掉大小为1的维，最后每个样本都仅返回[num_classes]的预测标签。def inception_v1_base（）中我们可以自主选择模型的出口，这套模型深度很高，参数也很多，有时候我们并不需要如此巨大网络，这时候我们可以选择深度适中的出口，但注意的是我们需要在最后添加全连接层，并且注释掉tf.squeeze。

注意我的代码处理的是回归问题，所以模型最后没有softmax,如果是分类问题，需要在模型添加softmax且标签使用one-hot编码。

## 损失函数与优化函数
由于我处理的回归问题，所以我的损失函数是均方差，优化函数我用的Aadelta，tensorflow有很多实现好的损失函数与优化函数，读者需要借助官方文档选择最合适的，
```python
label = tf.reshape(y_,[-1,LEN_Y])
y_conv = tf.reshape(y_conv,[-1,LEN_Y])

# Define loss and optimizer
with tf.name_scope('loss'):
  loss = tf.square(y_conv-label)
loss = tf.reduce_mean(loss)

with tf.name_scope('adam_optimizer'):
  optimizer = tf.train.AdadeltaOptimizer(0.0001).minimize(loss)
```

## 训练网络
首先定义saver以用于后面保存模型,然后写好从tf_record生成batch的接口。读取tf_record的代码与生成tf_record的代码相呼应，同样是“固定搭配”的套路。
```python
# define the saver 
saver = tf.train.Saver(max_to_keep = 1)
save_model = captcha_params_and_cfg.save_model

data_files = []
for i in range(13):
    data_files.append("./tf_records/traindata.tfrecords-%.3d" % i)


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

image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,capacity=1000)
```

接着终于要使用Session了，每个定义好的tf变量将被赋予灵魂。
初始化变量：
```python
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
```
多线程读取batch语法：
```python
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)
# code here
coord.request_stop()
coord.join(threads)
```
使用损失函数和优化函数训练参数：
```python
batch_xs,batch_ys = sess.run([image_batch,label_batch])	
sess.run([optimizer,loss],feed_dict={x: batch_xs, y_: batch_ys})
```
训练途中查看损失和预测结果：
```python
train_loss = loss.eval(feed_dict={
                    x: batch_xs, y_:batch_ys})
print('step %d, training loss %g' % (step, train_loss))
predict_result = sess.run(y_conv, feed_dict={x: batch_xs})
print(batch_ys[:10])
print(predict_result[:10])
```
该部分完整版如下：
```python
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
```
## 使用模型预测
和训练模型类似，这里我就偷下懒，注意理解好sess.run的原理。
