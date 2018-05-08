import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[idx].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables,feed_dict=feed_dict)
            
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
              .format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss,total_correct
def my_model(X,y,is_training):
    
    conv1 = tf.layers.conv2d(
        inputs=X,
        filters=32,
        kernel_size=[7, 7],
        padding="SAME",
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        activity_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=32,
        kernel_size=[7, 7],
        padding="SAME",
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        activity_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    bn1 = tf.layers.batch_normalization(pool1,axis = 1,training = is_training)
    
    conv3 = tf.layers.conv2d(
        inputs=bn1,
        filters=32,
        kernel_size=[7, 7],
        padding="SAME",
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        activity_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=32,
        kernel_size=[7, 7],
        padding="SAME",
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        activity_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
    bn2 = tf.layers.batch_normalization(pool2,axis = 1,training = is_training)
    
    conv5 = tf.layers.conv2d(
        inputs=bn2,
        filters=32,
        kernel_size=[7, 7],
        padding="SAME",
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        activity_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=32,
        kernel_size=[7, 7],
        padding="SAME",
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        activity_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)
    bn3 = tf.layers.batch_normalization(pool3,axis = 1,training = is_training)
    
    
    flat = tf.reshape(bn3, [-1, 4*4*32])
    
    dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.5, training= is_training)
    y_out = tf.layers.dense(inputs=dropout, units=10)
   
    return y_out

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

y_out = my_model(X,y,is_training)
mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y, 10), logits=y_out))
optimizer = tf.train.AdamOptimizer()


# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)
# Feel free to play with this cell
# This default code creates a session
# and trains your model for 10 epochs
# then prints the validation set accuracy
sess = tf.Session()

sess.run(tf.global_variables_initializer())
print('Training')
run_model(sess,y_out,mean_loss,X_train,y_train,20,64,100,train_step,True)
print('Validation')
run_model(sess,y_out,mean_loss,X_val,y_val,1,64)
# Test your model here, and make sure 
# the output of this cell is the accuracy
# of your best model on the training and val sets
# We're looking for >= 70% accuracy on Validation
print('Training')
run_model(sess,y_out,mean_loss,X_train,y_train,1,64)
print('Validation')
run_model(sess,y_out,mean_loss,X_val,y_val,1,64)
print('Test')
run_model(sess,y_out,mean_loss,X_test,y_test,1,64)
