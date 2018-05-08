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

# Sorry, no gpu, no time, at the time!
def my_model(X,y,is_training,keep_prob,beta):
    # Setup Vars
    Wconv1 = tf.get_variable("Wconv1", shape=[3, 3, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    sconv1 = tf.get_variable("sconv1", shape=[30, 30, 32]) # bn1 scale param
    oconv1 = tf.get_variable("oconv1", shape=[30, 30, 32]) # bn1 offset param
    
    Wconv2 = tf.get_variable("Wconv2", shape=[3, 3, 32, 64])
    bconv2 = tf.get_variable("bconv2", shape=[64])
    sconv2 = tf.get_variable("sconv2", shape=[28, 28, 64]) # bn1 scale param
    oconv2 = tf.get_variable("oconv2", shape=[28, 28, 64]) # bn1 offset param
    
    Wconv3 = tf.get_variable("Wconv3", shape=[3, 3, 64, 128])
    bconv3 = tf.get_variable("bconv3", shape=[128])
    sconv3 = tf.get_variable("sconv3", shape=[12, 12, 128]) # bn1 scale param
    oconv3 = tf.get_variable("oconv3", shape=[12, 12, 128]) # bn1 offset param
    
    W1 = tf.get_variable("W1", shape=[12*12*128, 1024])
    b1 = tf.get_variable("b1", shape=[1024])
    s1 = tf.get_variable("s1", shape=[1024])
    o1 = tf.get_variable("o1", shape=[1024])
    W2 = tf.get_variable("W2", shape=[1024, 10])
    b2 = tf.get_variable("b2", shape=[10])
    
    # Setup Graph
    # conv1
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1,1,1,1], padding='VALID') + bconv1
    # bn_conv1
    mc1, vc1 = tf.nn.moments(a1, axes=[0], keep_dims=False)
    a2 = tf.nn.batch_normalization(a1, mc1, vc1, oconv1, sconv1, 1e-6)
    # relu_conv1
    a3 = tf.nn.relu(a2)
    # pool1
    #a4 = tf.nn.max_pool(a3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', data_format='NHWC')
    a4 = a3
    
    # conv2
    a5 = tf.nn.conv2d(a4, Wconv2, strides=[1,1,1,1], padding='VALID') + bconv2
    # bn_conv2
    mc2, vc2 = tf.nn.moments(a5, axes=[0], keep_dims=False)
    a6 = tf.nn.batch_normalization(a5, mc2, vc2, oconv2, sconv2, 1e-6)
    # relu_conv2
    a7 = tf.nn.relu(a6)
    # pool2
    a8 = tf.nn.max_pool(a7, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', data_format='NHWC')
    
    # conv3
    aa1 = tf.nn.conv2d(a8, Wconv3, strides=[1,1,1,1], padding='VALID') + bconv3
    # bn_conv3
    mc3, vc3 = tf.nn.moments(aa1, axes=[0], keep_dims=False)
    aa2 = tf.nn.batch_normalization(aa1, mc3, vc3, oconv3, sconv3, 1e-6)
    # relu_conv2
    aa3 = tf.nn.relu(aa2)
    
    # affine 1
    a9 = tf.reshape(aa3, [-1, 12*12*128])
    a10 = tf.matmul(a9, W1) + b1
    # bn1
    m1, v1 = tf.nn.moments(a10, axes=[0], keep_dims=False)
    a11 = tf.nn.batch_normalization(a10, m1, v1, o1, s1, 1e-6)
    # relu1
    a12 = tf.nn.relu(a11)
    # drop_out
    a13 = tf.nn.dropout(a12, keep_prob=keep_prob)
    # affine 2
    a14 = tf.matmul(a13, W2) + b2
    
    y_out = a14
    reg = tf.nn.l2_loss(Wconv1) + tf.nn.l2_loss(Wconv2) + tf.nn.l2_loss(Wconv3) + tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
    reg *= beta
    
    return y_out, reg

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)
beta = tf.placeholder(tf.float32)
global_step = tf.Variable(0, trainable=False)
starting_learning_rate = 1e-3

y_out, reg = my_model(X,y,is_training,keep_prob,beta)

mean_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(y,10), logits=y_out)) + reg

learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step,
                                           100, 0.95, staircase=True)

optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_step = optimizer.minimize(mean_loss,global_step=global_step)

# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)

def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False, kp= 0.5, regbeta= 1.0):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [loss_val,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))): # TODO: ceil?? Is this right?!
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now,
                         keep_prob: kp,
                         beta: regbeta,
                         global_step: iter_cnt
                        }
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

# Feel free to play with this cell
# This default code creates a session
# and trains your model for 10 epochs
# then prints the validation set accuracy
sess = tf.Session()

sess.run(tf.global_variables_initializer())
print('Training')
run_model(sess,y_out,mean_loss,X_train,y_train,5,64,100,train_step,True,regbeta=1e-3)
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
