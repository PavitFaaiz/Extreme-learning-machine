import numpy as np
import tensorflow as tf

#Convert a label vector to a one-hot label matrix
def to_categorical(labels):
    num_classes = np.max(labels).astype(np.int32)+1
    res = np.zeros([len(labels), num_classes])
    for i in range(len(labels)):
        label = (labels[i]).astype(np.int32)
        res[i][label] = 1
    return res

#Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Logit function
def logit(x):
    return tf.log(x/(1-x))

#Generate a random orthogonal matrix of the specified shape
def random_weights(shape):
    s = max(shape)
    a = np.random.randn(s,s)
    q,r = np.linalg.qr(a)
    return q[0:shape[0], 0:shape[1]]

#Generate a random bias vector (with magnitude of 1) of the specified shape
def random_bias(shape):
    b = np.random.randn(shape[0],shape[1])
    norm_b = b/np.sqrt(np.sum(b**2))
    return norm_b

#Calculate regularized least square solution of matrix A
def regularized_ls(A, _lambda):
    shape = A.shape
    A_t = tf.transpose(A)
    if shape[0] < shape[1]:
        _A = tf.matmul(A_t, tf.matrix_inverse(_lambda * np.eye(shape[0]) + tf.matmul(A, A_t)))
    else:
        _A = tf.matmul(tf.matrix_inverse(_lambda * np.eye(shape[1]) + tf.matmul(A_t, A)), A_t)
    return _A