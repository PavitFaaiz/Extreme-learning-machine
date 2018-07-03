from utils import*

DTYPE = np.float64
TF_TYPE = None
if DTYPE == np.float64:
    TF_TYPE = tf.float64
else:
    TF_TYPE = tf.float32

#Traditional ELM class
class ELM:
    def __init__(self, input_dim, output_dim, num_hidden_node, gamma=None, W=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_node = num_hidden_node

        #If W is not given, generate it randomly
        if W is None:
            self.W = random_weights([self.input_dim, self.num_hidden_node]).astype(DTYPE)
            self.b = random_bias([1,self.num_hidden_node])
            self.W = np.append(self.W, self.b, axis=0)
        else:
            self.W = W
        self.H = None
        self.Beta = np.random.randn(self.num_hidden_node, self.output_dim)
        self.train_offset = 0
        self.gamma = gamma

    #Train the network
    def train(self, data_samples, labels):
        data_samples = np.array(data_samples, DTYPE)
        labels = labels.astype(DTYPE)
        self.W = self.W.astype(DTYPE)
        self.b = self.b.astype(DTYPE)

        #Define placeholders for train samples X and labels Y
        X = tf.placeholder(shape=[None, self.input_dim], dtype=TF_TYPE)
        Y = tf.placeholder(shape=[None, self.output_dim], dtype=TF_TYPE)
        X_aug = tf.concat([X, np.ones([len(data_samples), 1])], axis=1)
        H = tf.nn.sigmoid(tf.matmul(X_aug, self.W))
        if self.gamma is None:  # If no gamma then calculate least square solution using moore-penrose pseudoinverse
            h_inv = tf.py_func(np.linalg.pinv, [H], DTYPE)
        else:  # Else calculate a regularized least square solution
            h_inv = regularized_ls(H, self.gamma / (self.output_dim * self.num_hidden_node))
        Beta = tf.matmul(h_inv, Y)
        with tf.Session() as sess:
            self.Beta = sess.run(Beta, feed_dict={X:data_samples, Y:labels})

    #Evaluate the network
    def evaluate(self, test_samples, labels):
        res = self.predict(test_samples)
        correct = np.sum(np.argmax(res, axis=1) == np.argmax(labels, axis=1))
        return correct/len(labels)

    def predict(self, data_samples):
        data_samples = data_samples.astype(DTYPE)
        X = tf.placeholder(shape=[len(data_samples), self.input_dim], dtype=DTYPE)
        X_aug = tf.concat([X, np.ones([len(data_samples), 1])], axis=1)
        H = tf.nn.sigmoid(tf.matmul(X_aug, self.W))
        _res = tf.matmul(H,self.Beta)
        with tf.Session() as sess:
            res = sess.run(_res, feed_dict={X:data_samples})
        return res
