import tensorflow as tf

def build_model(X):
    X_img = tf.reshape(X, shape=[-1, 28, 28, 1])

    X_img = tf.pad(X_img, [[0, 0], [2, 2], [2, 2], [0, 0]])
    X_img.get_shape()

    with tf.name_scope("conv1"):
        W1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=0.0, stddev=0.1))
        #QW1 = tf.fake_quant_with_min_max_args(W1, min=-0.3161901533603668, max=0.33857008814811707)
        W1_hist = tf.summary.histogram("W1", W1)
        b1 = tf.Variable(tf.zeros([6]))
        b1_hist = tf.summary.histogram("b1", b1)
        conv1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding="VALID")
        conv1 = tf.nn.bias_add(conv1, b1)
        conv1 = tf.nn.relu(conv1)
        conv1_hist = tf.summary.histogram("conv1", conv1)
        print(conv1.shape)

    with tf.name_scope("pool1"):
        pool_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        print(pool_1.shape)


    with tf.name_scope("conv2"):
        W2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=0.0, stddev=0.1))
        W2_hist = tf.summary.histogram("W2", W2)
        b2 = tf.Variable(tf.zeros([16]))
        b2_hist = tf.summary.histogram("b2", b2)
        conv2 = tf.nn.conv2d(pool_1, W2, strides=[1, 1, 1, 1], padding="VALID")
        conv2 = tf.nn.bias_add(conv2, b2)
        conv2 = tf.nn.relu(conv2)
        conv2_hist = tf.summary.histogram("conv2", conv2)
        print(conv2.shape)

    with tf.name_scope("pool2"):
        pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        print(pool_2.shape)

    with tf.name_scope("fc1"):
        flatten = tf.reshape(pool_2, shape=[-1, 5*5*16])
        W3 = tf.Variable(tf.truncated_normal(shape=[5*5*16, 120], mean=0.0, stddev=0.1))
        W3_hist = tf.summary.histogram("W3", W3)
        b3 = tf.Variable(tf.zeros([120]))
        b3_hist = tf.summary.histogram("b3", b3)
        fc1 = tf.matmul(flatten, W3) + b3
        fc1 = tf.nn.relu(fc1)
        fc1_hist = tf.summary.histogram("fc1", fc1)
        print(fc1.shape)


    with tf.name_scope("fc2"):
        W4 = tf.Variable(tf.truncated_normal(shape=[120, 84], mean=0.0, stddev=0.1))
        W4_hist = tf.summary.histogram("W4", W4)
        b4 = tf.Variable(tf.zeros([84]))
        b4_hist = tf.summary.histogram("b4", b4)
        fc2 = tf.matmul(fc1, W4) + b4
        fc2 = tf.nn.relu(fc2)
        fc2_hist = tf.summary.histogram("fc2", fc2)
        print(fc2.shape)

    with tf.name_scope("fc3"):
        W5 = tf.Variable(tf.truncated_normal(shape=[84, 10], mean=0.0, stddev=0.1))
        W5_hist = tf.summary.histogram("W5", W5)
        b5 = tf.Variable(tf.zeros([10]))
        b5_hist = tf.summary.histogram("b5", b5)
        logits = tf.matmul(fc2, W5) + b5

    
    summary = tf.summary.merge_all()
    return summary, logits
